import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
import numpy as np


# settings of per_channel: per-channel=0 for weight (filter), per-channel=1 for activations (per channel).
# per-channel=4 to deactivate and do per-tensor quantization
def dsp_quant(input, bit, per_channel=4, model=None, max_range=60.):
    input = torch.clamp(input, min=-max_range, max=max_range)
    x = input.detach()
    enc_min, enc_max = compute_encodings(x, bit, per_channel)
    if model is not None:
        if model.training:
            model.a_enc_min = 0.2 * model.a_enc_min + 0.8 * enc_min
            model.a_enc_max = 0.2 * model.a_enc_max + 0.8 * enc_max

        enc_min = model.a_enc_min
        enc_max = model.a_enc_max

    scale = float(2 ** bit - 1)
    # quantize
    quantized = torch.round(scale * (x - enc_min) / (enc_max - enc_min))
    # dequantize
    step_size = (enc_max - enc_min) / scale
    dequantized_out = quantized * step_size + enc_min

    residual = dequantized_out - x

    return input + residual


def compute_encodings(data, bit, per_channel=4):
    if per_channel < 2:
        flat = data.reshape(data.size(per_channel), -1)
        enc_min, _ = torch.min(flat, dim=1, )
        enc_max, _ = torch.max(flat, dim=1, )
        if per_channel == 1:
            enc_min = enc_min.reshape(1, data.size(1), 1, 1)
            enc_max = enc_max.reshape(1, data.size(1), 1, 1)
        elif per_channel == 0:
            enc_min = enc_min.reshape(data.size(0), 1, 1, 1)
            enc_max = enc_max.reshape(data.size(0), 1, 1, 1)
        else:
            raise NotImplementedError
    else:
        enc_min = torch.min(data)
        enc_max = torch.max(data)

    if per_channel < 2:
        enc_max = torch.maximum(enc_max, enc_min + 0.01)
    else:
        enc_max = torch.cat((enc_max.reshape(1), enc_min.reshape(1) + 0.01))
        enc_max = torch.max(enc_max)

    if torch.max(enc_min) > 0 or torch.min(enc_max) < 0:
        if per_channel < 2:
            enc_min = torch.minimum(enc_min, torch.zeros(enc_min.size()).to(enc_min.device))
            enc_max = torch.maximum(enc_max, torch.zeros(enc_max.size()).to(enc_max.device))
        else:
            enc_min = torch.cat((enc_min.reshape(1), torch.zeros(1).to(enc_max.device)))
            enc_max = torch.cat((enc_max.reshape(1), torch.zeros(1).to(enc_max.device)))
            enc_min = torch.min(enc_min)
            enc_max = torch.max(enc_max)
    else:
        scale = float(2 ** bit - 1)
        nearest = scale * torch.abs(enc_min) / (enc_max - enc_min)
        shift = (torch.round(nearest) - nearest) * (enc_max - enc_min) / scale
        enc_min -= shift
        enc_max -= shift

    return enc_min, enc_max


class Quantize(nn.Module):
    """docstring for QuanConv"""

    def __init__(self, bit, per_channel, max_range=60.):
        super(Quantize, self).__init__()

        self.nbit_a = bit
        self.per_channel_a = per_channel
        self.max_range = max_range
        self.register_buffer('a_enc_min', torch.tensor(0.))
        self.register_buffer('a_enc_max', torch.tensor(0.))

    # @weak_script_method
    def forward(self, x):
        if self.training:
            out = dsp_quant(x, self.nbit_a, per_channel=self.per_channel_a, model=self, max_range=self.max_range)
        else:
            out = torch.clamp(x, min=self.a_enc_min, max=self.a_enc_max)
        return out


class Add(nn.Module):
    """docstring for QuanConv"""

    def __init__(self, bit, per_channel):
        super(Add, self).__init__()

        self.nbit_a = bit
        self.per_channel_a = per_channel
        self.register_buffer('a_enc_min', torch.tensor(0.))
        self.register_buffer('a_enc_max', torch.tensor(0.))

    # @weak_script_method
    def forward(self, x, y):
        if self.training:
            total = torch.cat((x, y), dim=1)
            out = dsp_quant(total, self.nbit_a, per_channel=self.per_channel_a, model=self)
            out1, out2 = torch.chunk(out, 2, 1)
        else:
            out1 = torch.clamp(x, min=self.a_enc_min, max=self.a_enc_max)
            out2 = torch.clamp(y, min=self.a_enc_min, max=self.a_enc_max)
        return out1 + out2


# per_channel = 'all', 'w', 'a', 'none'
class QConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 nbit_w=8, nbit_a=8, per_channel='no'):
        super(QConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.per_channel_w = 4
        self.per_channel_a = 4
        if per_channel == 'all' or per_channel == 'w':
            self.per_channel_w = 0
        if per_channel == 'all' or per_channel == 'a':
            self.per_channel_a = 1

        self.register_buffer('a_enc_min', torch.tensor(0.))
        self.register_buffer('a_enc_max', torch.tensor(0.))

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32 and self.training:
            w = dsp_quant(self.weight, self.nbit_w, per_channel=self.per_channel_w)
        else:
            w = self.weight

        if self.nbit_a < 32 and self.training:
            x = dsp_quant(input, self.nbit_a, per_channel=self.per_channel_a, model=self)
        else:
            x = torch.clamp(input, min=self.a_enc_min, max=self.a_enc_max)
            # x = input

        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


# FC linear won't support per channel quantization
class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 nbit_w=8, nbit_a=8, ):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.nbit_w = nbit_w
        self.nbit_a = nbit_a

        self.register_buffer('a_enc_min', torch.tensor(0.))
        self.register_buffer('a_enc_max', torch.tensor(0.))

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32 and self.training:
            w = dsp_quant(self.weight, self.nbit_w, per_channel=4)
        else:
            w = self.weight

        if self.nbit_a < 32 and self.training:
            x = dsp_quant(input, self.nbit_a, per_channel=4,
                          model=self)
        else:
            x = torch.clamp(input, min=self.a_enc_min, max=self.a_enc_max)
            # x = input

        output = F.linear(x, w, self.bias)
        return output


# usage: replace all 'conv + bn' block with this module to perform BN folding aware quantization
# per_channel = 'all', 'w', 'a', 'none'
class ConvBN2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, nbit_w=8, nbit_a=8, per_channel='no'):
        super(ConvBN2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.per_channel_w = 4
        self.per_channel_a = 4
        if per_channel == 'all' or per_channel == 'w':
            self.per_channel_w = 0
        if per_channel == 'all' or per_channel == 'a':
            self.per_channel_a = 1

        self.register_buffer('a_enc_min', torch.tensor(0.))
        self.register_buffer('a_enc_max', torch.tensor(0.))

    def forward(self, x):
        term = self.bn.weight.detach().div(torch.sqrt(self.bn.eps + self.bn.running_var)).reshape(-1, 1, 1, 1)
        sign = torch.sign(term)
        processed_term = torch.clamp(torch.abs(term), min=1e-5) * sign

        if self.training:
            fused_weight = self.conv.weight.detach().mul(processed_term)
            if self.nbit_w < 32:
                fused_weight = dsp_quant(fused_weight.detach(), self.nbit_w, per_channel=self.per_channel_w)
                reverse_weight = fused_weight.div(processed_term)
                residual_weight = reverse_weight - self.conv.weight.detach()
            else:
                residual_weight = 0.

            if self.nbit_a < 32:
                x = dsp_quant(x, self.nbit_a, per_channel=self.per_channel_a,
                              model=self)

            return self.bn(F.conv2d(x, self.conv.weight + residual_weight, self.conv.bias, self.conv.stride,
                                    self.conv.padding, self.conv.dilation, self.conv.groups))
        else:
            fused_weight = self.conv.weight.detach().mul(processed_term)
            fused_bias = self.bn.bias.detach() - self.bn.weight.detach().mul(self.bn.running_mean).div(
                torch.sqrt(self.bn.running_var + self.bn.eps))

            if self.nbit_w < 32:
                fused_weight = dsp_quant(fused_weight, self.nbit_w, per_channel=self.per_channel_w)

            if self.nbit_a < 32:
                x = dsp_quant(x, self.nbit_a, per_channel=self.per_channel_a,
                              model=self)

            return F.conv2d(x, fused_weight, fused_bias, self.conv.stride, self.conv.padding,
                            self.conv.dilation, self.conv.groups)

#
# def test():
#     ts = torch.randn(4, 8, 32, 32)
#     # tnew = torch.randn(1, 3, 32, 32)
#     # net = torch.nn.utils.weight_norm(QConv(3, 10, 3, stride=1, padding=1))
#     net = ConvBN2d(8, 8, 1, 1, 0, )
#
#     # net.eval()
#     # with torch.no_grad():
#     #     out1 = net(ts)
#     # out2 = net.inference(tnew)
#
#     net.eval()
#     ot = net(ts)
#     out1 = net.inference(ts)
#
#     print(torch.max(torch.abs(out1 - ot)))
#     print(out1.size())

# test()
# enc_min, enc_max = compute_encodings(torch.tensor([-1.8, -1.0, 0., 0.5]), bit=8)
# out = dsp_quant(torch.tensor([-1.8, -1.0, 0.1, 0.5]), 8)
# print(out)

# y, _ = torch.min(x, dim=0)
# print(y)
# print(y.size())
