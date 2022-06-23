import numpy as np
import torch
from utils import torch_utils
from models.yolo import Model

def test_sparsity(model):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)

    print("ONLY consider CONV layers: ")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")
    return comp_ratio



if __name__ == '__main__':
    print("Check Dense model: ")
    device = 'cuda'
    # weights = './runs/train/exp18/weights/last.pt'
    weights = 'yolov5s.pt'
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    cfg = './models/yolov5s.yaml'
    nc = 80
    img_size = 320

    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    model.load_state_dict(csd, strict=False)  # load
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    torch_utils.model_info(model, img_size=img_size)
    #
    # print("Check 4x prunned model: ")
    # weights = './runs/train/exp70/last.pt'
    # ckpt = torch.load(weights, map_location=device)  # load checkpoint
    # model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    # model.load_state_dict(csd, strict=False)  # load
    # torch_utils.prunned_model_info(model, img_size=img_size)
    # test_sparsity(model)