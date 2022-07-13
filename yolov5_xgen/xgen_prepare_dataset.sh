#!/bin/bash

data_path="/data/object-detection-yolov5"

# 进入到数据存放目录并下载数据
mkdir ${data_path} && cd ${data_path}
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/object-detection-yolov5/train2017.zip
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/object-detection-yolov5/val2017.zip
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/object-detection-yolov5/coco2017labels.zip
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/object-detection-yolov5/coco128.zip

# 如果缓存目录不存在则进行数据准备
cd ${data_path}
if "folder1" and "folder2" 不存在：
    解压 train2017.zip 
    解压 val2017.zip 
    解压 coco2017labels.zip
