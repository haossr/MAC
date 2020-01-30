from torchvision.datasets import CocoDetection, VOCDetection
import os
import tensorflow as tf
import sys
import numpy as np
from PIL import Image
import zipfile
import shutil
import json
from tqdm import tqdm
from thop import clever_format
from thop import profile
import time
import fire

from models import MatroidModel, EfficientDet, EFFICIENTDET
from data import VOC
from util import nms
from util.constants import *


def detect(split="val",
           root_path="sandbox",
           year=2012,
           gpu=True):

    m = MatroidModel("matroid/Everyday-Objects.matroid", gpu)
    voc_train = VOCDetection("~/data/voc/",
                             image_set=split,
                             download=True,
                             year=str(year))
    # voc_train = VOCDetection("/deep/group/haosheng/voc/", image_set=split)i

    GROUNDTRUTH_PATH = os.path.join("Object-Detection-Metrics", "groundtruths")
    PREDICTION_PATH = os.path.join("Object-Detection-Metrics", "detections")
    os.makedirs(GROUNDTRUTH_PATH, exist_ok=True)
    os.makedirs(PREDICTION_PATH, exist_ok=True)

    for img, target in tqdm(voc_train):
        # Ground Truth
        file_name = target['annotation']['filename'].replace("jpg", "txt")
        with open(os.path.join(GROUNDTRUTH_PATH, file_name), "w") as f:
            objs = target['annotation']['object']
            if not isinstance(objs, list):
                objs = [objs]
            for obj in objs:
                name = obj['name']
                bbox = obj['bndbox']
                xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                f.write(f"{name} {xmin} {ymin} {xmax} {ymax}\n")

        # Prediction
        with open(os.path.join(PREDICTION_PATH, file_name), "w") as f:
            boxes, probs = m.predict(img)
            preds = nms(boxes, probs)
            
            h, w = img.size
            for label, confidence, bbox in preds:
                xmin, ymin, xmax, ymax = bbox[0] * \
                    h, bbox[1]*w, bbox[2]*h, bbox[3]*w
                name = VOC_LABEL2NAME[label]
                f.write(
                    f"{name} {confidence} {xmin:.0f} {ymin:.0f} {xmax:.0f} {ymax:.0f}\n")


def profileMD(gpu=True,
            split="val",
            year=2012,
            n=100):
    m = MatroidModel("matroid/Everyday-Objects.matroid", gpu)
    voc_train = VOCDetection(
        "~/data/voc/", image_set=split, download=True, year=str(year))
    time_used = 0
    for i in tqdm(range(n)):
        img, _ = voc_train[i]
        start = time.time()
        m.predict(img)
        end = time.time()
        time_used += (end-start)
    print(f"Time used: {time_used / n * 1000:.2f}(ms) on [{m.device}]")


def profileED(gpu=True,
              split="val",
              network="efficientdet-d0",
              year=2012,
              n=100):
    m = EfficientDet(num_classes=20,
                     network=network,
                     W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                     D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                     D_class=EFFICIENTDET[network]['D_class'])
    device = "GPU" if gpu else "CPU"
    
    dataset = VOC(root="data/voc/", split="train")
    img = dataset[0]['img'].unsqueeze(0)
    annot = dataset[0]['annot'].unsqueeze(0)
    if gpu: 
        m = m.cuda()
        img = img.cuda()
        annot = annot.cuda()

    # Count MACs
    macs, params = profile(m, inputs=([img, annot], ))
    macs, params = clever_format([macs, params], "%.3f")

    # Profile 
    time_used = 0
    for i in tqdm(range(n)):
        img = dataset[i]['img']
        if gpu:
            img = img.cuda()
        start = time.time()
        m.detect(img.unsqueeze(0))
        end = time.time()
        time_used += (end-start)
    print("="*40)
    print(f"MACs: {macs}; # of params: {params}")
    print(f"Time used: {time_used / n * 1000:.2f}(ms) on [{device}]")


if __name__ == "__main__":
    fire.Fire()
