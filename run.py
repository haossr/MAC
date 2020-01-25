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
from models import MatroidModel
import time
import fire


def detect(split="val",
           root_path="sandbox",
           year=2012,
           gpu=False):

    m = MatroidModel("Everyday-Objects.matroid", gpu)
    voc_train = VOCDetection("~/data/voc/", image_set=split, download=True, year=str(year))
    #voc_train = VOCDetection("/deep/group/haosheng/voc/", image_set=split)i

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
            pred = m.predict(img)
            h, w = img.size
            for bbox, probs in pred:
                xmin, ymin, xmax, ymax = bbox[0]*h, bbox[1]*w, bbox[2]*h, bbox[3]*w
                name, confidence = sorted(probs, key=lambda x: x[1])[-1]
                f.write(f"{name} {confidence} {xmin:.0f} {ymin:.0f} {xmax:.0f} {ymax:.0f}\n")
            
def profile(gpu=False,
            n=100):
    m = MatroidModel("Everyday-Objects.matroid", gpu)
    voc_train = VOCDetection("/deep/group/haosheng/voc/", image_set="train")
    start = time.time()
    for i in tqdm(range(n)):
        img, target = voc_train[i]
        m.predict(img)
    end = time.time()
    print(f"Time used: {(end - start) / n:.2f} on {m.device}")

    
if __name__ == "__main__":
    fire.Fire()
 
