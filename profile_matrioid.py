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

def profile(gpu=False):
    m = MatroidModel("Everyday-Objects.matroid", gpu)
    voc_train = VOCDetection("/deep/group/haosheng/voc/", image_set="train")
    start = time.time()
    for i in tqdm(range(100)):
        img, target = voc_train[i]
        m.predict(img)
    end = time.time()
    print(f"Time used: {(end - start) / 100:.2f} on {m.device}")

    
if __name__ == "__main__":
    fire.Fire()
 