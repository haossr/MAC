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


class MatroidModel:
    TARGET_DIR = '~/tmp/matroid'
    def __init__(self, model_file, gpu=False):
        self.model_file = model_file
        self.load_matroid()
        self.device = "/device:cpu:*"
        if gpu:
            self.device = "/device:gpu:*"
        self.create_tf_graph()
    
    def load_matroid(self):
        model_file = self.model_file

        # Unzip file 
        if os.path.exists(self.TARGET_DIR):
            shutil.rmtree(self.TARGET_DIR)
        os.makedirs(self.TARGET_DIR)
        with zipfile.ZipFile(model_file, 'r') as zip_ref:
            zip_ref.extractall(self.TARGET_DIR)
        for file in os.listdir(self.TARGET_DIR):
            if file.endswith('.pb'):
                self.pb_file = os.path.join(self.TARGET_DIR, file)
            elif file.endswith('.json'):
                self.json_file = os.path.join(self.TARGET_DIR, file)

        assert(self.pb_file and self.json_file)
        
        # Load meta data
        self.meta = json.loads(open(self.json_file).read())

    def create_tf_graph(self):
        # Load model
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                                     log_device_placement=True))
        with tf.device(self.device):
            with open(self.pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                ops = tf.import_graph_def(graph_def, name='')

        self.output_tensor = self.sess.graph.get_tensor_by_name(self.meta['outputs'][0]['name'] + ':0')
        self.input_tensor = self.sess.graph.get_tensor_by_name(self.meta['inputs'][0]['name'] + ':0')
    
    def predict(self, img):
        # Process image
        data = [np.array(img.resize(tuple(self.meta['inputs'][0]['tensor_shape'][:2]), Image.BILINEAR))]
        
        # Run model
        output = self.sess.run(self.output_tensor, {self.input_tensor: np.array(data)})
        
        # Process output
        if self.meta['detector_type'] == 'object_localization':
            boxes, probs = [], []
            if output[0, 0, 0] < 0:
                return np.array([]), np.array([])
                #print('no object detected')
            for j in range(output.shape[1]):
                # a valid detection
                if output[0, j, 0] >= 0.0:
                    boxes.append(output[0, j, :4])
                    prob = list(output[0, j, 4:])
                    probs.append(prob)
        return np.array(boxes), np.array(probs)
