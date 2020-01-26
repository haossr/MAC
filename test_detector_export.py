"""Run exported .matroid model for the given images"""
import os
import tensorflow as tf
import sys
import numpy as np
from PIL import Image
import zipfile
import shutil
import json

def create_tf_graph(sess, model_path):
  with open(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def detect_faces(gray_img):
  import cv2
  data = '/tmp/haarcascade_frontalface_default.xml'
  if not os.path.exists(data):
    downloaded = os.system('wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O /tmp/haarcascade_frontalface_default.xml')
    if downloaded != 0:
      raise Exception('Cannot download face detector')

  haar_face_cascade = cv2.CascadeClassifier(data)
  faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
  return faces # x,y,w,h

targetdir = '/tmp/test_detector_export'
if os.path.exists(targetdir):
  shutil.rmtree(targetdir)
os.makedirs(targetdir)

model_file = sys.argv[1]
assert(model_file.endswith('.matroid'))
images = sys.argv[2:]

with zipfile.ZipFile(model_file, 'r') as zip_ref:
  zip_ref.extractall(targetdir)

pb_file = None
json_file = None
for file in os.listdir(targetdir):
  if file.endswith('.pb'):
    pb_file = os.path.join(targetdir, file)
  elif file.endswith('.json'):
    json_file = os.path.join(targetdir, file)

assert(pb_file and json_file)
meta = json.loads(open(json_file).read())

sess = tf.Session()
create_tf_graph(sess, pb_file)

data = []
faces = [] # list of (filename, bbox)
for index, im in enumerate(images):
  img = Image.open(im)
  if meta['detector_type'] == 'facial_recognition' or meta['detector_type'] == 'facial_characteristics':
    bboxes = detect_faces(np.array(img.convert('L')))
    if len(bboxes) == 0:
      print(im, 'no face detected')
    for index2, f in enumerate(bboxes):
      faces.append((images[index], f))
      face = img.crop((f[0], f[1], f[2]+f[0], f[3]+f[1]))
      face.save('/tmp/face-%s-%s.jpg' % (index, index2))
      face = face.resize(tuple(meta['inputs'][0]['tensor_shape'][:2]), Image.BILINEAR)
      data.append(np.array(face))
  else:
    img = img.resize(tuple(meta['inputs'][0]['tensor_shape'][:2]), Image.BILINEAR)
    data.append(np.array(img))

if data:
  output_tensor = sess.graph.get_tensor_by_name(meta['outputs'][0]['name'] + ':0')
  input_tensor = sess.graph.get_tensor_by_name(meta['inputs'][0]['name'] + ':0')

  output = sess.run(output_tensor, {input_tensor: np.array(data)})
  for i in range(output.shape[0]):
    if meta['detector_type'] == 'object_localization':
      image = images[i]
      if output[i, 0, 0] < 0:
        print(image, 'no object detected')
      for j in range(output.shape[1]):
        if output[i, j, 0] >= 0.0:
          # a valid detection
          bbox = output[i, j, :4]
          prob = list(zip(meta['label'], output[i, j, 4:]))
          print(image, bbox.tolist(), prob)
    elif faces:
      prob = zip(meta['label'], output[i, :])
      print(faces[i], prob)
    else:
      prob = zip(meta['label'], output[i, :])
      print(images[i], prob)
else:
  print('no face detected')
