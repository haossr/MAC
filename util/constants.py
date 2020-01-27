"""Define constants to be used throughout the repository."""

# Main paths

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# VOC Labels
VOC_NAMES = ['aeroplane',
             'bicycle',
             'bird',
             'boat',
             'bottle',
             'bus',
             'car',
             'cat',
             'chair',
             'cow',
             'diningtable',
             'dog',
             'horse',
             'motorbike',
             'person',
             'pottedplant',
             'sheep',
             'sofa',
             'train',
             'tvmonitor']
VOC_LABEL2NAME = {i: n for i, n in enumerate(VOC_NAMES)}
VOC_NAME2LABEL = {n: i for i, n in enumerate(VOC_NAMES)}
