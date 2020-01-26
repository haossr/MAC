import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torchvision
import torch
import numpy as np
import cv2


def get_augumentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.LongestMaxSize(
                max_size=width, always_apply=True),
            albu.PadIfNeeded(min_height=height, min_width=width,
                             always_apply=True, border_mode=0, value=[0, 0, 0]),
            albu.augmentations.transforms.RandomResizedCrop(
                height=height,
                width=width, p=0.3),
            albu.augmentations.transforms.Flip(),
            albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.4),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=20, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=5,
                                        sat_shift_limit=5),
                albu.NoOp()
            ]),
            albu.CLAHE(p=0.8),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test' or phase == 'valid'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=min_area,
                                                                     min_visibility=min_visibility, label_fields=['category_id']))


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return (imgs, torch.FloatTensor(annot_padded))


class Annotator:
    """Parse the annotation"""
    LABELS = ['aeroplane',
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

    def __init__(self):
        self.label_map = {l: i for i, l in enumerate(self.LABELS)}

    def __call__(self, target):
        annotation = target['annotation']
        objs = annotation['object']
        annots = []
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            label = self.label_map[obj['name']]
            bbox = [int(obj['bndbox'][key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']]
            annots.append(bbox + [label])
        return np.array(annots).astype(np.float32)


class Resizer(object):
    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        image = torchvision.transforms.functional.resize(image, common_size)
        height, width = image.size
        if height > width:
            scale = common_size / height
        else:
            scale = common_size / width
        annots[:, :4] *= scale
        return {'img': image, 'annot': annots, 'scale': scale}


class ToTensorDict(object):
    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        annots = torch.from_numpy(annots)
        image = torchvision.transforms.functional.to_tensor(image)
        return {'img': image, 'annot': annots, 'scale': scale}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
