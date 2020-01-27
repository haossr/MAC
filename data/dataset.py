import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose

from .augmentation import (Resizer, Normalizer,
                           Augmenter, Annotator, ToTensorDict)


class VOC(torch.utils.data.Dataset):
    def __init__(self, root="~/data/voc/",
                 year=2007, split="train", transforms=None):
        self._data = VOCDetection(root=root,
                                  image_set=split,
                                  year=str(year),
                                  download=True,
                                  target_transform=Annotator())
        if transforms is None:
            transforms = Compose([Resizer(), ToTensorDict()])
        #                        [Normalizer(), Augmenter(), Resizer()])

        self._transforms = transforms

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image, target = self._data[index]
        batch = {"img": image, "annot": target}
        if self._transforms is not None:
            batch = self._transforms(batch)
        return batch
