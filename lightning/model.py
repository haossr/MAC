import os
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection, VOCDetection
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings

from data import VOC, collater
from eval import get_loss_fn, evaluate, log_detection, pretty_detection
from models import EfficientDet, get_model
from util import init_exp_folder, Args, nms_batch


warnings.simplefilter(action='ignore', category=FutureWarning)
class Model(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args
        self.model = get_model(args)

    def forward(self, x):
        images, annots, scales = x
        classification_loss, regression_loss = self.model([images, annots])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.forward(batch)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        """ Generate the anchor box and NMS """
        images, annots, scales = batch 
        boxes, probs = self.model.detect(images, scales)
        detections = nms_batch(boxes, probs, iou_threshold=0.01)
        for i, (img, annot, detection) in enumerate(zip(images, annots, detections)):
            print("="*15 + f"sample [{i}]" + "="*15)
            pretty_detection(detection)
            log_detection(self.logger, img, None, annots, [500,500], image_name=f"valid_image_{batch_nb}_{i}")
        with torch.no_grad():
            loss = self.forward(batch)
        return {'val_loss': loss,
                'detections': detections,
                'annotations': annots}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        detections = np.stack([x['detections'] for x in outputs])
        annotations = np.stack([x['annotations'] for x in outputs])
        #print(f"detections: {detections}")
        mAP = evaluate(detections, annotations)
        return {'val_loss': avg_loss,
                'mAP': mAP,
                'log': {'_val_loss': avg_loss},
                'progress_bar': {'avg_val_loss': avg_loss}}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.view(-1), y)
        return {'test_loss': loss, 'log': {'test_loss': loss}}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=3, verbose=True)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        dataset = VOC(root=self.hparams.root_dir, split="train")
        return DataLoader(dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=8,
                          shuffle=True,
                          collate_fn=collater,
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = VOC(self.hparams.root_dir, split="val")
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          collate_fn=collater)
                          #sampler=dist_sampler,
                          #pin_memory=True)
