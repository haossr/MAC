import os
import fire
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from torch.utils.data import DataLoader

from data import ImageClassificationDataset
from eval import get_loss_fn
from models import get_model
from util import init_exp_folder, Args


class Model(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args
        self.model = get_model(args)
        self.loss = get_loss_fn(args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.view(-1), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.view(-1), y)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        """
        Aggregate and return the validation metrics

        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss,
                'log': {'avg_val_loss': avg_loss},
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
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def train_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)

    @pl.data_loader
    def val_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    @pl.data_loader
    def test_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)


def train(save_dir="./sandbox",
          exp_name="DemoExperiment",
          model="ResNet18",
          gpus=4,
          pretrained=True,
          num_classes=1,
          log_save_interval=1,
          distributed_backend="ddp",
          gradient_clip_val=0.5,
          max_nb_epochs=3,
          train_percent_check=1.0,
          tb_path="./sandbox/tb",
          loss_fn="BCE",
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        log_save_interval: Logging saving frequency (in batch)
        distributed_backend: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        train_percent_check: Proportion of training data to use
        max_nb_epochs: Max number of epochs
        tb_path: Path to global tb folder
        loss_fn: Loss function to use

    Returns: None

    """
    args = Args(locals())
    init_exp_folder(args)
    m = Model(args)
    trainer = Trainer(distributed_backend=distributed_backend,
                      gpus=gpus,
                      logger=TestTubeLogger(
                          save_dir=os.path.join(save_dir, exp_name),
                          name='lightning_logs',
                          version="0"
                      ),
                      default_save_path=os.path.join(save_dir, exp_name),
                      log_save_interval=log_save_interval,
                      gradient_clip_val=gradient_clip_val,
                      train_percent_check=train_percent_check,
                      max_nb_epochs=max_nb_epochs)
    trainer.fit(m)


def test(save_dir="./sandbox/",
         gpus=4,
         checkpoint_path="./sandbox/DemoExperiment/ckpts/_ckpt_epoch_1.ckpt"):
    """
    Run the testing experiment.

    Args:
        model: Model name
        save_dir: Path to save the checkpoints and logs
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        checkpoint_path: Path for the experiment to load
        loss_fn: Loss function to use
    Returns: None

    """
    args = Args(locals())
    m = Model.load_from_checkpoint(checkpoint_path)
    trainer = Trainer(default_save_path=os.path.join(save_dir,
                                                     "result/test"),
                      gpus=gpus)
    trainer.test(m)


if __name__ == "__main__":
    fire.Fire()
