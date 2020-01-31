import fire
import os
from pytorch_lightning import Trainer
from util import init_exp_folder, Args

from lightning import Model
from lightning import get_logger

def train(save_dir="./sandbox",
          root_dir="./dataset",
          exp_name="DemoExperiment",
          model="efficientdet-d0",
          lr=1e-4,
          gpus=1,
          batch_size=16,
          pretrained=True,
          num_class=20,
          log_save_interval=1,
          distributed_backend="dp",
          gradient_clip_val=0.5,
          max_nb_epochs=3,
          train_percent_check=1,
          val_percent_check=1,
          tb_path="./sandbox/tb",
          debug=False,
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
        batch_size: int. Number of samples in a batch
        pretrained: Whether or not to use the pretrained model
        num_class: Number of classes
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
    if debug:
        train_percent_check = val_percent_check = 0.01
    trainer = Trainer(distributed_backend=distributed_backend,
                      gpus=gpus,
                      logger=get_logger(save_dir, exp_name),
                      default_save_path=os.path.join(save_dir, exp_name),
                      log_save_interval=log_save_interval,
                      gradient_clip_val=gradient_clip_val,
                      train_percent_check=train_percent_check,
                      val_percent_check=val_percent_check,
                      max_nb_epochs=max_nb_epochs)
    trainer.fit(m)


if __name__ == "__main__":
    fire.Fire()
