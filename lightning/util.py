import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger


def get_ckpt_callback(save_path, exp_name):
    return ModelCheckpoint(filepath=os.path.join(save_path, exp_name, "ckpts"),
                           save_top_k=1,
                           verbose=True,
                           monitor='val_loss',
                           mode='min',
                           prefix='')


def get_early_stop_callback(patience):
    return EarlyStopping(monitor='val_loss',
                         patience=patience,
                         verbose=True,
                         mode='min')

def get_logger(save_dir, exp_name):
    return TestTubeLogger(
                          save_dir=os.path.join(save_dir, exp_name),
                          name='lightning_logs',
                          version="0"
                      )
