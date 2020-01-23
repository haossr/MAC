import os
import shutil
import unittest
from os.path import join

from main import train, test


class TestTrainFunc(unittest.TestCase):
    def setUp(self):
        self.path = "./sandbox"

    def tearDown(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_default(self):
        train(save_dir=self.path, gpus=None)

    def test_slurm(self):
        os.environ['SLURM_JOB_ID'] = "138"
        train(save_dir=self.path, gpus=None)
        self.assertTrue(os.path.exists(join(self.path,
                        "DemoExperiment/lightning_logs/version_0")))

    def test_override(self):
        train(save_dir=self.path, gpus=None)
        with self.assertRaises(FileExistsError):
            train(save_dir=self.path, gpus=None)


class TestTestFunc(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath("./sandbox")
        self.checkpoint_path = join(self.path,
                                    "DemoExperiment/ckpts/_ckpt_epoch_1.ckpt")
        train(save_dir=self.path, gpus=None)

    def tearDown(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_default(self):
        test(save_dir=self.path,
             checkpoint_path=self.checkpoint_path,
             gpus=None)


if __name__ == "__main__":
    unittest.main(verbosity=0)
