# -*- coding: utf-8 -*-
import os
import sys
import time
from importlib.util import spec_from_file_location, module_from_spec

import yaml
import torch
import numpy as np
from autolab_core import Logger


class GQCNNPYTORCH(object):
    def __init__(self, model_dir, verbose=True, log_file=None):
        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=log_file,
                                         silence=(not verbose),
                                         global_log_file=verbose)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        self.load(model_dir)
        self.init_mean_and_std(model_dir)

        self.im_height = 96
        self.im_width = 96
        self.num_channels = 1

    def load(self, model_dir):
        """Load the GQ-CNN model from the provided directory.

        Parameters
        ----------
        model_dir : str
            Path to the directory containing the model.
        verbose : bool
            Whether or not to log initialization output to `stdout`.
        log_file : str
            Path to the file to log initialization output to.
        """
        # load config file from model directory
        config_path = os.path.join(model_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            pytorch_gqcnn_config = yaml.load(f)

        # parse config
        self.gripper_mode = pytorch_gqcnn_config['gripper_mode']
        if self.gripper_mode == 'parallel_jaw':
            self.pose_dim = 1
        elif self.gripper_mode == 'suction':
            self.pose_dim = 2

        # import model from checkpoint
        spec = spec_from_file_location('model.GQCNN', os.path.join(model_dir, 'model.py'))
        module = module_from_spec(spec)
        if 'model.GQCNN' in sys.modules:
            del sys.modules['model.GQCNN']
        sys.modules['model.GQCNN'] = module
        spec.loader.exec_module(module)

        # checkpoint
        checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))

        # init model
        self._model = module.GQCNN(self.gripper_mode)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self.device)

        # set model to eval mode
        self._model.eval()

    def init_mean_and_std(self, model_dir):
        self._im_mean = np.load(os.path.join(model_dir, 'im_mean.npy'))
        self._im_std = np.load(os.path.join(model_dir, 'im_std.npy'))
        self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))

    def predict(self, image_arr, pose_arr, verbose=False):
        """Predict the probability of grasp success given a depth image and
        gripper pose.

        Parameters
        ----------
        image_arr : :obj:`numpy ndarray`
            4D tensor of depth images.
        pose_arr : :obj:`numpy ndarray`
            Tensor of gripper poses.
        verbose : bool
            Whether or not to log progress to stdout, useful to turn off during
            training.

        Returns
        -------
        :obj:`numpy ndarray`
            Tensor of grasp success probabilities.
        """
        # Get prediction start time.
        start_time = time.time()

        if verbose:
            self._logger.info("Predicting...")

        # preprocess data
        image_arr = (image_arr - self._im_mean) / self._im_std
        image_arr = np.transpose(image_arr, (0, 3, 1, 2))
        pose_arr = (pose_arr - self._pose_mean) / self._pose_std

        # convert to torch tensor and move to device
        image_tensor = torch.from_numpy(image_arr).float().to(self.device)
        pose_tensor = torch.from_numpy(pose_arr).float().to(self.device)

        # inference
        output_arr = self._model(image_tensor, pose_tensor)
        output_arr = output_arr.detach().cpu().numpy()

        # Get total prediction time.
        pred_time = time.time() - start_time
        if verbose:
            self._logger.info("Prediction took {} seconds.".format(pred_time))

        return output_arr
