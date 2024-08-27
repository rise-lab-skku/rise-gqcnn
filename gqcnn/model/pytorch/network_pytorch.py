# -*- coding: utf-8 -*-
import os
import sys
import time
from itertools import cycle
from importlib.util import spec_from_file_location, module_from_spec

import yaml
import torch
import numpy as np
from autolab_core import Logger
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from .dataset import DexNetDataset


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)


class DatasetBuffer(torch.utils.data.Dataset):
    def __init__(self, images, poses, labels, im_mean, im_std, pose_mean, pose_std):
        self.images = images
        self.poses = poses
        self.labels = labels
        self.im_mean = im_mean
        self.im_std = im_std
        self.pose_mean = pose_mean
        self.pose_std = pose_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get data
        image = self.images[idx]
        pose = self.poses[idx]
        label = self.labels[idx]

        # normalize
        image = (image - self.im_mean) / self.im_std
        pose = (pose - self.pose_mean) / self.pose_std

        # typing
        image = image.astype(np.float32)
        pose = pose.astype(np.float32)
        label = label.astype(np.float32)

        # transpose
        image = image.transpose(2, 0, 1)

        return image, pose, label


class GQCNNPYTORCH(object):
    PKG_NAME = 'gqcnn_pytorch_deploy'

    def __init__(self, model_dir, verbose=True, log_file=None):
        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=log_file,
                                         silence=(not verbose),
                                         global_log_file=verbose)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        self._model_dir = model_dir
        self.load(model_dir)
        self.init_mean_and_std(model_dir)

        # set up input
        self.im_height = 96
        self.im_width = 96
        self.num_channels = 1

        # set up optimizer
        self._optimizer = None
        self._loss_fn = None

        # set flag
        self.FREEZE_BN_PRINT_FLAG = False

    def set_bn_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

            # print with color
            if not self.FREEZE_BN_PRINT_FLAG:
                self._logger.warn('BN layers are frozen')
                self.FREEZE_BN_PRINT_FLAG = True

    def save(self, save_path):
        """Save the GQ-CNN model to the provided path.

        Args:
            save_path (str): path to save the model to. Should be a .pt file.
        """
        torch.save({'model_state_dict': self._model.state_dict()}, save_path)

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
        config_path = os.path.join(model_dir, 'gqcnn_pytorch_deploy', 'config.yaml')
        with open(config_path, 'r') as f:
            pytorch_gqcnn_config = yaml.safe_load(f)

        # parse config
        self.gripper_mode = pytorch_gqcnn_config['gripper_mode']
        if self.gripper_mode == 'parallel_jaw':
            self.pose_dim = 1
        elif self.gripper_mode == 'suction':
            self.pose_dim = 2

        # import GQCNN model
        sys.path.append(model_dir)
        if pytorch_gqcnn_config['model']['architecture'] == 'gqcnn-large':
            from gqcnn_pytorch_deploy.architecture import GQCNNLarge as GQCNN
        else:
            raise ValueError('Invalid model type: {}'.format(pytorch_gqcnn_config['model']))

        # create model
        self._model = GQCNN(self.gripper_mode)
        assert isinstance(self._model, torch.nn.Module)
        self._model.to(self.device)

        # checkpoint
        checkpoint = torch.load(os.path.join(model_dir, 'gqcnn_pytorch_deploy', 'model_ckpt/checkpoint.pt'))
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self.device)

        # load ewc parameters
        ewc_pram_path = os.path.join(model_dir, 'gqcnn_pytorch_deploy', 'ewc/fisher.pt')
        if os.path.exists(ewc_pram_path):
            fisher = torch.load(ewc_pram_path)
            self._model.ewc_fisher = fisher
            color_str = '\033[92m{}\033[00m'
            print(color_str.format(
                'EWC parameters are loaded'))
        else:
            color_str = '\033[93m{}\033[00m'
            print(color_str.format(
                'Warning: EWC parameters are not loaded'))

        # print checkpoint info
        self._logger.warn('Model loaded from : {}'.format(os.path.join(model_dir)))
        self._logger.warn('Loaded checkpoint from epoch {} with loss {}.'.format(
            checkpoint['epoch'], checkpoint['loss']))

        # set model to eval mode
        self._model.eval()

    def init_mean_and_std(self, model_dir):
        """Initialize the mean and standard deviation of the training data.

        Args:
            model_dir (str): path to the directory containing the model.
        """
        norm_dir = os.path.join(model_dir, 'gqcnn_pytorch_deploy', 'norm_params')
        self._im_mean = np.load(os.path.join(norm_dir, 'im_mean.npy'))
        self._im_std = np.load(os.path.join(norm_dir, 'im_std.npy'))
        self._pose_mean = np.load(os.path.join(norm_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(norm_dir, 'pose_std.npy'))

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
            (N, 2) predictions. The second column is the probability of grasp success.
        """
        # Set model to eval mode.
        self._model.eval()

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
        output_arr = torch.nn.functional.softmax(output_arr, dim=1)
        output_arr = output_arr.detach().cpu().numpy()

        # Get total prediction time.
        pred_time = time.time() - start_time
        if verbose:
            self._logger.info("Prediction took {} seconds.".format(pred_time))

        return output_arr

    def finetune(self, image_arr, pose_arr, labels, batch_size, lr, ewc_penalty=0.0):
        """Finetune the model on the provided data.

        Args:
            image_arr (numpy.ndarray): 4D tensor of depth images.
            pose_arr (numpy.ndarray): Tensor of gripper poses.
            labels (numpy.ndarray): Tensor of labels.
            batch_size (int): batch size.
            lr (float): learning rate.
            ewc_penalty (float): EWC penalty.
        """
        # set optimizer and loss function
        if (self._optimizer is None) or (self._loss_fn is None):
            self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
            self._loss_fn = torch.nn.CrossEntropyLoss()

        # Set model to train mode.
        self._model.train()

        # set freeze bn
        self._model.apply(self.set_bn_eval)

        # get dataset
        dataset = DatasetBuffer(image_arr, pose_arr, labels, self._im_mean, self._im_std, self._pose_mean, self._pose_std)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True)

        # finetune
        for image_tensor, pose_tensor, labels_tensor in dataloader:
            image_tensor = image_tensor.to(self.device, non_blocking=True)
            pose_tensor = pose_tensor.to(self.device, non_blocking=True)
            labels_tensor = labels_tensor.to(self.device, non_blocking=True)

            # forward pass
            preds = self._model(image_tensor, pose_tensor)
            cls_loss = self._loss_fn(preds, labels_tensor)
            ewc_loss = self._model.ewc_loss(ewc_penalty)
            loss = cls_loss + ewc_loss

            # zero gradients
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # Set model to eval mode.
        self._model.eval()

    def finetune_bakcup(self, image_arr, pose_arr, labels, batch_size, lr, ewc_penalty=0.0):
        """Finetune the model on the provided data.

        Args:
            image_arr (numpy.ndarray): 4D tensor of depth images.
            pose_arr (numpy.ndarray): Tensor of gripper poses.
            labels (numpy.ndarray): Tensor of labels.
            batch_size (int): batch size.
            lr (float): learning rate.
            ewc_penalty (float): EWC penalty.
        """
        # set optimizer and loss function
        if (self._optimizer is None) or (self._loss_fn is None):
            self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
            self._loss_fn = torch.nn.CrossEntropyLoss()

        # Set model to train mode.
        self._model.train()

        # set freeze bn
        self._model.apply(self.set_bn_eval)

        # get dataset
        dataset = DatasetBuffer(image_arr, pose_arr, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            max_workers=8,
            pin_memory=True)

        # shuffle data
        idx = np.arange(image_arr.shape[0])
        np.random.shuffle(idx)

        # partition data into batches
        num_batches = int(np.ceil(image_arr.shape[0] / batch_size))
        batch_indices = []
        for i in range(num_batches):
            batch_indices.append(idx[i * batch_size:(i + 1) * batch_size])

        # preprocess data
        image_arr = (image_arr - self._im_mean) / self._im_std
        image_arr = np.transpose(image_arr, (0, 3, 1, 2))
        pose_arr = (pose_arr - self._pose_mean) / self._pose_std

        # convert to torch tensor and move to device
        image_tensor = torch.from_numpy(image_arr).float()
        pose_tensor = torch.from_numpy(pose_arr).float()
        labels_tensor = torch.from_numpy(labels).float()

        # finetune
        for batch_idx in batch_indices:
            # add rehearsal examples
            _image_tensor = image_tensor[batch_idx].to(self.device)
            _pose_tensor = pose_tensor[batch_idx].to(self.device)
            _labels_tensor = labels_tensor[batch_idx].to(self.device)

            # forward pass
            preds = self._model(_image_tensor, _pose_tensor)
            cls_loss = self._loss_fn(preds, _labels_tensor)
            ewc_loss = self._model.ewc_loss(ewc_penalty)
            loss = cls_loss + ewc_loss

            # zero gradients
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # Set model to eval mode.
        self._model.eval()
