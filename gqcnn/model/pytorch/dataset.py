# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
from torch.utils.data import Dataset


# field parameters
IMAGE_FIELD_NAME = "tf_depth_ims"
POSE_FEILD_NAME = "grasps"
GRASP_METRIC_FIELD_NAME = "grasp_metrics"


class DexNetDataset(Dataset):
    def __init__(self, dataset_dir, gripper_mode, split, relabel=False, transform=True, augment='random', debug=False):
        """Dataset class for DexNet.

        Args:
            dataset_dir (str): path to the dataset directory.
            gripper_mode (str): dataset type. (e.g. 'suction', 'parallel_jaw')
            split (str, optional): train, val, or all.
            relabel (str, optional): relabel name. (e.g. 'official_batchnorm/sim'). Defaults to False.
            transform (bool, optional): Transform the data to train the model. Defaults to True.
            augment (str, optional): Augment the data to train the model. Options are 'random', 'lr', 'ud', 'rot180', 'none'. Defaults to 'random'.
            debug (bool, optional): Debug mode. Defaults to False.
        """
        # dataset option
        self.gripper_mode = gripper_mode
        self.split = split
        self.transform = transform
        self.augment = augment
        self.debug = debug

        # data directory
        dataset_dir = os.path.join(dataset_dir, gripper_mode)
        self.tensor_dir = os.path.join(dataset_dir, 'tensors_uncompressed')
        if relabel is False:
            self.label_dir = self.tensor_dir
        else:
            self.label_dir = os.path.join(dataset_dir, 'tensors_uncompressed_labels', relabel)
        self.split_dir = os.path.join(dataset_dir, 'splits/image_wise')
        self.norm_dir = os.path.join(dataset_dir, 'norm_params')

        # load std, mean
        im_mean_file = os.path.join(self.norm_dir, 'im_mean.npy')
        im_std_file = os.path.join(self.norm_dir, 'im_std.npy')
        pose_mean_file = os.path.join(self.norm_dir, 'pose_mean.npy')
        pose_std_file = os.path.join(self.norm_dir, 'pose_std.npy')
        self.im_mean = np.load(im_mean_file).astype(np.float32)
        self.im_std = np.load(im_std_file).astype(np.float32)
        self.pose_mean = np.load(pose_mean_file).astype(np.float32)
        self.pose_std = np.load(pose_std_file).astype(np.float32)

        # load indices
        if self.split == 'train':
            self.indices = np.load(os.path.join(self.split_dir, 'train_indices.npz'))['arr_0']
            self.indices = np.sort(self.indices)
        elif self.split == 'val':
            self.indices = np.load(os.path.join(self.split_dir, 'val_indices.npz'))['arr_0']
            self.indices = np.sort(self.indices)
        elif self.split == 'all':
            train_indices = np.load(os.path.join(self.split_dir, 'train_indices.npz'))['arr_0']
            val_indices = np.load(os.path.join(self.split_dir, 'val_indices.npz'))['arr_0']
            self.indices = np.sort(np.concatenate([train_indices, val_indices]))

        # reduce data len
        if debug:
            self.indices = self.indices[:1000]
        print('DexNetDataset is loaded with:')
        print('  - dataset_dir: {}'.format(dataset_dir))
        print('  - gripper_mode: {}'.format(gripper_mode))
        print('  - split: {}'.format(split))
        print('  - label: {}'.format(self.label_dir))
        print('  - transform: {}'.format(transform))
        print('  - augment: {}\n'.format(augment))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        file_idx = idx // 100
        data_idx = idx % 100

        # get raw data
        image = self.load_image_tensor(file_idx, data_idx)
        pose = self.load_pose_tensor(file_idx, data_idx)
        label = self.load_label_tensor(file_idx, data_idx)

        # augment(random flip)
        if self.augment == 'random':
            p = np.random.rand()
            if p < 0.25:
                pass
            elif p < 0.5:  # Left and right flip
                image = np.flip(image, axis=1)
            elif p < 0.75:  # Up and down flip
                image = np.flip(image, axis=0)
                if self.gripper_mode == 'suction':
                    pose[1] = -pose[1]
            else:
                image = np.flip(image, axis=1)
                image = np.flip(image, axis=0)
                if self.gripper_mode == 'suction':
                    pose[1] = -pose[1]
        elif self.augment == 'lr':
            image = np.flip(image, axis=1)
        elif self.augment == 'ud':
            image = np.flip(image, axis=0)
            if self.gripper_mode == 'suction':
                pose[1] = -pose[1]
        elif self.augment == 'rot180':
            image = np.flip(image, axis=1)
            image = np.flip(image, axis=0)
            if self.gripper_mode == 'suction':
                pose[1] = -pose[1]
        elif self.augment == 'none':
            pass
        else:
            raise ValueError('augment option is not valid.')

        # preprocessing
        if self.transform:
            image = image.transpose([2, 0, 1])
            image = (image - self.im_mean) / self.im_std
            pose = (pose - self.pose_mean) / self.pose_std
            if label > 0.5:
                label = np.array([0, 1], dtype=np.float32)
            else:
                label = np.array([1, 0], dtype=np.float32)
        else:
            image = image.copy()  # must copy:??? why?
        return image, pose, label, idx

    def load_image_tensor(self, file_idx, data_idx):
        """Load pose tensor from file.

        Args:
            file_idx (`int`): file index. (e.g. 0, 1, 2, ...)
            data_idx (`np.ndarray`): data indices. (e.g. [0, 1, 2, ...])

        Returns:
            `np.ndarray`: image tensor. shape = (H, W, C)
        """
        # dir 5 zero padding
        im_tensor_path = os.path.join(
            self.tensor_dir,
            "{}_{:05d}/{}.npy".format(IMAGE_FIELD_NAME, file_idx, data_idx))
        im_tensor = np.load(im_tensor_path)
        if len(im_tensor.shape) == 2:
            im_tensor = np.expand_dims(im_tensor, axis=2)
        return im_tensor.astype(np.float32)

    def load_pose_tensor(self, file_idx, data_idx):
        """Load pose tensor from file.

        Args:
            file_idx (`int`): file index. (e.g. 0, 1, 2, ...)
            data_idx (`np.ndarray`): data indices. (e.g. [0, 1, 2, ...])

        Returns:
            `np.ndarray`: [depth] or [depth, approach_angle]
        """
        pose_tensor_path = os.path.join(
            self.tensor_dir,
            "{}_{:05d}/{}.npy".format(POSE_FEILD_NAME, file_idx, data_idx))
        if self.gripper_mode == 'parallel_jaw':
            pose_tensor = np.load(pose_tensor_path)[2:3]
        elif self.gripper_mode == 'suction':
            pose_tensor = np.load(pose_tensor_path)[[2, 4]]
        return pose_tensor.astype(np.float32)

    def load_label_tensor(self, file_idx, data_idx):
        """load label tensor.

        Args:
            file_idx (`int`): file index. (e.g. 0, 1, 2, ...)
            data_idx (`np.ndarray`): data indices. (e.g. [0, 1, 2, ...])

        Returns:
            `np.ndarray`: label tensor. shape = (,)
        """
        label_tensor_path = os.path.join(
            self.label_dir,
            "{}_{:05d}/{}.npy".format(GRASP_METRIC_FIELD_NAME, file_idx, data_idx))
        label_tensor = np.load(label_tensor_path)
        return label_tensor

    def save_label_tensor(self, file_idx, data_idx, label_tensor):
        """save label tensor.

        Args:
            file_idx (`int`): file index. (e.g. 0, 1, 2, ...)
            data_idx (`np.ndarray`): data indices. (e.g. [0, 1, 2, ...])
            label_tensor (`np.ndarray`): label tensor. shape = (,)
        """
        label_tensor_path = os.path.join(
            self.tensor_dir,
            "{}_{:05d}/{}.npy".format(GRASP_METRIC_FIELD_NAME, file_idx, data_idx))
        np.save(label_tensor_path, label_tensor)


if __name__ == '__main__':
    # TEST yaml
    yaml_path = 'config/train.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config = config['train_dataset']
    print(dataset_config)
    exit()

    # USER INPUT
    # dataset_name = 'dexnet_4.0'
    dataset_name = 'dexnet_4.0_phys/official_batchnorm/sim/level_1-3/round_1-5'
    gripper_mode = 'parallel_jaw'
    dataset_path = os.path.join(
        '/media/sungwon/WorkSpace/dataset/AUTOLAB/gqcnn_training_dataset',
        dataset_name)
    dataset = DexNetDataset(
        dataset_dir=dataset_path,
        gripper_mode=gripper_mode,
        split='all',
        label=None,
        transform=True,
        augment='random')
    print(dataset[0])
