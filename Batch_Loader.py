import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from data.subsample import create_mask_for_mask_type
# from tensorboardX import SummaryWriter
# from utils.options import args_parser
from torch.utils.data import random_split
from data.mri_data import SliceData
from data import transforms
from data import transformsPB as Tpb
import pathlib
from torch.utils.data import DataLoader
import fastmri
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        target = Tpb.to_tensor(target.astype(np.complex64)) # (Height x Width x 2)
        kspace = Tpb.fft2c_new(target) # (Height x Width x 2)
        
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(
                kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        image = Tpb.ifft2c_new(masked_kspace)

        abs_image = Tpb.complex_abs(image)
        mean = torch.tensor(0.0)
        std = abs_image.mean()

        if image.dim() == 3 and image.shape[-1] == 2:
            image = image.permute(2, 0, 1)  # (2, H, W)
        else:
            raise ValueError(f"[Slice {slice}] Unexpected image shape before permute: {image.shape}")

        if target.dim() == 3 and target.shape[-1] == 2:
            target = target.permute(2, 0, 1)  # (2, H, W)
        else:
            raise ValueError(f"[Slice {slice}] Unexpected target shape before permute: {target.shape}")

        if masked_kspace.dim() == 3 and masked_kspace.shape[-1] == 2:
            masked_kspace = masked_kspace.permute(2, 0, 1)  # (2, H, W)
        else:
            raise ValueError(f"[Slice {slice}] Unexpected masked_kspace shape before permute: {masked_kspace.shape}")
        image = transforms.normalize(image, mean, std, eps=0)
        
        masked_kspace = transforms.normalize(masked_kspace, mean, std, eps=0)
        target = transforms.normalize(target, mean, std, eps=0)
        mask = mask.repeat(image.shape[1], 1, 1).squeeze().unsqueeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).repeat(image.shape[1], 1, 1)
        elif mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask
        else:
            raise ValueError(f"[Slice {slice}] Unexpected mask shape: {mask.shape}")

        return image, target, mean, std, attrs['norm'].astype(np.float32), fname, slice, attrs['max'].astype(np.float32), mask, masked_kspace

def _create_dataset(data_paths, transform, batch_size, shuffle, sample_rate, num_workers, pin_memory, mode='train'):
    datasets = [SliceData(root=pathlib.Path(path), transform=transform, sample_rate=sample_rate, challenge='multicoil',sequence='PD') for path in data_paths]
    datasets = torch.utils.data.ConcatDataset(datasets)

    if mode == 'train':
        train_ratio = 0.75
        train_size = int(len(datasets) * train_ratio)
        val_size = len(datasets) - train_size
        train_dataset, _ = random_split(datasets, [train_size, val_size])
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    elif mode == 'val':
        train_ratio = 0.75
        train_size = int(len(datasets) * train_ratio)
        val_size = len(datasets) - train_size
        _, val_dataset = random_split(datasets, [train_size, val_size])
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    elif mode == 'test':
        return DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    else:
        raise ValueError(f"Invalid mode {mode}. Use 'train', 'val', or 'test'.")

def complex_to_magnitude(x):
    # (B, 2, H, W)
    x = x.permute(0, 2, 3, 1) # (B, H, W, 2)
    magnitude = fastmri.complex_abs(x) # (B, H, W)
    return magnitude.unsqueeze(1) # (B, 1, H, W)