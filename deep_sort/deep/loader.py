from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class CustomFolder(DatasetFolder):
    """A custom data loader where the samples are arranged in this way: ::

            root/0001/xxx.jpg
            root/0001/xxy.jpg

            root/0002/123.jpg
            root/0002/nsdf3.jpg

        Args:
            root (string): Root directory path.
            loader (callable): A function to load a sample given its path.
            extensions (tuple[string]): A list of allowed extensions.
                both extensions and is_valid_file should not be passed.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
            is_valid_file (callable, optional): A function that takes path of a file
                and check if the file is a valid file (used to check of corrupt files)
                both extensions and is_valid_file should not be passed.

         Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
        """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(CustomFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                           transform=transform,
                                           target_transform=target_transform,
                                           is_valid_file=is_valid_file)
        self.idx_to_class = {idx: int(cls_name.lstrip('0')) for cls_name, idx in self.class_to_idx.items()}
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, self.idx_to_class[target]


