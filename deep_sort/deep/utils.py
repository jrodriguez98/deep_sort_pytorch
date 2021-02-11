import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           normalize=None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - normalize: normalization transform

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg


    if normalize == None:
        normalize = transforms.Normalize(
            mean=[0.34985983, 0.32663023, 0.29609418],
            std=[0.14047915, 0.134799, 0.13037874]
        )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    valid_dataset = datasets.ImageFolder(data_dir, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return train_loader, valid_loader

def test_loader(test_dir,
                batch_size=1,
                normalize=None):

    if normalize == None:
        normalize = transforms.Normalize(
            mean=[0.34985983, 0.32663023, 0.29609418],
            std=[0.14047915, 0.134799, 0.13037874]
        )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return test_loader



def calculate_mean_std_per_channel(train_dir, batch_size):
    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_dir, transform=transforms.ToTensor()),
        batch_size=1
    )

    mean_red, std_red = 0, 0
    mean_blue, std_blue = 0, 0
    mean_green, std_green = 0, 0
    dataset_lenght = len(trainloader.dataset)

    for data_input, _ in trainloader:

        mean_red += data_input[0][0].mean()
        mean_blue += data_input[0][1].mean()
        mean_green += data_input[0][2].mean()

        std_red += data_input[0][0].std()
        std_blue += data_input[0][1].std()
        std_green += data_input[0][2].std()

    mean_red, std_red = mean_red*batch_size/dataset_lenght, std_red*batch_size/dataset_lenght
    mean_blue, std_blue = mean_blue*batch_size/dataset_lenght, std_blue*batch_size/dataset_lenght
    mean_green, std_green = mean_green*batch_size/dataset_lenght, std_green*batch_size/dataset_lenght

    normalize = transforms.Normalize(
        [float(mean_red.numpy()), float(mean_blue.numpy()), float(mean_green.numpy())],
        [float(std_red.numpy()), float(std_blue.numpy()), float(std_green.numpy())],
    )

    return normalize
