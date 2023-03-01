import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class SVHN:
    """ 
        SVHN dataset.
    """

    def __init__(self, args):
        self.args = args

        # self.mean = torch.Tensor([0.5, 0.5, 0.5])
        # self.std = torch.Tensor([0.5, 0.5, 0.5])
        self.mean = torch.Tensor([0.43090966, 0.4302428, 0.44634357])
        self.std = torch.Tensor([0.19759192, 0.20029082, 0.19811132])

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.SVHN(
            root=os.path.join(self.args.data_dir, "SVHN"),
            split="train",
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.SVHN(
            root=os.path.join(self.args.data_dir, "SVHN"),
            split="test",
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader
