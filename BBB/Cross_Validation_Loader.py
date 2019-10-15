"""
Cross_Validation_Loader

This class will create an iterator that does k-fold cross validation
"""

import sys
import os
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler


BASE_DIR = '/homes/murph213/DeepLearning/code_final'

class CrossValidationLoader(object):
    def __init__(self, transform, outputs, inputs, batch_size, num_folds):

        assert outputs == 10 and inputs == 1, "Unanticipated # of inputs and outputs for mnist"
        # assuming mnist!
        self.num_training = 60000

        # quick & dirty: same transform for each fold
        self.transform = transform
        self.outputs = outputs
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_folds = num_folds

        if self.num_training % self.num_folds != 0:
            print("Number training observations: {}".format(self.num_training))
            print("Number folds entered: {}".format(self.num_folds))
            raise RuntimeError('Folds should divide into the number of training observations evenly')

        # Determine size of each cross-validation fold
        self.fold_size = self.num_training // self.num_folds

    def __len__(self):
        r"""Length of the class"""
        return self.num_folds

    def __iter__(self):
        r"""Iterator of the class"""
        return self.Iterator(self)

    class Iterator(object):
        def __init__(self, loader):
            self.loader = loader
            self.ptr = 0

        def __len__(self):
            return len(self.loader)

        def get_train_valid_loader(self, transform, batch_size, train_idx, valid_idx,
                                   num_workers=1,
                                   pin_memory=True):
            """
            Utility function for loading and returning train and valid
            multi-process iterators
            If using CUDA, num_workers should be set to 1 and pin_memory to True.
            Params
            ------
            - batch_size: how many samples per batch to load.
            - num_workers: number of subprocesses to use when loading the dataset.
            - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
              True if using GPU.
            Returns
            -------
            - train_loader: training set iterator.
            - valid_loader: validation set iterator.
            """

            my_root = os.path.join(BASE_DIR, "data")

            # load the dataset
            train_dataset = torchvision.datasets.MNIST(root=my_root, train=True, download=True, transform=transform)
            valid_dataset = torchvision.datasets.MNIST(root=my_root, train=True, download=True, transform=transform)

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=num_workers, pin_memory=pin_memory,
            )

            return train_loader, valid_loader

        def __next__(self):
            if self.ptr >= self.loader.num_folds:
                raise StopIteration
            else:
                pass

            # update pointers in raw data for next element
            ptr0 = self.ptr * self.loader.fold_size
            ptr1 = min(ptr0 + self.loader.fold_size, self.loader.num_training - 1)
            self.ptr += 1

            all_idx = set(range(self.loader.num_training))
            fold_valid_idx = list(range(ptr0, ptr1))
            fold_train_idx = all_idx.difference(set(fold_valid_idx))
            fold_train_idx = list(fold_train_idx)

            train_size = len(fold_train_idx)
            test_size = len(fold_valid_idx)  # becomes test in their parlance
            # Create dataset loader for this batch
            train_loader, valid_loader = self.get_train_valid_loader(self.loader.transform,
                                                                     self.loader.batch_size,
                                                                     fold_train_idx,
                                                                     fold_valid_idx)
            return train_loader, valid_loader, train_size, test_size


if __name__ == "__main__":
    #
    # Test the class above in various ways
    # Mostly make sure you are getting different folds each time,
    #  the number of folds is what you expect, etc.
    #
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    cross_valid_loader = CrossValidationLoader(transform_train,
                                               10,
                                               1,
                                               54000,
                                               10)
    iterator = iter(cross_valid_loader)
    for ii in range(len(iterator)):
        train_loader, test_loader, train_size, test_size = next(iterator)
        #print("train {}\ntest{}".format(train_size, test_size))

        for batch_idx, (inputs_value, targets) in enumerate(train_loader):
            print(torch.mean(inputs_value))
            print(batch_idx)


