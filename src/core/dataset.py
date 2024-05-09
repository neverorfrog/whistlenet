from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import os
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from core.utils import *

projroot = project_root()
root = f"{projroot}/data"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Dataset(Parameters, ABC):
    """The abstract class for handling datasets"""
    def __init__(self,
    tobeloaded: bool, 
    params: dict,
    name = None,
    train_data = None,
    test_data = None,
    val_data = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.name = name
        path = os.path.join(root,self.name)
        self.load(path) if tobeloaded else self.save(path)
        self.params = params
        
    def train_dataloader(self, batch_size):
        """
        Returns a training dataloader with a specified batch size.

        Args:
            batch_size (int): The number of samples in each batch.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.
        """
        return self._get_dataloader(self.train_data, batch_size, False)
    
    def val_dataloader(self, batch_size):
        """
        Creates a validation data loader with the given batch size.
        
        Args:
            batch_size (int): The size of each batch.
        
        Returns:
            torch.utils.data.DataLoader: The validation data loader.
        """
        return self._get_dataloader(self.val_data, batch_size, False)

    def test_dataloader(self, batch_size):
        """
        A function to create a test data loader with the given batch size.
        
        Args:
            self: The object instance
            batch_size (int): The size of the batch for the data loader
        
        Returns:
            DataLoader: The test data loader
        """
        return self._get_dataloader(self.test_data, batch_size, False)

    def _get_dataloader(self, dataset, batch_size, use_weighting):
        """
        A function to get a DataLoader with optional weighted sampling.

        Parameters:
            dataset (Dataset): The dataset to load.
            batch_size (int): The batch size for the DataLoader.
            use_weighting (bool): Flag to enable weighted sampling.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        #Stuff for weighted sampling
        weighted_sampler = None
        if use_weighting:
            weights = [self.params['class_weights'][int(c)] for c in dataset.labels]
            weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        #Dataloader stuff
        g = torch.Generator()
        g.manual_seed(2000)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = weighted_sampler,
            shuffle = not use_weighting,
            num_workers=12,
            worker_init_fn=seed_worker,
            generator=g
        )
        
    def __len__(self):
        return len(self.train_data)
        
    def summarize(self):
        # gathering data
        data = self.train_data
        
        # summarizing
        print(f'N Examples: {len(data.data)}')
        print(f'N Classes: {len(data.classes)}')
        print(f'Classes: {data.classes}')
        
        # class breakdown
        for c in self.classes:
            total = len(data.labels[data.labels == c])
            ratio = (total / float(len(data.labels))) * 100
            print(f' - Class {str(c)}: {total} ({ratio})')
            
    def save(self, path=None):
        if path is None: return
        # path = os.path.join("data",self.name)
        torch.save(self.train_data, open(os.path.join(path,"train_data.dat"), "wb"))
        torch.save(self.val_data, open(os.path.join(path,"val_data.dat"), "wb"))
        torch.save(self.test_data, open(os.path.join(path,"test_data.dat"), "wb"))
        print("DATA SAVED!")
        
    def load(self, path=None):
        # path = os.path.join("data",self.name)
        self.train_data = torch.load(open(os.path.join(path,"train_data.dat"),"rb"))
        self.val_data = torch.load(open(os.path.join(path,"val_data.dat"),"rb"))
        self.test_data = torch.load(open(os.path.join(path,"test_data.dat"),"rb"))
        print("DATA LOADED!\n")
        
    def split(self, data, labels, ratio):
        """
        Split the data into two disjunct sets.

        Parameters:
            data (list): The input data.

        Returns:
            tuple: A tuple containing both splits
        """
        #decide the size
        first_size = len(labels)
        second_size = int(ratio * first_size)
        second_indices = np.random.permutation(first_size)[:second_size]
        
        #exclude samples that go into validation set
        first_data = data[torch.tensor(list(set(range(first_size)) - set(second_indices)))]
        first_labels = labels[torch.tensor(list(set(range(first_size)) - set(second_indices)))]
        
        #data with sampled indices
        second_data = data[second_indices]
        second_labels = labels[second_indices]
        
        return first_data, first_labels, second_data, second_labels
        
class TensorData(Dataset):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels
        self.classes = np.unique(self.labels)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        frame = self.data[index]
        label = self.labels[index]
        return frame, label