import os
import pandas as pd
import numpy as np
import torch
import pickle
import torch as t
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from utils import *

EPS = 1e-8




class CancerDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CancerDataset, self).__init__('.', None, None)
        self.data_list = data_list
        self._data, self.slices = self.collate(self.data_list)
        self.num_slices = len(self.data_list)
        self.num_nodes = self.data_list[0].num_nodes
        self._data.num_classes = 2

    def get_idx_split(self, i):
        train_idx = torch.where(self.data_list[0].train_mask[:, i])[0]
        test_idx = torch.where(self.data_list[0].test_mask[:, i])[0]
        valid_idx = torch.where(self.data_list[0].valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)




def get_data(configs, stable=True):
    data_dir = configs["data_dir"]
    load_data = configs["load_data"]
    cell_line = get_cell_line(data_dir)

    def get_dataset_dir(overlap, stable):
        dataset_suffix = "_dataset_final" if stable else "_dataset"
        if overlap:
            dataset_suffix = '_op' + dataset_suffix

        dataset_dir = os.path.join(
            data_dir, cell_line + dataset_suffix + '.pkl')
        
        return dataset_dir

    if load_data:
        dataset_dir = get_dataset_dir(configs['overlap'], stable)
        print(f"Loading dataset from: {dataset_dir} ......")
        with open(dataset_dir, 'rb') as f:
            cv_dataset = pickle.load(f)
        if configs['model'] == 'EFMGNN':
            print('EFMGNN dataset loader')
        return cv_dataset



