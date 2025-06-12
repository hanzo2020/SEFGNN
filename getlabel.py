import os
import numpy as np
import torch
import pickle
import torch as t
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from utils import *

class CancerDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CancerDataset, self).__init__('.', None, None)
        self.data_list = data_list
        self._data, self.slices = self.collate(self.data_list)
        self.num_slices = len(self.data_list)
        self.num_nodes = self.data_list[0].num_nodes
        self._data.num_classes = 2

    def get_idx_split(self, i):
        temp2 = self.data_list[0].train_mask
        temp = self.data_list[0].train_mask[:, i]
        temp1 = torch.where(temp)
        temp3 = self.data_list[1].train_mask
        train_idx = torch.where(self.data_list[0].train_mask[:, i])[0]#这相当于只取了最后一列，不知道啥情况
        test_idx = torch.where(self.data_list[0].test_mask[:, i])[0]
        valid_idx = torch.where(self.data_list[0].valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def convert_tensor(tensor):
    result = np.zeros((tensor.shape[0], 1), dtype=int)  # 初始化结果数组
    result[tensor[:, 0] == 1] = -1  # 第一列为1时，结果为-1
    result[tensor[:, 1] == 1] = 1   # 第二列为1时，结果为1
    return result

dataset_dir = 'data/Lung_Cancer_Matrix/A549_dataset_final.pkl'
data_dir='data/Breast_Cancer_Matrix'
cell_line = 'MCF7'
data = pd.read_csv(data_dir + '/' + cell_line +"-Label.txt", sep='\t')
with open(dataset_dir, 'rb') as f:
    cv_dataset = pickle.load(f)
    y = cv_dataset.data_list[0].y
    print(y)
    label = convert_tensor(y)
    print(label)
    data['label'] = label
    data.to_csv('A549-Label.txt', index=False, sep='\t')