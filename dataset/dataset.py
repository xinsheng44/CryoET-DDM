import os
from tqdm import *
import torch
import torch.utils.data as data
import numpy as np


class CustomDataset_noise(data.Dataset):
    def __init__(self, inputs, labels, noises):
       self.inputs = inputs
       self.labels = labels
       self.noises = noises

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.float32)
        labels = torch.tensor(self.labels[index], dtype=torch.float32)
        noises = torch.tensor(self.noises[index], dtype=torch.float32)
        return inputs, labels, noises