import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from torchvision import transforms
from PIL import Image


class CustomDataLoader:
    def __init__(self, data, target, shuffle=True, batch_size=32):
        self._data = data
        self._target = target
        self._shuffle = bool(shuffle)
        self._batch_size = int(batch_size)

        self._rand_idxs = None
        self._start_idx = None

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        if self._shuffle:
            self._rand_idxs = torch.randperm(
                len(self._data), device=self._data.device)
        self._start_idx = 0
        return self

    def __next__(self):
        start, nexamples = self._start_idx, len(self._data)
        if start >= nexamples:
            raise StopIteration
        end = min(start + self._batch_size, nexamples)
        if self._rand_idxs is not None:
            idxs = self._rand_idxs[start:end]
            batch = self._data[idxs], self._target[idxs]
        else:
            batch = self._data[start:end], self._target[start:end]
        self._start_idx = end

        return batch
