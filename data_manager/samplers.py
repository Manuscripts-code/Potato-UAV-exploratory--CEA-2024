from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import shuffle

from data_manager.loaders import StructuredData


class Splitter(Protocol):
    def __call__(self, y_data, labels) -> tuple[np.ndarray, np.ndarray]:
        ...


class Sampler:
    def __init__(
        self,
        splitter: Splitter,
        data: StructuredData,
        *,
        split_size_test=0.2,
        random_state=-1,
        shuffle=True
    ):
        self.splitter = splitter
        self.data = data
        self.split_size_test = split_size_test
        self.random_state = None if random_state == -1 else random_state
        self.shuffle = shuffle


class StratifySplitter:
    def __call__(self, y_data, labels):
        idx_full = np.arange(len(y_data))

        train_index, test_index = train_test_split(
            idx_full,
            test_size=self.train_test_split_size,
            stratify=y_data,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        return train_index, test_index


class RandomSplitter:
    def __call__(self, y_data, labels):
        idx_full = np.arange(len(y_data))

        train_index, test_index = train_test_split(
            idx_full,
            test_size=self.train_test_split_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        return train_index, test_index
