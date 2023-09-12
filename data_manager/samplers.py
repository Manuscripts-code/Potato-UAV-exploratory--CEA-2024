from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from data_manager.loaders import StructuredData


class Splitter(Protocol):
    def __call__(self, data: StructuredData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class Sampler:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter

    def sample(self, data: StructuredData) -> tuple[StructuredData, StructuredData, StructuredData]:
        assert (
            len(data.data) == len(data.meta) == len(data.target)
        ), "'data.data', 'data.meta' and 'data.target' must have the same length!"
        train_indices, val_indices, test_indices = self.splitter(data)
        return data[train_indices], data[val_indices], data[test_indices]


class SimpleSplitter:
    def __init__(
        self,
        split_size_val=0.2,
        split_size_test=0.2,
        random_state=-1,
        shuffle=True,
        stratify=True,
    ):
        self.split_size_val = split_size_val
        self.split_size_test = split_size_test
        self.random_state = None if random_state == -1 else random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def __call__(self, data: StructuredData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_full = np.arange(len(data.target))
        stratify = data.target.value if self.stratify else None

        train_indices, test_indices = train_test_split(
            idx_full,
            test_size=self.split_size_test,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
        )
        stratify = data.target.value[train_indices] if self.stratify else None

        if self.split_size_val == 0:
            return np.sort(train_indices), np.array([]), np.sort(test_indices)

        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.split_size_val,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify,
        )
        return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)
