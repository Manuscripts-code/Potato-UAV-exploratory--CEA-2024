from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs import configs
from data_manager.loaders import StructuredData
from utils.utils import set_random_seed


class Splitter(ABC):
    def __init__(
        self,
        split_size_val: float = 0.2,
        split_size_test: float = 0.2,
        random_state: int = -1,
        shuffle: bool = True,
        stratify: bool = True,
    ):
        self.split_size_val = split_size_val
        self.split_size_test = split_size_test
        self.random_state = None if random_state == -1 else random_state
        self.shuffle = shuffle
        self.stratify = stratify

    @abstractmethod
    def __call__(self, data: StructuredData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class Sampler:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter

    def sample(self, data: StructuredData) -> tuple[StructuredData, StructuredData, StructuredData]:
        assert (
            len(data.data) == len(data.meta) == len(data.target)
        ), "'data.data', 'data.meta' and 'data.target' must have the same length!"
        set_random_seed(configs.RANDOM_SEED)
        train_indices, val_indices, test_indices = self.splitter(data)
        return data[train_indices], data[val_indices], data[test_indices]


class SimpleSplitter(Splitter):
    def __call__(self, data: StructuredData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_full = np.arange(len(data.target))
        stratify_indices = data.target.value if self.stratify else None

        train_indices, test_indices = train_test_split(
            idx_full,
            test_size=self.split_size_test,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_indices,
        )

        if self.split_size_val == 0:
            return np.sort(train_indices), np.array([]), np.sort(test_indices)

        stratify_indices = data.target.value[train_indices] if self.stratify else None

        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.split_size_val,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_indices,
        )
        return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)


class StratifyAllSplitter(Splitter):
    def __call__(self, data: StructuredData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_full = np.arange(len(data.target))
        stratify_indices = self._create_stratify_indices(data.meta)

        train_indices, test_indices = train_test_split(
            idx_full,
            test_size=self.split_size_test,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_indices,
        )

        if self.split_size_val == 0:
            return np.sort(train_indices), np.array([]), np.sort(test_indices)

        stratify_indices = self._create_stratify_indices(data.meta.iloc[train_indices])

        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.split_size_val,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_indices,
        )
        return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)

    def _create_stratify_indices(self, data: pd.DataFrame) -> np.ndarray:
        # create one label from multiple columns
        label = data[
            [configs.BLOCK_ENG, configs.VARIETY_ENG, configs.TREATMENT_ENG, configs.DATE_ENG]
        ].apply(tuple, axis=1)
        # encode to numbers
        stratify_indices, _ = pd.factorize(label)
        return stratify_indices
