import esm
import torch
import numpy as np
from Bio import SeqIO
from typing import Tuple
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from globals import DataSplit, SplitResult

def convert_split_to_tensor(
    dataset: DataSplit
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y_bin, y_cont = dataset
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32) if arr.ndim > 1 else torch.tensor(arr)
    return to_tensor(X), to_tensor(y_bin), to_tensor(y_cont)

def get_stratified_split(X: np.ndarray,
                         y_bin: np.ndarray,
                         y_cont: np.ndarray,
                         train_size: float = 0.8,
                         random_state: int = 0,
                         as_tensor: bool = True
) -> SplitResult:
    splitter = StratifiedShuffleSplit(n_splits=1,train_size=train_size,random_state=random_state)
    train_idx, val_test_idx = list(splitter.split(X, y_bin))[0]

    X_train = X[train_idx]
    y_bin_train = y_bin[train_idx]
    y_cont_train = y_cont[train_idx]

    X_val_test = X[val_test_idx]
    y_bin_val_test = y_bin[val_test_idx]
    y_cont_val_test = y_cont[val_test_idx]

    splitter = StratifiedShuffleSplit(n_splits=1,train_size=0.5,random_state=random_state)
    val_index, test_index = list(splitter.split(X_val_test, y_bin_val_test))[0]
    X_val = X_val_test[val_index]
    y_bin_val = y_bin_val_test[val_index]
    y_cont_val = y_cont_val_test[val_index]

    X_test = X_val_test[test_index]
    y_bin_test = y_bin_val_test[test_index]
    y_cont_test = y_cont_val_test[test_index]

    val_idx = val_test_idx[val_index]
    test_idx = val_test_idx[test_index]

    X_val = X[val_idx]
    y_bin_val = y_bin[val_idx]
    y_cont_val = y_cont[val_idx]

    X_test = X[test_idx]
    y_bin_test = y_bin[test_idx]
    y_cont_test = y_cont[test_idx]

    train = (X_train, y_bin_train, y_cont_train)
    val = (X_val, y_bin_val, y_cont_val)
    test = (X_test, y_bin_test, y_cont_test)

    if as_tensor:
        train = convert_split_to_tensor(train)
        val = convert_split_to_tensor(val)
        test = convert_split_to_tensor(test)
    data_split = (train, val, test)
    
    idx_split = (train_idx, val_idx, test_idx)

    return  data_split, idx_split
        

def split_dataset(X: np.ndarray,
                  y: np.ndarray,
                  thresh: float,
                  random_state: int = 0,
                  train_size: float = 0.8,
                  as_tensor: bool = True,
                  scaler: str = 'standard'
) -> SplitResult:
    y_bin = (y >= thresh).astype(np.int64).reshape(-1, 1)
    y_cont = y.reshape(-1, 1)
    if scaler == 'standard':
        y_cont = preprocessing.StandardScaler().fit_transform(y_cont)
    elif scaler == 'minmax':
        y_cont = preprocessing.MinMaxScaler().fit_transform(y_cont)
    elif scaler == 'both':
        y_cont = preprocessing.MinMaxScaler().fit_transform(y_cont)
        y_cont = preprocessing.StandardScaler().fit_transform(y_cont)        
    else:
        raise Exception('invalid scaler choice')
    dataset, idx = get_stratified_split(X,
                                        y_bin,
                                        y_cont,
                                        train_size,
                                        random_state,
                                        as_tensor)
    return dataset, idx