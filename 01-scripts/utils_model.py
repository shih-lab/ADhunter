import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Union, List, Dict
from tqdm import tqdm
import random
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar)
from pytorch_lightning.loggers import CSVLogger

from globals import BATCH_SIZE, PATIENCE, MAX_EPOCHS, DataSplit

class EmacsProgressBar(TQDMProgressBar):
    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(desc="Training",
                   initial=self.train_batch_idx,
                   position=(2 * self.process_position))
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(desc="Validating",
                   position=(2 * self.process_position),
                   disable=True)
        return bar

def force_cudnn_initialization() -> None:
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev),
        torch.zeros(s, s, s, s, device=dev))

def set_seed(seed: int = 0) -> None:
    force_cudnn_initialization()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)

def preprocess_tensor(unscaled_tensor: np.ndarray,
                      scaler: str = 'standard'
) -> np.ndarray:
    unscaled_tensor_2d = unscaled_tensor.reshape(-1, unscaled_tensor.shape[-1])

    if scaler == 'standard':
        scaled_tensor_2d = preprocessing.StandardScaler().fit_transform(unscaled_tensor_2d)
    elif scaler == 'minmax':
        scaled_tensor_2d = preprocessing.MinMaxScaler().fit_transform(unscaled_tensor_2d)
    elif scaler == 'both':
        scaled_tensor_2d = preprocessing.StandardScaler().fit_transform(unscaled_tensor_2d)        
        scaled_tensor_2d = preprocessing.MinMaxScaler().fit_transform(scaled_tensor_2d)
    else:
        raise ValueError(f"Invalid scaler type: {scaler}. Choose from 'standard', 'minmax', or 'both'.")
    
    scaled_tensor_3d = scaled_tensor_2d.reshape(unscaled_tensor.shape)
    return scaled_tensor_3d

def get_threshold(y_bin_test: np.ndarray,
                  y_test_hat: np.ndarray
) -> float:
    fpr, tpr, thresholds = roc_curve(y_bin_test, y_test_hat)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_thresh = thresholds[ix]
    return best_thresh

def train_model(dataset: Tuple[DataSplit, DataSplit, DataSplit],
          model_dict: Dict[str, pl.LightningModule], 
          out_dir: str,
          encoding: str = 'integer',
          is_ensemble: bool = False,
          random_state: int = 0,
          patience: int = PATIENCE,
          batch_size: int = None,
) -> List[pl.LightningModule]:
    train, val, _ = dataset
    (X_train, _, y_train) = train
    (X_val, _, y_val) = val

    os.makedirs(f'{out_dir}/03-results',exist_ok=True)

    def to_float(x): return x.to(torch.float) if encoding != 'integer' else x

    for name, model in model_dict.items():
        out_file = f"{name}_state{random_state}"
        if not batch_size:
            batch_size = int(name.split('_b')[-1].split('_')[0])

        train_ds = TensorDataset(to_float(X_train), y_train.to(torch.float))
        val_ds = TensorDataset(to_float(X_val), y_val.to(torch.float))
        train_dl = DataLoader(train_ds,batch_size,shuffle=False)
        val_dl = DataLoader(val_ds,batch_size,shuffle=False)
        
        csv_logger = CSVLogger(f"{out_dir}/01-logs",name=out_file,version='')
        checkpoint_callback = ModelCheckpoint(dirpath=f"{out_dir}/02-models",
                                              monitor="val_loss",
                                              filename=out_file,
                                              save_last=False)
        early_stopping = EarlyStopping('val_loss',patience=PATIENCE)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             callbacks=[EmacsProgressBar(),checkpoint_callback,early_stopping],
                             logger=[csv_logger],
                             max_epochs=MAX_EPOCHS)
        trainer.fit(model,
                    train_dataloaders=train_dl,
                    val_dataloaders=val_dl)
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
        torch.save(model.cpu().state_dict(),f'{out_dir}/02-models/{out_file}.pt')
        train_df = pd.read_csv(f'{out_dir}/01-logs/{out_file}/metrics.csv')
        train_df = train_df.groupby('epoch').max().reset_index()
        train_df.to_pickle(f'{out_dir}/03-results/{out_file}_train_df.pkl')

def test_model(test_dataset: DataSplit,
               model_dict: Dict[str, pl.LightningModule], 
               model_dir: str,
               out_dir: str,
               encoding: str = 'integer',
               random_state: int = 0,
               batch_size: int = BATCH_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (X_test, y_bin_test, y_test) = test_dataset
    def to_float(x): return x.to(torch.float) if encoding != 'integer' else x

    os.makedirs(f'{out_dir}/03-results',exist_ok=True)

    if y_test is not None:
        test_dataset = TensorDataset(to_float(X_test), y_test.to(torch.float))
    else:
        test_dataset = TensorDataset(to_float(X_test))  # Only features, no labels

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    if len(model_dict) == 1:
        for name, model in tqdm(model_dict.items()):
            model.load_state_dict(torch.load(f'{model_dir}/{name}.pt'))

            y_test_hat = []
            y_test_actual = []
            model.eval()
            with torch.no_grad():
                if y_test is not None:
                    for xb, yb in test_loader:
                        preds = model(xb)
                        y_test_hat.append(preds.numpy().reshape(-1))
                        y_test_actual.append(yb.reshape(-1).detach().cpu())
                else:
                    for xb in test_loader:
                        for x in xb:
                            pred = model(x)
                            y_test_hat.append(pred.numpy().reshape(-1))

            if y_bin_test is not None:
                test_df = pd.DataFrame({
                    'y_test_hat': np.concatenate(y_test_hat,axis=0),
                    'y_test': np.concatenate(y_test_actual,axis=0),
                    'y_bin_test': y_bin_test.reshape(-1).numpy(),
                })
            else:
                test_df = pd.DataFrame({
                    'y_test_hat': np.concatenate(y_test_hat,axis=0),
                    'y_test': np.concatenate(y_test_actual,axis=0),
                })
            test_df.to_pickle(f'{out_dir}/03-results/{name}_test_df.pkl')
    else:
        ensemble_preds = []
        actuals = []
        for name, model in tqdm(model_dict.items()):
            model.load_state_dict(torch.load(f'{model_dir}/{name}.pt'))
            model.eval()
            fold_preds = []

            with torch.no_grad():
                if y_test is not None:
                    all_preds = []
                    for xb, yb in test_loader:
                        preds = model(xb)
                        all_preds.append(preds.detach().cpu().numpy())
                        if name == list(model_dict.keys())[0]:
                            actuals.append(yb.reshape(-1).detach().cpu().numpy())
                    fold_preds = np.concatenate(all_preds, axis=0)
                else:
                    all_preds = []
                    for xb in test_loader:
                        preds = model(xb)
                        all_preds.append(preds.detach().cpu().numpy())
                    fold_preds = np.concatenate(all_preds, axis=0)

            ensemble_preds.append(fold_preds)

        preds = np.mean(ensemble_preds, axis=0)
        uncertainty = np.var(ensemble_preds, axis=0)

        test_df = pd.DataFrame({
            'y_test': y_test.detach().cpu().numpy().flatten(),
            'y_test_hat': preds.flatten(),
            'uncertainty': uncertainty.flatten()
        })
        if y_bin_test is not None:
            test_df['y_bin_test'] = y_bin_test.reshape(-1).numpy()
    return test_df

def evaluate_model(dataset,
                   model,
                   out_dir: str,
                   out_file: str,
                   encoding: str = 'integer',
                   is_ensemble: bool = False,
                   random_state: int = 0,
                   batch_size: int = BATCH_SIZE
) -> None:
    trained_model = train_model(dataset,
                                 model,
                                 out_dir,
                                 out_file,
                                 encoding,
                                 is_ensemble,
                                 random_state,
                                 batch_size)
    test_model(dataset,
                trained_model,
                out_dir,
                out_file,
                encoding,
                is_ensemble,
                random_state,
                batch_size)

def scan_optimal_threshold():
    return ...

def split_seq_similarity():
    return ...

def split_spectral():
    return ...

def evaluate_esm_additivity():
    return ...