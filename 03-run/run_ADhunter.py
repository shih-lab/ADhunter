import sys
sys.path.insert(1, '../01-scripts')

from adhunter import ADhunterSystem

import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

MODEL_DIR = './models'

def get_ensemble_models(df,X,device='cpu'):
    ensemble_models = []
    emb_size = X[0].shape[1]
    seq_len = X[0].shape[0]

    for _, row in df.iterrows():
        model = ADhunterSystem(
            encoding='esm',
            embedding_size=emb_size,
            seq_len=seq_len,
            hidden_size=row['hidden_size'],
            kernel_size=row['kernel_size'],
            dilation=row['dilation'],
            num_res_blocks=row['num_res_blocks'],
        ).to(device)

        model_file = (
            f'ADhunter_h{row["hidden_size"]}'
            f'_k{row["kernel_size"]}'
            f'_d{row["dilation"]}'
            f'_r{row["num_res_blocks"]}'
            f'_b{row["batch_size"]}_harmonized'
        )

        weight_path = f'{MODEL_DIR}/{model_file}.pt'
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        ensemble_models.append(model)

    return ensemble_models

def ensemble_predict(models, X_test, device = 'cpu'):
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = X_test.to(device)

    y_hats = []
    for model in tqdm(models):
        model.eval()
        with torch.no_grad():
            y_hat = model(X_test)
            y_hat = y_hat.detach().cpu().numpy()
        y_hats.append(y_hat)

    y_hats = np.array(y_hats)
    variances = np.squeeze(np.var(y_hats, axis=0))
    means = np.squeeze(np.mean(y_hats, axis=0))
    return means, variances

def parse_args():
    p = argparse.ArgumentParser(
        description='Run ADhunter ensemble predictions from ESM embeddings.'
    )
    p.add_argument('--embedding-file', required=True, help='Path to FASTA file.')
    p.add_argument('--output-file', required=True, help='Output file containing ADhunter predictions and uncertainty.')
    p.add_argument('--use-gpu', action='store_true', help='Use GPU if available.')
    return p.parse_args()

def main():
    args = parse_args()

    device = 'cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
    if args.use_gpu and device != 'cuda':
        print('CUDA is not available. Falling back to CPU.')

    embedding_df = pd.read_pickle(args.embedding_file)

    X_test = np.asarray([np.array(emb) for emb in embedding_df['esm2_t33_650M_UR50D']], dtype=np.float32)
    embedding_df = embedding_df.drop(columns=['esm2_t33_650M_UR50D'])

    param_df = pd.read_csv('ADhunter_ensemble_params_n20.csv')

    ensemble_models = get_ensemble_models(param_df, X_test, device=device)
    predictions, uncertainties = ensemble_predict(ensemble_models, X_test, device=device)

    embedding_df['prediction'] = predictions
    embedding_df['uncertainty'] = uncertainties

    embedding_df.to_csv(f'{args.output_file}.csv', index=False)

if __name__ == '__main__':
    main()