import os
import sys
sys.path.insert(1, '../01-scripts')

from utils_encoding import run_esm, get_embedding
from utils_model import set_seed
from utils_bio import fasta_to_df

import argparse
import torch

def parse_args():
    p = argparse.ArgumentParser(
        description='Run ESM.'
    )
    p.add_argument('--fasta-file', required=True, help='Path to FASTA file.')
    p.add_argument('--use-gpu', action='store_true', help='Use GPU if available.')
    p.add_argument('--run-esm', dest='run_esm', action='store_true',
                   help='Generate embeddings with ESM before inference.')
    p.add_argument('--output-file', required=True, help='Output file containing ESM embeddings.')
    return p.parse_args()

def main():
    set_seed(seed=0)
    args = parse_args()

    device = 'cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
    if args.use_gpu and device != 'cuda':
        print('CUDA is not available. Falling back to CPU.')

    embedding_path = './embeddings'
    os.makedirs(embedding_path)
    run_esm(args.fasta_file, embedding_path, use_gpu=(device == 'cuda'))

    embedding_df = fasta_to_df(args.fasta_file)
    embedding_df['esm2_t33_650M_UR50D'] = embedding_df['ID'].apply(
        lambda x: get_embedding(embedding_path, x)
    )
    if args.output_file:
        embedding_df.to_pickle(f'{args.output_file}.pkl')
    else:
        embedding_df.to_pickle('esm_embeddings.pkl')
    
if __name__ == '__main__':
    main()