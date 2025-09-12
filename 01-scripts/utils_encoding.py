import os
import shutil
import esm
import torch
import numpy as np
from Bio import SeqIO
import epitopepredict as ep
import pandas as pd
from typing import List
from tqdm import tqdm

from globals import AMINO_ACIDS

def run_esm(in_file: str,
            out_dir: str,
            model_name: str = 'esm2_t33_650M_UR50D',
            layer: int = 33, 
            use_gpu: bool = True
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    data = [(record.id, str(record.seq)) for record in SeqIO.parse(in_file, 'fasta')]

    batch_size = 4
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        if use_gpu and torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        token_representations = results['representations'][layer]
        for j, (label, seq) in enumerate(batch_data):
            embedding = token_representations[j, 1:len(seq)+1].cpu().numpy()
            out_path = f'{out_dir}/{label}.npy'
            np.save(out_path, embedding)

def get_embedding(embedding_dir: str, embedding_file: str, layer=33, extension='npy') -> np.ndarray:
    path = f'{embedding_dir}/{embedding_file}.{extension}'
    if extension == 'npy':
        return np.load(path)
    else:
        return torch.load(path)['representations'][layer].numpy()

def del_esm(embedding_dir: str) -> None:
    shutil.rmtree(embedding_dir, ignore_errors=True)

BLOSUM = ep.blosum62
def blosum_encode(seq: str, 
                  flatten: bool = False
) -> np.ndarray:
    s = list(seq)
    x = pd.DataFrame([BLOSUM[i] for i in seq])
    if flatten: 
        return x.values.flatten()
    else:
        return x.values

def random_encode(seq: str) -> List[int]:
    return [np.random.randint(len(AMINO_ACIDS)) for i in seq]

NLF = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)
def nlf_encode(seq: str,
               flatten: bool = False
) -> np.ndarray:    
    x = pd.DataFrame([NLF[i] for i in seq])
    if flatten:
        return x.values.flatten()
    else:
        return x.values

def one_hot_encode(seq: str,
                   flatten: bool = False
) -> np.ndarray:
    o = list(set(list(AMINO_ACIDS)) - set(seq))
    s = pd.DataFrame(list(seq))  
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    if flatten:
        return a.values.flatten()
    else:
        return a.values