import os

os.chdir('/data/lucas/01-experiments/eLW028-active_learning/04-CODE/01-scripts/04-paddle')

import sys
sys.path.insert(0,'/data/lucas/01-experiments/eLW028-active_learning/04-CODE/01-scripts/04-paddle')

import pandas as pd
import paddle
import iupred3_lib as iup
from tqdm import tqdm
from Bio import SeqIO

pad = paddle.PADDLE()
pad_noSS = paddle.PADDLE_noSS()

import torch

from network import S4PRED

def aas2int(seq):
    aanumdict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 
            'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16,
            'W':17, 'Y':18, 'V':19}
    return [aanumdict.get(res, 20) for res in seq]

device = torch.device('cuda') 
s4pred=S4PRED().to('cuda')
s4pred.eval()
s4pred.requires_grad = False

weight_files=['s4pred/weights/weights_1.pt',
            's4pred/weights/weights_2.pt',
            's4pred/weights/weights_3.pt',
            's4pred/weights/weights_4.pt',
            's4pred/weights/weights_5.pt']

s4pred.model_1.load_state_dict(torch.load(weight_files[0],map_location=lambda storage,loc: storage))
s4pred.model_2.load_state_dict(torch.load(weight_files[1],map_location=lambda storage,loc: storage))
s4pred.model_3.load_state_dict(torch.load(weight_files[2],map_location=lambda storage,loc: storage))
s4pred.model_4.load_state_dict(torch.load(weight_files[3],map_location=lambda storage,loc: storage))
s4pred.model_5.load_state_dict(torch.load(weight_files[4],map_location=lambda storage,loc: storage))

def predict_single_tile(prot,use_SS=True):
    if use_SS:
        ind2char = {0:"C",
                    1:"H",
                    2:"E"}
        
        #psipred section
        conf_list = []
        secstring = ''
        iseqs = aas2int(prot)
        with torch.no_grad():
            ss_conf = s4pred(torch.tensor([iseqs]).to(device))
            ss = ss_conf.argmax(-1)
            # move the confidence scores out of log space
            ss_conf = ss_conf.exp()
            # renormalize to assuage any precision issues
            tsum = ss_conf.sum(-1)
            tsum = torch.cat((tsum.unsqueeze(1),tsum.unsqueeze(1),tsum.unsqueeze(1)),1)
            ss_conf /= tsum
            ss = ss.cpu().numpy()
            ss_conf = ss_conf.cpu().numpy()

        for i in range(len(ss)):
            secstring += ind2char[ss[i]]
            conf_list.append([ss_conf[i,0],ss_conf[i,1],ss_conf[i,2]])

        #iupred section
        short = iup.iupred(prot, mode='short')
        long = iup.iupred(prot, mode='long')

        #Generate output
        helix = [sublist[1] for sublist in conf_list]
        coil = [sublist[0] for sublist in conf_list]
        dis_short = [float(x) for x in short[0]]
        dis_long=[float(x) for x in long[0]]
        zscore = pad.predict_protein(prot, helix, coil, dis_short, dis_long)[0]
    else:
        zscore = pad_noSS.predict(prot)
    act = paddle.score2act(zscore)
    return zscore,act

def fasta_to_df(fasta):
    fasta_dict = {}
    fasta_sequences = SeqIO.parse(open(fasta),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        fasta_dict[name] = sequence

    df = pd.DataFrame([fasta_dict]).T.reset_index()
    df.columns = ['tile_id','tile_seq']
    return df

def test_PADDLE(fasta_file: str,
                out_file: str,
                use_SS: bool = True) -> None:
    df = fasta_to_df(fasta_file)
    for idx, row in tqdm(df.iterrows(), desc="Predicting with PADDLE",total=len(df), ncols=80):
        zscore, act = predict_single_tile(row['tile_seq'],use_SS)
        df.loc[idx,'PADDLE_zscore'] = zscore
        df.loc[idx,'PADDLE_activation'] = act
    df.to_csv(f'{out_file}.csv',index=False)

if __name__ == "__main__":
    fasta_file = sys.argv[1]
    out_file = sys.argv[2]
    test_PADDLE(fasta_file,out_file)