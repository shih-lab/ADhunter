import torch
import torch.nn as nn

from models import BaseSystem, ResBlock
from globals import (HIDDEN_SIZE, KERNEL_SIZE, DILATION, 
                     NUM_RES_BLOCKS, SEQ_LEN, AMINO_ACIDS, 
                     WEIGHT_DECAY, LEARNING_RATE)

class ADhunter(nn.Module):
    def __init__(self,
                 encoding: str = 'integer',
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 dilation: int =  DILATION,
                 num_res_blocks: int = NUM_RES_BLOCKS,
                 seq_len: int = SEQ_LEN):
        super().__init__()

        if encoding == 'integer':
            self.use_embedding = True
            self.embedding = nn.Embedding(num_embeddings=embedding_size, embedding_dim=hidden_size)
            conv_in_channels = hidden_size
        else:
            self.use_embedding = False
            conv_in_channels = embedding_size

        self.conv_init = nn.Conv1d(
            in_channels=conv_in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding="same"
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size=hidden_size, kernel_size=kernel_size, dilation=dilation)
            for _ in range(num_res_blocks)
        ])
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_embedding:
            out = self.embedding(X)
        else:
            out = X
        out = out.transpose(2, 1)
        out = self.conv_init(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.pool(out).squeeze(-1)
        out = self.lin(out)
        return out

class ADhunterSystem(BaseSystem):
    def __init__(self,
                 encoding: str = 'integer',
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 dilation: int = DILATION,
                 num_res_blocks: int = NUM_RES_BLOCKS,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE,
                 seq_len: int = SEQ_LEN):
        self.save_hyperparameters()
        model = ADhunter(
            encoding,
            embedding_size,
            hidden_size,
            kernel_size,
            dilation,
            num_res_blocks,
            seq_len)
        super().__init__(model,weight_decay,learning_rate)
