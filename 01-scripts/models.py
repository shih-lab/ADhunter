import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
from typing import Tuple, List, Dict

from globals import (HIDDEN_SIZE, KERNEL_SIZE, DILATION, 
                     NUM_RES_BLOCKS, SEQ_LEN, AMINO_ACIDS, 
                     WEIGHT_DECAY, LEARNING_RATE)

class ResBlock(nn.Module):
    def __init__(self,
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 dilation: int = DILATION):
        super().__init__()
        self.bn_1 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu_1 = nn.ReLU()
        self.conv_res = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation
        )
        self.bn_2 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu_2 = nn.ReLU()
        self.conv_block = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            padding="same")

    def forward(self, X: torch.Tensor) -> torch.Tensor :
        out = self.bn_1(X)
        out = self.relu_1(out)
        out = self.conv_res(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.conv_block(out)
        out = out + X
        return out
    
class BaseSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, weight_decay: float, learning_rate: float):
        super().__init__()
        self.model = model
        self.wd = weight_decay
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.metrics = {
            "rmse": MeanSquaredError(squared=False),
            "pearsonr": PearsonCorrCoef(),
            "spearmanr": SpearmanCorrCoef()
        }
    def on_fit_start(self) -> None:
        for metric in self.metrics.values():
            metric.to(self.device)

    def forward(self, X: torch.Tensor)  -> torch.Tensor:
        return self.model(X)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.wd)
        return optimizer

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> dict:
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return {
            "loss": loss,
            "y_target": y.view(-1),
            "y_pred": y_pred.detach().view(-1),
        }

    def training_epoch_end(self, train_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        y_preds = [d['y_pred'] for d in train_step_outputs]
        y_targets = [d['y_target'] for d in train_step_outputs]
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        train_loss = self.metrics['rmse'](y_preds, y_targets)
        for metric_name, metric in self.metrics.items():
            metric_name = "train_" + metric_name
            self.log(metric_name, metric(y_preds, y_targets))

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> Tuple:
        X, y = batch
        y_pred = self.model(X)
        return (y_pred.view(-1), y.view(-1))

    def validation_epoch_end(self, val_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        y_preds, y_targets = zip(*val_step_outputs)
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        val_loss = self.metrics['rmse'](y_preds, y_targets)
        self.log("val_loss", val_loss)
        for metric_name, metric in self.metrics.items():
            metric_name = "val_" + metric_name
            print(metric_name, metric(y_preds, y_targets).item(), flush=True)
            self.log(metric_name, metric(y_preds, y_targets))
        return val_loss
        
class NN(nn.Module):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE
                 ):
        super().__init__()
        self.fc1 = nn.Linear(in_features=embedding_size,out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size,out_features=1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(X))
        out = self.fc2(out)
        return out
    
class CNN(nn.Module):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 seq_len: int = SEQ_LEN):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=hidden_size, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = X.transpose(2,1)
        out = self.pool(self.conv(out)).squeeze()
        out = self.fc(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 dilation: int = DILATION,
                 num_res_blocks: int = NUM_RES_BLOCKS,
                 seq_len: int = SEQ_LEN):
        super().__init__()
        self.conv_init = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding="same"
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size,kernel_size,dilation)
            for _ in range(num_res_blocks)
        ])
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.lin = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X) -> torch.Tensor:
        out = X.transpose(2, 1)
        out = self.conv_init(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.pool(out).squeeze()
        out = self.lin(out)
        return out

class RNN(nn.Module):
    def __init__(self, 
                 embedding_size: int = len(AMINO_ACIDS), 
                 hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.rnn = nn.RNN(embedding_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(X)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS), 
                 hidden_size: int = HIDDEN_SIZE, 
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)

        self.linear = nn.Linear(in_features=hidden_size * self.num_directions, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=X.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=X.device)

        out, (hn, cn) = self.lstm(X, (h0, c0))

        if self.bidirectional:
            forward_final = out[:, -1, :self.hidden_size]
            backward_final = out[:, 0, self.hidden_size:]
            final_output = torch.cat((forward_final, backward_final), dim=1)
        else:
            final_output = out[:, -1, :]

        output = self.linear(final_output)
        return output

class Transformer(nn.Module):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 num_transformer_heads: int = 8,
                 num_encoder_layers: int = 3 ,
                 feedforward_size: int = 2048,
                 dropout: int = 0.1):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_transformer_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)
        self.linear = nn.Linear(embedding_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1,0)
        out = self.transformer_encoder(out)
        out = self.linear(out[-1])
        return out

class NNSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE):
        model = NN(embedding_size,hidden_size)
        super().__init__(model,weight_decay,learning_rate)
        self.save_hyperparameters()

class CNNSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE,
                 seq_len: int = SEQ_LEN):
        self.save_hyperparameters()
        model = CNN(embedding_size,
                    hidden_size,
                    kernel_size,
                    seq_len)
        super().__init__(model,weight_decay,learning_rate)

class ResNetSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 kernel_size: int = KERNEL_SIZE,
                 dilation_size: int = DILATION,
                 num_res_blocks: int = NUM_RES_BLOCKS,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE,
                 seq_len: int = SEQ_LEN):
        self.save_hyperparameters()
        model = ResNet(embedding_size,
                        hidden_size,
                        kernel_size,
                        dilation_size,
                        num_res_blocks,
                        seq_len)
        super().__init__(model,weight_decay,learning_rate)
    
class RNNSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE):
        self.save_hyperparameters()
        model = RNN(embedding_size,hidden_size)
        super().__init__(model,weight_decay,learning_rate)

class LSTMSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = 2,
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE,
                 bidirectional: bool = False):
        self.save_hyperparameters()
        model = LSTM(embedding_size, 
                          hidden_size,
                          num_layers,
                          bidirectional=bidirectional)
        super().__init__(model,weight_decay,learning_rate)

class TransformerSystem(BaseSystem):
    def __init__(self,
                 embedding_size: int = len(AMINO_ACIDS),
                 num_transformer_heads: int = 8,
                 num_encoder_layers: int = 3,
                 feedforward_size: int = 2048, 
                 weight_decay: float = WEIGHT_DECAY,
                 learning_rate: float = LEARNING_RATE):
        self.save_hyperparameters()
        model = Transformer(embedding_size,
                            num_transformer_heads,
                            num_encoder_layers,
                            feedforward_size)
        super().__init__(model,weight_decay,learning_rate)