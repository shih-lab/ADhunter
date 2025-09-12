import time
import copy
import torch
import torch.nn as nn
import optuna
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import TensorDataset, DataLoader

from globals import PARAMETER_DIR, PATIENCE, MAX_EPOCHS
from utils_model import train_model

def optimize_parameters_grid(trial,
                             model,
                             criterion,
                             optimizer,
                             dataloaders,
                             dataset_sizes,
                             num_epochs=MAX_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_pearson = 0.0
    best_spearman = 0.0
    best_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  
            running_loss = 0.0
            ys = []
            y_preds = []
            for X, y in dataloaders[phase]:
                # X = X.to(device)
                # y = y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X).reshape(-1)
                    loss = criterion(y_pred.view(-1), y.view(-1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * y.size(0)
                ys.append(y.view(-1))
                y_preds.append(y_pred.view(-1))

            ys = torch.concat(ys).cpu().numpy()
            y_preds = torch.concat(y_preds).detach().cpu().numpy()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_pearson = pearsonr(ys,y_preds).correlation
            epoch_spearman = spearmanr(ys,y_preds).correlation

            print('{} Loss: {:.4f} Pearson: {:.4f} Spearman: {:.4f}'.format(
                phase, epoch_loss, epoch_pearson, epoch_spearman))

            if phase == 'val' and epoch_pearson > best_pearson:
                best_pearson = epoch_pearson
                best_spearman = epoch_spearman
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
        
        trial.report(epoch_pearson, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if epoch > 0 and epoch_loss > best_loss:
            early_stopping_counter += 1
            print(f'epoch loss: {epoch_loss:.2f} best loss: {best_loss}\nadded to early stopping! counter: {early_stopping_counter}')
        else:
            early_stopping_counter = 0

        if early_stopping_counter >= PATIENCE:
            print("Early stopping after", epoch + 1, "epochs with no improvement in validation loss.")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Pearson: {:4f} Spearman: {:4f}'.format(best_pearson, best_spearman))

    model.load_state_dict(best_model_wts)
    return model, best_pearson

def optimize_parameters_optuna(dataset,
                               model,
                               trial,
                               out_file):
    params = {
        "lr": trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        "optimizer_name": trial.suggest_categorical('optimizer_name',["Adam"]),
        "hidden_size": trial.suggest_categorical("hidden", [32, 64, 128, 256, 512, 1024]),
        "kernel_size": trial.suggest_int("kernel_size", 2, 16),
        "dilation": trial.suggest_int("dilation", 1, 16),
        "num_res_blocks": trial.suggest_int("num_res_blocks", 1, 16),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size" : trial.suggest_categorical("batch size", [16, 32, 64, 128, 256, 512])
        }
    train, val, _ = dataset
    X_train, _, y_test = train
    X_val, _, y_val = val

    train_ds = TensorDataset(X_train.to(torch.float), y_test.to(torch.float))
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=False)

    val_ds = TensorDataset(X_val.to(torch.float), y_val.to(torch.float))
    val_dl = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

    dataloaders = {'train':train_dl,'val':val_dl}
    dataset_sizes = {'train':len(X_train),'val':len(X_val)}

    seq_len = X_train[0].shape[0]
    embedding_size = X_train[0].shape[1]

    model = model(
        embedding_size=embedding_size,
        seq_len=seq_len,
        hidden=params['hidden_size'], 
        kernel_size=params['kernel_size'], 
        dilation=params['dilation'], 
        num_res_blocks=params['num_res_blocks'],
        weight_decay=params['weight_decay']
        )

    criterion = nn.MSELoss()
    optimizer = getattr(
        torch.optim, 
        params["optimizer_name"]
    )(model.parameters(), lr=params['lr'])
    best_model, best_pearson = train_model(trial,
                                           model,
                                           criterion,
                                           optimizer,
                                           dataloaders,
                                           dataset_sizes,
                                           num_epochs=MAX_EPOCHS)
    torch.save(best_model.state_dict(),f"{PARAMETER_DIR}/03-optuna_ADhunter_v2/optimized_CNN_{trial.number}.pt")
    return best_pearson
