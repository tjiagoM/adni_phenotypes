import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, average_precision_score
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from datasets import BrainFeaturesDataset
from models import SimpleMLP


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def calculate_metrics(labels, pred_prob, pred_binary, loss_value) -> Dict[str, float]:
    return {'loss': loss_value,
            'roc': roc_auc_score(labels, pred_prob),
            'p-r': average_precision_score(labels, pred_prob),
            'acc': accuracy_score(labels, pred_binary),
            'f1': f1_score(labels, pred_binary, zero_division=0),
            'sensitivity': recall_score(labels, pred_binary, zero_division=0),
            'specificity': recall_score(labels, pred_binary, pos_label=0, zero_division=0)}


def log_to_wandb(train_metrics, val_metrics):
    train_dict = {f'train_{elem[0]}': elem[1] for elem in train_metrics.items()}
    val_dict = {f'val_{elem[0]}': elem[1] for elem in val_metrics.items()}

    wandb.log({**train_dict, **val_dict})


def model_forward_pass(model, loader, is_train, device, criterion, optimiser=None) -> Dict[str, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    # For evaluation
    predictions = []
    labels = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if is_train:
            optimiser.zero_grad()
            y_pred = model(X_batch)
            # y_pred: BN X 1, y_batch: BN
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))

        epoch_loss += loss.item()

        if not is_train:
            predictions.append(y_pred.squeeze().detach().cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    if not is_train:
        predictions = np.hstack(predictions)
        pred_binary = np.where(predictions > 0.5, 1, 0)
        labels = np.hstack(labels)

        return calculate_metrics(labels, predictions, pred_binary,
                                 loss_value=epoch_loss / len(loader))
    else:
        return {'loss': epoch_loss / len(loader)}


def train_simple_mlp(balance_dataset=False, device='cuda:1'):
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.8

    wandb.config.lr = LEARNING_RATE
    wandb.config.weight_decay = WEIGHT_DECAY
    wandb.config.dropout = DROPOUT_RATE

    train_dataset = BrainFeaturesDataset('data/adni_train_scaled_corrected.csv')
    val_dataset = BrainFeaturesDataset('data/adni_test_scaled_corrected.csv')
    
    if balance_dataset:
        print('Running training with balanced train set!')
        ids = [elem[1] for elem in train_dataset]
        # Removing 60 elements of Control people for balanced dataset
        updated_ids = sorted(list(np.where(np.array(ids) == 0)[0][:-60]) + list(np.where(np.array(ids) == 1)[0]))
        train_dataset = torch.utils.data.Subset(train_dataset, updated_ids)

    all_labels = [elem[1] for elem in train_dataset]
    print('Distribution of labels:', np.unique(all_labels, return_counts=True))
    
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)

    device = torch.device(device)

    model = SimpleMLP(dim_in=next(iter(train_loader))[0].shape[1], dropout_rate=DROPOUT_RATE).to(device)
    wandb.watch(model, log='all', log_freq=2) # setting to log_freq=1 significantly slows down the script

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = BCELoss()

    ##
    # Main cycle
    best_loss = 1000
    epoch_best = -1
    for e in range(1, EPOCHS + 1):
        _ = model_forward_pass(model=model, loader=train_loader, is_train=True,
                               device=device, optimiser=optimiser, criterion=criterion)
        train_metrics = model_forward_pass(model=model, loader=train_loader, is_train=False,
                                           device=device, criterion=criterion)
        val_metrics = model_forward_pass(model=model, loader=val_loader, is_train=False,
                                         device=device, criterion=criterion)

        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            epoch_best = e
            torch.save(model.state_dict(), 'saved_models/simple_mlp.pt')
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'simple_mlp.pt'))

        log_to_wandb(train_metrics, val_metrics)
        print(f'{e + 0:03}| L: {train_metrics["loss"]:.3f} / {val_metrics["loss"]:.3f}'
              f' | Acc: {train_metrics["acc"]:.2f} / {val_metrics["acc"]:.2f}'
              f' | ROC: {train_metrics["roc"]:.2f} / {val_metrics["roc"]:.2f}'
              f' | P-R: {train_metrics["p-r"]:.2f} / {val_metrics["p-r"]:.2f}')

    print(f'Best val loss {best_loss:.2f} at epoch {epoch_best}.')
    wandb.run.summary['best_val_loss'] = best_loss
    wandb.run.summary['best_val_epoch'] = epoch_best

    
def parse_args():
    parser = argparse.ArgumentParser(description='ADNI training')

    parser.add_argument('--balance_dataset',
                        action='store_true',
                        help='Whether to train on a balanced dataset.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='Which GPU device to use.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    wandb.init(project='adni_phenotypes', save_code=True)
    train_simple_mlp(balance_dataset=args.balance_dataset, device=args.device)
