import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from datasets import BrainFeaturesDataset
from models import SimpleMLP


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            print('Enabling dropout layer.')
            each_module.train()


def mc_passes(model, loader, n_passes: int, device: str, single_batch: bool = False) -> Tuple[list, np.ndarray, np.ndarray]:
    mean_predictions = []
    std_predictions = []
    id_predictions = []
    for id_batch, X_batch, _ in loader:
        id_batch, X_batch = id_batch, X_batch.to(device)

        all_preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                all_preds.append(model(X_batch))

        mean_batch = torch.mean(torch.stack(all_preds), dim=0)
        std_batch = torch.std(torch.stack(all_preds), dim=0)

        mean_predictions.append(mean_batch.squeeze().detach().cpu().numpy())
        std_predictions.append(std_batch.squeeze().detach().cpu().numpy())

        # For those IDs that are strings rather than integers
        if type(id_batch) != tuple:
            id_batch = id_batch.cpu().numpy()

        id_predictions.append(id_batch)

        if single_batch:
            break

    return np.hstack(id_predictions).tolist(), np.hstack(mean_predictions), np.hstack(std_predictions)


def run_inference(dataset_location: str, dataset_id: str, single_pass: bool, device: str, wb_model_id: str) -> None:
    if single_pass:
        print('Warning: Single pass activated. Not using MC Dropout!')

    device = torch.device(device)
    run_id = f'tjiagom/adni_phenotypes/{wb_model_id}'
    api = wandb.Api()
    best_run = api.run(run_id)

    model = SimpleMLP(dim_in=155, dropout_rate=best_run.config['dropout']).to(device)

    restored_path = wandb.restore('simple_mlp.pt', replace=True, run_path=run_id)
    model.load_state_dict(torch.load(restored_path.name))
    model.eval()

    if not single_pass:
        enable_dropout(model)

    dataset = BrainFeaturesDataset(dataset_location, has_target=False, keep_ids=True)
    loader = DataLoader(dataset, batch_size=200, shuffle=False)

    if single_pass:
        num_samples = 1
    else:
        num_samples = 50

    ids, means, stds = mc_passes(model, loader, num_samples, device)

    ret_df = pd.DataFrame(list(zip(ids, means, stds)), columns=[f'{dataset_id}_id', 'mean', 'std'])
    ret_df = ret_df.set_index(f'{dataset_id}_id')

    if run_id == '2cxy59fk':
        ret_df.to_csv(f'results/latest_output_{dataset_id}_{num_samples}.csv')
    else:
        ret_df.to_csv(f'results/latest_output_{dataset_id}_{num_samples}_{wb_model_id}.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='ADNI Phenotypes')
    parser.add_argument('--dataset_location',
                        type=str,
                        choices=['data/ukb_scaled_corrected.csv',
                                 'data/nacc_scaled_corrected.csv',
                                 'data/adni_test_scaled_corrected.csv',
                                 'data/adni_train_scaled_corrected.csv'],
                        help='The location of the dataset.')

    parser.add_argument('--dataset_id',
                        type=str,
                        choices=['ukb', 'nacc', 'adni', 'adni_train'],
                        help='Small identification of dataset.')
    
    parser.add_argument('--wb_model_id',
                        type=str,
                        default='2cxy59fk',
                        help='Wandb run ID to download trained model.')

    parser.add_argument('--do_single_pass',
                        action='store_true',
                        help='Whether to do one single pass, rather than MC-Drop.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='Which GPU device to use.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    run_inference(dataset_location=args.dataset_location, dataset_id=args.dataset_id,
                  single_pass=args.do_single_pass, device=args.device, wb_model_id=args.wb_model_id)
