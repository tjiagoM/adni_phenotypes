import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from datasets import BrainFeaturesDataset
from inference import enable_dropout, mc_passes
from models import SimpleMLP

if __name__ == '__main__':
    device = torch.device('cuda:1')
    run_id = 'tjiagom/adni_phenotypes/2cxy59fk'
    api = wandb.Api()
    best_run = api.run(run_id)
    best_dropout = best_run.config['dropout']

    print('Dropout: ', best_dropout)

    model = SimpleMLP(dim_in=155, dropout_rate=best_dropout).to(device)

    restored_path = wandb.restore('simple_mlp.pt', replace=True, run_path=run_id)
    model.load_state_dict(torch.load(restored_path.name))
    model.eval()

    enable_dropout(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Parameters: ', params)

    dataset = BrainFeaturesDataset('data/adni_test_scaled_corrected.csv', has_target=False, keep_ids=True)
    loader = DataLoader(dataset, batch_size=200, shuffle=False)
    num_samples = 50

    # Warming up the GPU
    for _ in range(5):
        _ = mc_passes(model, loader, num_samples, device, single_batch=True)

    times = []
    for _ in range(1000):
        start_time = time.time()
        _ = mc_passes(model, loader, num_samples, device, single_batch=True)
        times.append((time.time() - start_time) * 1000)

    print(f'Mean: {np.mean(times):.1f} (+- {np.std(times):.2f}) ms')
