import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SineDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1, seed=0, train=True):
        np.random.seed(seed)
        self.x = np.random.uniform(-np.pi, np.pi, n_samples)
        self.y = np.sin(self.x) + np.random.normal(0, noise, n_samples)
        self.x = torch.tensor(self.x, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.tensor(self.y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_sine_dataloader(n_samples=1000, noise=0.1, seed=0, batch_size=32, train=True):
    dataset = SineDataset(n_samples, noise, seed, train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
