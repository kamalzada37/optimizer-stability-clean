# src/data.py
import torch
from torchvision import datasets, transforms
import numpy as np

def get_mnist(root='../data', train=True, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST(root, train=train, download=download, transform=transform)
    return ds

def inject_label_noise(dataset, noise_rate, num_classes=10, seed=0):
    """
    Inject uniform random label noise into torchvision MNIST dataset object.
    Returns the modified dataset (in-place).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Convert to tensor if needed
    targets = torch.as_tensor(dataset.targets, dtype=torch.long).clone()
    n = len(targets)
    num_noisy = int(noise_rate * n)
    if num_noisy == 0:
        print(f'Labels flipped: 0 out of {n}')
        dataset.targets = targets
        return dataset
    idx = np.random.choice(n, num_noisy, replace=False)
    new_labels = torch.randint(0, num_classes, (num_noisy,), dtype=targets.dtype)
    # ensure changed
    for i, index in enumerate(idx):
        orig = int(targets[index].item())
        while int(new_labels[i].item()) == orig:
            new_labels[i] = torch.randint(0, num_classes, (1,), dtype=targets.dtype)
    targets[idx] = new_labels
    dataset.targets = targets
    print(f'Labels flipped: {num_noisy} out of {n}')
    return dataset

def get_sine_data(n_samples=1000, noise=0.0, seed=0):
    """
    Return tensors (x, y) for sine regression.
    x shape (n,1), y shape (n,1)
    """
    np.random.seed(seed)
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_samples)
    y = np.sin(x).astype(np.float32)
    if noise > 0:
        y = y + np.random.normal(0, noise, size=y.shape).astype(np.float32)
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return x_t, y_t
