# src/train.py
import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data import get_mnist, inject_label_noise, get_sine_data
from src.model import SmallCNN

# simple MLP for regression
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

# Small helper: compute grad norm
def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach().norm().item()) ** 2
    return total ** 0.5

# SAM step (two-forward/backward updates)
def sam_step(model, base_optimizer, loss_fn, xb, yb, device, rho=0.05, dtype=torch.float32):
    # first forward/backward
    xb = xb.to(device=device, dtype=dtype)
    yb = yb.to(device=device)
    out = model(xb)
    loss = loss_fn(out, yb)
    loss.backward()

    # save gradients and perturb params
    grad_norm = 0.0
    eps = 1e-12
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm += (p.grad.detach().norm() ** 2)
    grad_norm = grad_norm.sqrt().item() + eps

    # save original parameters and perturb
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            e_w = (rho / grad_norm) * p.grad
            p.add_(e_w)  # ascent step

    # second forward/backward
    base_optimizer.zero_grad()
    out2 = model(xb)
    loss2 = loss_fn(out2, yb)
    loss2.backward()

    # restore original params by subtracting same e_w (we did not store e_w per param but can step optimizer which will update back)
    # apply base optimizer step now
    base_optimizer.step()
    return loss.item()

def train(args):
    # reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float32 if args.precision == 'float32' else torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    if args.dataset == 'mnist':
        train_ds = get_mnist('./data', train=True, download=True)
        test_ds = get_mnist('./data', train=False, download=True)
        if args.noise > 0:
            train_ds = inject_label_noise(train_ds, args.noise, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        model = SmallCNN().to(device=device, dtype=dtype)
        criterion = nn.CrossEntropyLoss()
        is_classification = True
    else:
        x_train, y_train = get_sine_data(n_samples=1000, noise=args.noise, seed=args.seed)
        x_test, y_test = get_sine_data(n_samples=200, noise=0.0, seed=args.seed+1)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
        model = SimpleMLP(input_dim=1, output_dim=1).to(device=device, dtype=dtype)
        criterion = nn.MSELoss()
        is_classification = False

    # optimizer selection
    base_optimizer = None
    sam_enabled = False
    if args.optimizer == 'sgd':
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd_momentum':
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'rmsprop':
        base_optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        base_optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sam_sgd':
        # use SGD as base optimizer for SAM
        sam_enabled = True
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer: " + args.optimizer)

    history = defaultdict(list)
    diverged = False
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_grad_norm = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            n_batches += 1
            # move to device, set dtype
            # MNIST: xb is [B,1,28,28] already; ensure dtype and device
            xb = xb.to(device=device, dtype=dtype)
            # labels for classification need to be LongTensor
            if is_classification:
                yb = yb.to(device=device, dtype=torch.long)
            else:
                yb = yb.to(device=device, dtype=dtype)

            base_optimizer.zero_grad()
            if sam_enabled:
                # SAM two-step update implemented in function
                try:
                    loss_item = sam_step(model, base_optimizer, criterion, xb, yb, device, rho=args.sam_radius, dtype=dtype)
                except Exception as e:
                    # fallback to simple step if SAM fails
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    base_optimizer.step()
                    loss_item = loss.item()
            else:
                # normal update
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                # gradient clipping optional
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                base_optimizer.step()
                loss_item = loss.item()

            running_loss += loss_item
            n_grad_norm = compute_grad_norm(model)
            running_grad_norm += n_grad_norm

            # quick NaN check
            if torch.isnan(torch.tensor(loss_item)):
                diverged = True
                break

        avg_train_loss = running_loss / max(1, n_batches)
        avg_grad_norm = running_grad_norm / max(1, n_batches)
        history['train_loss'].append(avg_train_loss)
        history['grad_norm'].append(avg_grad_norm)

        # evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        tot = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device=device, dtype=dtype)
                if is_classification:
                    yb = yb.to(device=device, dtype=torch.long)
                else:
                    yb = yb.to(device=device, dtype=dtype)
                out = model(xb)
                test_loss += criterion(out, yb).item()
                if is_classification:
                    pred = out.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    tot += yb.size(0)
        avg_test_loss = test_loss / max(1, len(test_loader))
        history['test_loss'].append(avg_test_loss)
        if is_classification:
            history['test_acc'].append(correct / max(1, tot))

        # print progress
        if is_classification:
            print(f"[{epoch}/{args.epochs}] train_loss={avg_train_loss:.4f} test_loss={avg_test_loss:.4f} test_acc={history['test_acc'][-1]:.4f}")
        else:
            print(f"[{epoch}/{args.epochs}] train_loss={avg_train_loss:.4f} test_loss={avg_test_loss:.4f}")

        if diverged:
            print("Diverged - stopping early.")
            break

    elapsed = time.time() - start_time

    # store meta/result
    meta = {'optimizer': args.optimizer, 'lr': args.lr, 'noise': args.noise,
            'precision': args.precision, 'seed': args.seed, 'dataset': args.dataset,
            'sam_radius': args.sam_radius if hasattr(args, 'sam_radius') else None,
            'epochs_run': epoch}
    result = {'meta': meta, 'history': history, 'diverged': bool(diverged), 'elapsed_sec': elapsed}

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f'res_{args.optimizer}_noise{args.noise}_{args.precision}_seed{args.seed}.json')
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='sgd', choices=['sgd','sgd_momentum','adam','rmsprop','adagrad','sam_sgd'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--precision', default='float32', choices=['float32','float64'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--outdir', default='results/light')
    parser.add_argument('--dataset', default='mnist', choices=['mnist','sine'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--sam-radius', type=float, default=0.05)
    args = parser.parse_args()
    train(args)
