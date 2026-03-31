"""
train.py
--------
Training loops for BaseNet (deterministic / MC Dropout) and BayesNet.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _to_loader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_basenet(model, X_train, y_train, X_val, y_val,
                  epochs=100, lr=1e-3, batch_size=256,
                  patience=10, device="cpu", verbose=True):
    """
    Train BaseNet (or MC Dropout) with early stopping on validation BCE loss.
    Returns the best validation loss and the number of epochs run.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss — | val_loss {val_loss:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val


def train_bnn(model, X_train, y_train, X_val, y_val,
              epochs=100, lr=1e-3, batch_size=256,
              patience=10, device="cpu", verbose=True):
    """
    Train BayesNet with ELBO loss. Uses NLL on validation set for early stopping.
    beta = 1 / n_batches for proper scaling of KL term.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val,   dtype=torch.float32).to(device)

    n_data   = len(X_train)
    n_batch  = max(1, n_data // batch_size)
    beta     = 1.0 / n_batch

    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    criterion    = nn.BCELoss()

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = model.elbo_loss(xb, yb, n_data, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Val: NLL using posterior mean
        model.eval()
        with torch.no_grad():
            y_hat, _ = model.predict(X_val_t.to(device), S=10)
            val_loss = criterion(y_hat, y_val_t).item()

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | val_nll {val_loss:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val
