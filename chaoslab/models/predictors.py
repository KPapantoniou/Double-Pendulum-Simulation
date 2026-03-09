import numpy as np
import torch as th
import torch.nn as nn

from chaoslab.analysis.metrics import mae, r2
from chaoslab.simulation import resolve_device

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def _train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1.0 - test_ratio))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def _fit_mlp_torch(X_train, y_train, X_test, device=None, epochs=250, lr=1e-3, batch_size=512):
    device = resolve_device(device)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    xtr = th.tensor(X_train_n, dtype=th.float32, device=device)
    ytr = th.tensor(y_train, dtype=th.float32, device=device).view(-1, 1)
    xte = th.tensor(X_test_n, dtype=th.float32, device=device)

    model = MLPRegressor(in_dim=X_train.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        perm = th.randperm(xtr.shape[0], device=device)
        for i in range(0, xtr.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with th.no_grad():
        y_pred = model(xte).squeeze(1).detach().cpu().numpy()
    return y_pred


def benchmark_models(X, y, seed=0, device=None):
    X_train, X_test, y_train, y_test = _train_test_split(X, y, seed=seed)

    results = {}

    if SKLEARN_AVAILABLE:
        rf = RandomForestRegressor(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        results["RandomForest"] = {
            "r2": r2(y_test, pred_rf),
            "mae": mae(y_test, pred_rf),
            "y_true": y_test,
            "y_pred": pred_rf,
        }

        gb = GradientBoostingRegressor(random_state=seed)
        gb.fit(X_train, y_train)
        pred_gb = gb.predict(X_test)
        results["GradientBoosting"] = {
            "r2": r2(y_test, pred_gb),
            "mae": mae(y_test, pred_gb),
            "y_true": y_test,
            "y_pred": pred_gb,
        }
    else:
        results["RandomForest"] = {"error": "scikit-learn not installed"}
        results["GradientBoosting"] = {"error": "scikit-learn not installed"}

    pred_mlp = _fit_mlp_torch(X_train, y_train, X_test, device=device)
    results["MLP"] = {
        "r2": r2(y_test, pred_mlp),
        "mae": mae(y_test, pred_mlp),
        "y_true": y_test,
        "y_pred": pred_mlp,
    }

    return results
