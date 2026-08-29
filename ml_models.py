"""Machine Learning and Deep Learning models for the Atlantic Playlist app.

Trains two models in the background:
  1. Core ML — RandomForestRegressor (scikit-learn)
  2. Deep Learning — simple PyTorch MLP neural network

Both predict `popularity` from playlist features.  Results are cached via
``@st.cache_resource`` so training only runs once per session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is in requirements.txt
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "position", "duration_ms", "total_tracks", "is_explicit",
    "album_type_single", "album_type_album",
]


def prepare_features(df: pd.DataFrame):
    """Return (X, y) numpy arrays ready for modelling."""
    d = df.copy()
    d["is_explicit"] = d["is_explicit"].astype(int)
    d = pd.get_dummies(d, columns=["album_type"], prefix="album_type", dtype=int)
    # Ensure all expected dummy columns exist
    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = 0
    X = d[FEATURE_COLS].values.astype(np.float32)
    y = d["popularity"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Core ML — Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(X: np.ndarray, y: np.ndarray):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    r2 = float(r2_score(y_te, preds))
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "importances": importances,
        "y_test": y_te,
        "y_pred": preds,
    }


# ---------------------------------------------------------------------------
# Deep Learning — PyTorch MLP
# ---------------------------------------------------------------------------

class PopularityMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_neural_network(X: np.ndarray, y: np.ndarray, epochs: int = 60):
    if not _HAS_TORCH:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42)

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    model = PopularityMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te)).numpy()
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    r2 = float(r2_score(y_te, preds))
    return {
        "model": model,
        "scaler": scaler,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_te,
        "y_pred": preds,
    }
