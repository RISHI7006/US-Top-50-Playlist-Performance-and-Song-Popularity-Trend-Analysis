"""Load pre-trained models for the Atlantic Playlist Streamlit app."""
from pathlib import Path
import pickle
import torch
from ml_models import PopularityMLP

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_models():
    required = [
        "random_forest.pkl",
        "pytorch_mlp.pth",
        "scaler.pkl",
        "metadata.pkl",
    ]
    missing = [name for name in required if not (MODELS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained model files: " + ", ".join(missing)
        )

    with open(MODELS_DIR / "random_forest.pkl", "rb") as f:
        rf = pickle.load(f)

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(MODELS_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    checkpoint = torch.load(
        MODELS_DIR / "pytorch_mlp.pth",
        map_location="cpu",
    )
    model = PopularityMLP(checkpoint["input_dim"])
    # Current artifact uses BetterPopularityMLP_v2 architecture, so recreate
    # that exact architecture when needed.
    if checkpoint.get("architecture") == "BetterPopularityMLP_v2":
        import torch.nn as nn

        class BetterPopularityMLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 96),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(96, 48),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(48, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = BetterPopularityMLP(checkpoint["input_dim"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return rf, model, scaler, metadata
