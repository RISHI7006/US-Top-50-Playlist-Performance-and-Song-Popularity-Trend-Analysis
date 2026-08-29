"""Optional one-time retraining script.

The final deployed app does NOT run this script. It only loads files from models/.
Run this only if you change the dataset or want to retrain the models.
"""
from pathlib import Path
import pickle
import pandas as pd
import torch
from ml_models import prepare_features, train_random_forest
from model_loader import load_models

BASE = Path(__file__).resolve().parent
DATA = BASE / "Atlantic_United_States.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X, y = prepare_features(df)

rf = train_random_forest(X, y)
with open(MODELS / "random_forest.pkl", "wb") as f:
    pickle.dump(rf["model"], f)

# For reproducible DL retraining, use the notebook/training pipeline used
# to generate the current pytorch_mlp.pth and scaler.pkl artifacts.
print("Random Forest retrained.")
print("Existing PyTorch artifact was not overwritten.")
print("For the current final project, use the included models/*.pth and scaler.pkl.")
