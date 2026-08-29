# Atlantic Playlist AI — Final Deployment Project

## Run in VS Code

```powershell
cd Atlantic_Playlist_Final_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads pre-trained artifacts from `models/` and does NOT train models at startup.

## Model files

- `models/random_forest.pkl`
- `models/pytorch_mlp.pth`
- `models/scaler.pkl`
- `models/metadata.pkl`

## Deploy

Push the whole project to GitHub and deploy `app.py` using a Streamlit-compatible hosting service.

Do not remove the `models/` folder.

## Retraining

Only retrain if the dataset or model design changes. The final website should normally only load the saved artifacts.

## Training notebook

`Atlantic_Playlist_Analysis_Final.ipynb` contains the original analysis plus the final deployment-aligned model export section.
