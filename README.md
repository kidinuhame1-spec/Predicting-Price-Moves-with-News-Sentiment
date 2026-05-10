# Predicting Price Moves with News Sentiment

This repository contains an instructional scaffold for the Week 1 challenge "Predicting Price Moves with News Sentiment".

Contents added:
- `requirements.txt` — Python dependencies for the notebooks and tests
- `notebooks/01-EDA.ipynb` — starter Jupyter notebook for Task 1 (EDA)
- `.github/workflows/unittests.yml` — basic CI to run unit tests

How to get started locally

1. Create a virtual environment and activate it.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Launch Jupyter:

```bash
jupyter notebook
```

3. Open `notebooks/01-EDA.ipynb` and follow the instructions to run the exploratory analysis.

Notes

- `TA-Lib` may require OS-level dependencies on some platforms; the `ta` package is included as a pure-Python alternative.
- The repository is a scaffolded starter for the challenge; replace the placeholder dataset in `data/raw/` with the FNSPID data CSV.
# Predicting-Price-Moves-with-News-Sentiment