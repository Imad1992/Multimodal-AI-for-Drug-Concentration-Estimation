# Multimodal Sensor AI — Fusion and Evolutionary Deconvolution

An end-to-end **research software prototype** for synthetic multimodal sensing. It combines SERS-like spectra, electrochemical-like traces, a PyTorch fusion network and differential-evolution deconvolution. Includes a Streamlit dashboard, FastAPI inference endpoint, tests and GitHub Actions CI.

> **Research use only.** Data and sensor models are entirely simulated. Outputs are dimensionless, not clinical drug levels. Do not use this system for patient care, diagnosis, dosing or medical decisions.

## Features

- Reproducible synthetic three-analyte mixtures and paired sensor modalities
- Sensor data shape and finite-value validation
- Two-branch PyTorch neural network with late feature fusion
- Train-only standardisation and held-out model evaluation (MAE, RMSE, R²)
- Differential-evolution-based optimisation against a synthetic forward model
- Interactive plots, side-by-side predictions, and training curves
- Local FastAPI `/health` and `/predict` endpoints
- Automated tests and GitHub Actions continuous integration

## Quick start — VS Code (Windows PowerShell)

Install Python 3.11 or 3.12, open this folder in VS Code, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
python -m scripts.bootstrap
streamlit run app.py
```

If `py -3.11` is unavailable, use `python -m venv .venv`. If PowerShell activation is restricted, use `.venv\Scripts\python.exe` instead of `python` for subsequent commands. On macOS/Linux use `source .venv/bin/activate`.

Open http://localhost:8501. Initial model training can take a few minutes depending on your computer.

## API

```bash
uvicorn api:app --reload
```

Visit http://127.0.0.1:8000/docs for interactive API documentation. Train first to create model artifacts. API has no authentication; keep it local.

## Testing and evaluation

```bash
python -m pytest -q
python -m scripts.evaluate
```

Training writes `artifacts/metrics.json`, `artifacts/history.json`, `artifacts/fusion_model.pt`, and `artifacts/scalers.npz`. These generated files are excluded from Git. Dataset is generated under `data/` and excluded as well.

## Project layout

```text
src/sensor_ai/      simulation, fusion model, training, optimisation
scripts/            bootstrap and evaluation
app.py              Streamlit dashboard
api.py              FastAPI service
tests/              automated tests
docs/architecture.md architecture and limitations
.github/workflows/  continuous integration
```

## Model and evaluation design

The model predicts three normalised concentrations using separate encoders for 128 SERS-like and 64 electrochemical-like measurements. A seeded random 80% development / 20% test split is used; development is further split into training and validation. Standardisation is fitted on training data only. The best validation model is evaluated on the held-out test subset. This demonstrates software mechanics rather than real-world generalisation.

Differential evolution separately estimates three concentrations by fitting the known synthetic signal basis; it is not a validated physical deconvolution method for real instruments.

## GitHub

```bash
git init
git add .
git commit -m "Initial multimodal sensor AI research prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/multimodal-sensor-ai.git
git push -u origin main
```

Replace `YOUR_USERNAME`. Create an empty repository first. Never commit real patient data or secrets.

## Future development

- Instrument-specific preprocessing and calibrated sensor models
- Cross-device, batch and external validation with real approved datasets
- Model uncertainty, out-of-distribution detection and drift monitoring
- Robust data governance, access controls and audit trails
- Streaming hardware interfaces and performance benchmarking

See [architecture documentation](docs/architecture.md) for limitations and deployment considerations.

## License

MIT — see [LICENSE](LICENSE).
