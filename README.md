<div align="center">

# OceanEmbed

**Physics-informed deep learning for 3D subsurface ocean reconstruction from satellite observations**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)
![Problem](https://img.shields.io/badge/Problem-SIH26066-F28C28?style=flat-square)

[Live Demo](https://oceanembed-phi.vercel.app/) &nbsp;|&nbsp; [Demo Video](https://youtu.be/GUdMyuLT980) &nbsp;|&nbsp; [Technical Report](./Technical%20Report.pdf) &nbsp;|&nbsp; [Presentation](./PPT.pdf)

</div>

<!--
  Add a dashboard screenshot here once available, for example:
  ![OceanEmbed dashboard](docs/dashboard.png)
-->

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Results](#4-results)
5. [Current Status and Limitations](#5-current-status-and-limitations)
6. [Repository Structure](#6-repository-structure)
7. [Getting Started](#7-getting-started)
8. [Configuration](#8-configuration)
9. [API Reference](#9-api-reference)
10. [Training and Evaluation Workflow](#10-training-and-evaluation-workflow)
11. [Testing and Validation](#11-testing-and-validation)
12. [Deployment](#12-deployment)
13. [Data Sources](#13-data-sources)
14. [Roadmap](#14-roadmap)
15. [Contributing](#15-contributing)
16. [License](#16-license)
17. [Acknowledgements](#17-acknowledgements)

---

## 1. Overview

OceanEmbed reconstructs the three-dimensional temperature and salinity structure of the North Indian Ocean using only surface satellite observations. From a 7-day window of surface fields (sea surface temperature, salinity, sea level anomaly, geostrophic currents and winds), the model produces daily subsurface fields at **0.25° × 0.25°** resolution across **15 standard depths (0–1000 m)**. The domain spans the Arabian Sea and the Bay of Bengal (5°N–30°N, 45°E–105°E).

The reconstruction feeds a set of TEOS-10 diagnostics (TCHP, MLD, Z₂₀, BLT, CIP), calibrated uncertainty estimates, and an operational dashboard that includes an AI-assisted maritime advisory.

### Motivation

Forecasters can observe tropical cyclones from space but cannot directly observe the warm-water reservoir beneath the surface that fuels rapid intensification. ARGO profiling floats measure the ocean interior but are sparse in the North Indian Ocean, particularly during fast-evolving events. OceanEmbed learns the relationship between surface signatures and subsurface structure to help close this observation gap.

### Problem Statement

| Attribute | Detail |
|---|---|
| **Identifier** | SIH26066 |
| **Organisation** | Ministry of Earth Sciences (MoES) / INCOIS |
| **Theme / Category** | Disaster Management / Software |
| **Objective** | Satellite-embedding-based deep learning framework to reconstruct depth-wise subsurface temperature from daily surface observations at 0.25° resolution over the North Indian Ocean |

---

## 2. Key Features

**Physics-informed modelling**

- Regularized Coriolis parameter, `f̃ = sign(f) · max(|f|, f₀)`, to avoid the equatorial singularity
- Ekman pumping, `wₑ = ∇×τ / (ρ₀ f̃)`, computed from wind stress curl and supplied as an input channel
- Climatological anomaly decomposition: the network predicts residuals relative to a daily climatology
- Stratification penalty in the loss to discourage unphysical temperature inversions
- TEOS-10 thermodynamics (via `gsw`) for density, mixed-layer depth and heat content

**Model architecture**

- Dual-stream encoder capturing local eddy-scale and basin-scale variability
- 512-dimensional latent ocean embedding
- Depth-attention decoder with learnable depth tokens and multi-head cross-attention
- Sub-basin heads for the Arabian Sea and Bay of Bengal (split at 80°E)
- Joint temperature and salinity prediction
- Heteroscedastic uncertainty (predicted log-σ) with Monte Carlo Dropout at evaluation time

**Operational console**

- 2D/3D geospatial viewer with click-to-profile at any coordinate
- Vertical diagnostics: thermal sounding, salinity and density, thermocline gradient
- Selectable variable and depth layers, including uncertainty layers
- Cyclone Amphan (May 2020) case study
- AI advisory generation in the style of an INCOIS bulletin. The language model only narrates values computed by the pipeline and does not generate physical numbers.

**Derived indices**

| Index | Description |
|---|---|
| TCHP | Tropical Cyclone Heat Potential (kJ/cm²) |
| MLD | Mixed Layer Depth (m) |
| Z₂₀ | Depth of the 20 °C isotherm (m) |
| BLT | Barrier Layer Thickness (m) |
| CIP | Cyclone Intensification Potential |

---

## 3. System Architecture

```text
Satellite inputs (0.25°)    SST · SSS · SLA · ugos/vgos · u10/v10
          │
          ▼
Physical pre-processing     Coriolis f̃ · Ekman pumping wₑ · bathymetry
                            day-of-year encoding · climatology anomaly
          │                 (7 days × 12 channels × 100 × 240 grid)
          ▼
Dual-stream encoder         Stream A: local eddies  |  Stream B: basin waves
          └────────────────► 512-d latent ocean embedding
          ▼
Depth-attention decoder     learnable depth tokens · cross-attention
                            sub-basin heads (Arabian Sea / Bay of Bengal)
          ▼
Outputs (15 depths)         Δθ · ΔSₚ · log-σ   (added to climatology)
          ▼
TEOS-10 diagnostics         TCHP · MLD · Z₂₀ · BLT · CIP
          ▼
Serving                     FastAPI → Express gateway (validation, cache) → Next.js console
```

| Item | Specification |
|---|---|
| **Model input** | `(B, 7, 12, 100, 240)`: 7 days × 12 channels on the 100 × 240 grid |
| **Model output** | Residual θ, residual Sₚ and log-σ, each `(B, 15, 100, 240)` |
| **Input channels** | `sst`, `sss`, `sla`, `ugos`, `vgos`, `u10`, `v10`, `w_e`, `f_tilde`, `bathymetry`, `doy_sin`, `doy_cos` |
| **Depth levels (m)** | 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 |
| **Grid** | 0.25°, 100 × 240 cells, cell-centre registration |

### Technology Stack

| Layer | Technologies |
|---|---|
| Machine learning | PyTorch, PyTorch Lightning, ONNX / ONNX Runtime |
| Scientific computing | `gsw` (TEOS-10), SciPy (PCHIP), xarray, Dask, Zarr, xESMF |
| Data access | `copernicusmarine`, `cdsapi`, NetCDF4 |
| Inference API | FastAPI, Uvicorn, Pydantic v2, rasterio |
| Gateway | Express.js, TypeScript, Zod, ioredis (optional), Axios |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS, deck.gl, MapLibre GL, Plotly.js |
| Advisory | Google Gemini (`google-genai`) |
| Deployment | Docker, Render, Vercel, GitHub Actions |

---

## 4. Results

Headline figures as reported in the [Technical Report](./Technical%20Report.pdf):

| Metric | Value |
|---|---|
| Thermocline RMSE (50–200 m) | 0.88 °C |
| ARGO validation RMSE | 0.107 °C (N = 1,420 profiles) |
| Climatology skill score | > 0.78 |
| Training MSE reduction | 87.28 % |
| Uncertainty coverage | 71.4 % at ±1σ |
| Neural inference time | 142 ms (single NVIDIA T4) |
| Trainable parameters | 2.64 M |

### Case Study: Cyclone Amphan (May 2020)

| Phase | TCHP (kJ/cm²) | MLD (m) | Z₂₀ (m) | SST (°C) |
|---|---|---|---|---|
| Pre-storm (16 May) | 115.2 | 48.0 | 112.0 | 31.2 |
| Eye transit (18 May) | 62.8 | 52.3 | 94.0 | 29.1 |
| Post-storm (20 May) | 35.6 | 75.0 | 78.0 | 27.8 |

Evaluation figures are available in `track3_validation_engine/output/`:
`real_model_evaluation.png`, `argo_sounding_matchup.png` and `vertical_error_plot.png`.

---

## 5. Current Status and Limitations

- **Bundled weights are a proof of concept.** `track4_backend/fastapi_engine/weights/oceanembed_real.pth` was trained for 15 epochs on a one-month slice (May 2020) over the Bay of Bengal (13–18°N, 84–89°E) using the sample data in `data/`.
- **Full-basin training is supported but not bundled.** The data pipeline and model target the full North Indian Ocean for 2012–2021 (train 2012–2018, validation 2019, test 2020–2021). Reproducing full-basin results requires running the Track 1 pipeline over that period and training with Track 2.
- **Amphan responses are pre-computed.** Requests near the storm track and dates are served from `static/amphan_case_study.json`.
- **Synthetic fallback.** If no trained weights or ONNX model are available and `ALLOW_SYNTHETIC=true`, the inference service returns synthetic fields so the API and console remain usable for demonstration.
- **Docker image scope.** The FastAPI Dockerfile does not currently copy `track2_model_engine` or install PyTorch, so containerised deployments may not load the `.pth` weights. Local runs from the repository root do.

---

## 6. Repository Structure

```text
OceanEmbed/
├── track1_data_engine/          Data ingestion and preprocessing
│   ├── ingestion/               CMEMS, OISST and wind downloaders
│   ├── preprocessing/           Regridder, bathymetry mask, physics features, climatology
│   ├── storage/                 Zarr converter and dataset validator
│   ├── configs/                 bounding_box.yaml, data_sources.yaml
│   └── run_pipeline.py          CLI: download → regrid → physics → climatology → Zarr
│
├── track2_model_engine/         Model, losses and training
│   ├── models/                  ocean_embed, dual_stream_encoder, depth_decoder,
│   │                            heads, layers, baseline_mlp, baseline_unet
│   ├── losses/                  physics_loss, area_weighted_loss
│   ├── dataset/                 Zarr and synthetic datasets, transforms
│   ├── configs/model.yaml       Hyperparameters and data split
│   ├── train.py                 PyTorch Lightning training
│   ├── evaluate.py              Monte Carlo Dropout evaluation
│   └── export_onnx.py           ONNX and TorchScript export
│
├── track3_validation_engine/    Validation and scientific diagnostics
│   ├── validation/              ARGO and RAMA matchers
│   ├── diagnostics/             TEOS-10, PCHIP profiler, TCHP/MLD/Z20/BLT indices
│   ├── benchmarks/              Skill scores and report generation
│   ├── plots/                   Evaluation plotting scripts
│   ├── output/                  Generated figures
│   └── test_diagnostics.py      Numerical tests
│
├── track4_backend/              Serving layer
│   ├── fastapi_engine/          Inference API, advisory service, trained weights
│   ├── express_gateway/         TypeScript gateway (Zod validation, Redis cache)
│   └── docker-compose.yml       Redis, inference service and gateway
│
├── track5_frontend/             Next.js 14 operational console
│   └── src/                     App routes, map and chart components, hooks, API client
│
├── data/                        Sample raw and processed data (May 2020 slice)
├── download_real_sample.py      Download the proof-of-concept CMEMS slice
├── prepare_training_tensors.py  Build training tensors from the raw slice
├── run_local_training.py        Train on the sample slice
├── render.yaml                  Render deployment blueprint
├── requirements.txt             Python dependencies (data and model)
├── PPT.pdf                      Project presentation
└── Technical Report.pdf         Full technical report
```

---

## 7. Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 is used in the Docker image |
| Node.js 20+ | Gateway and frontend |
| Docker and Docker Compose | Optional |
| Copernicus Marine account | Only needed to download new data |
| Gemini API key | Only needed for AI-generated advisories |

### Clone the repository

```bash
git clone https://github.com/Jashwanth006/OceanEmbed.git
cd OceanEmbed
```

### Inference engine (FastAPI)

```bash
python -m venv venv
source venv/bin/activate                  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r track4_backend/fastapi_engine/requirements.txt

cp track4_backend/fastapi_engine/.env.example track4_backend/fastapi_engine/.env
uvicorn fastapi_engine.app:app --app-dir track4_backend --port 8000 --reload
```

Run this from the repository root so that `track2_model_engine` can be imported. Interactive documentation is available at `http://localhost:8000/docs`.

### API gateway (Express)

```bash
cd track4_backend/express_gateway
cp .env.example .env
npm install
npm run dev                               # http://localhost:8080
```

Redis is optional. If `REDIS_URL` is not set, the gateway skips caching.

### Operational console (Next.js)

```bash
cd track5_frontend
cp .env.example .env.local
npm install
npm run dev                               # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` in `.env.local` to the gateway, for example `http://127.0.0.1:8080/api/v1`.

### Docker Compose (Redis, inference, gateway)

```bash
cd track4_backend
docker compose up --build
```

| Service | Port |
|---|---|
| Inference (FastAPI) | 8000 |
| Gateway (Express) | 8080 |
| Redis | 6379 |

Start the frontend separately as described above.

---

## 8. Configuration

### `track4_backend/fastapi_engine/.env`

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `oceanembed` | Root of the Zarr data store |
| `ONNX_PATH` | `oceanembed/export/oceanembed.onnx` | Exported ONNX model |
| `TORCHSCRIPT_PATH` | `oceanembed/export/oceanembed.ts` | Exported TorchScript model |
| `ALLOW_SYNTHETIC` | `true` | Permit synthetic fallback when no model or data is available |
| `ONNX_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | ONNX Runtime execution providers |
| `GEMINI_API_KEY` | none | Enables AI-generated advisories |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Server bind address |

### `track4_backend/express_gateway/.env`

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_URL` | `http://127.0.0.1:8000` | Base URL of the inference service |
| `REDIS_URL` | `redis://127.0.0.1:6379` | Optional cache |
| `CACHE_TTL_SECONDS` | `86400` | Cache lifetime in seconds |
| `PORT` | `8080` | Gateway port |

### `track5_frontend/.env.local`

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Gateway base URL, including `/api/v1` |

Do not commit `.env` files or API keys. `.env` is listed in `.gitignore`.

---

## 9. API Reference

### Gateway: base path `/api/v1`

| Method | Endpoint | Parameters | Description |
|---|---|---|---|
| GET | `/health` | none | Gateway and inference health |
| GET | `/ocean/profile` | `lat`, `lon`, `date` | Vertical θ/S profile with uncertainty and indices |
| GET | `/ocean/layer` | `date`, `depth`, `variable` | Horizontal layer at a standard depth |
| GET | `/ocean/indices` | `date`, `index` | Basin map of a derived index |
| POST | `/ocean/advisory` | `lat`, `lon`, `date`, `sst`, `tchp`, `mld`, `z20`, `inversion_flag`, `uncertainty` | Generate an advisory bulletin |

**Parameter constraints**

| Parameter | Allowed values |
|---|---|
| `lat` | 5 to 30 |
| `lon` | 45 to 105 |
| `date` | `YYYY-MM-DD` |
| `depth` | One of the 15 standard levels |
| `variable` | `temperature`, `salinity`, `uncertainty_theta`, `uncertainty_sp` |
| `index` | `tchp`, `mld`, `z20`, `blt`, `cip` |

`GET` responses are cached in Redis when available; cache hits return an `X-Cache: HIT` header.

**Example**

```bash
curl "http://localhost:8080/api/v1/ocean/profile?lat=15.5&lon=86.3&date=2020-05-18"
```

### Inference service (direct access)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend status, weights and domain |
| POST | `/predict/profile` | Profile at a coordinate and date |
| POST | `/predict/grid` | Gridded layer (`json`, `geotiff` or `binary`) |
| POST | `/predict/indices` | Derived index map |
| POST | `/advisory/generate` | Advisory bulletin |
| GET | `/docs` | Swagger UI |

---

## 10. Training and Evaluation Workflow

### Proof of concept on the bundled sample

```bash
pip install -r requirements.txt

python download_real_sample.py          # Optional: requires Copernicus Marine login
python prepare_training_tensors.py      # Writes data/processed/training_tensors.pt
python run_local_training.py            # Writes track4_backend/fastapi_engine/weights/oceanembed_real.pth
```

The sample files are already included in `data/`, so the download step can be skipped.

### Full pipeline

```bash
# Track 1: build the Zarr dataset
python -m track1_data_engine.run_pipeline --stage smoke
python -m track1_data_engine.run_pipeline --stage all --start 2012-01-01 --end 2012-01-31
python -m track1_data_engine.run_pipeline --stage download --products glorys,oisst,winds

# Track 2: train, evaluate and export
python -m track2_model_engine.train --model oceanembed --data-root oceanembed --max-epochs 80
python -m track2_model_engine.train --synthetic --fast-dev-run
python -m track2_model_engine.evaluate --ckpt <path/to/checkpoint> --split test --mc 50
python -m track2_model_engine.export_onnx --ckpt <path/to/checkpoint> --out-dir oceanembed/export
```

`--model` also accepts `mlp` and `unet` to train baseline models for comparison. `--synthetic --fast-dev-run` runs a quick smoke test without any data.

### Training configuration

Defined in `track2_model_engine/configs/model.yaml`:

| Parameter | Value |
|---|---|
| Latent dimension | 512 |
| Attention heads | 8 |
| Dropout | 0.1 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Precision | 16-bit mixed |
| Gradient clipping | 1.0 |
| Max epochs | 80 |

**Loss:** Gaussian negative log-likelihood on temperature and salinity (weights 1.0 and 0.5), a stratification-inversion penalty (λ = 0.1), and an area-weighted term.

---

## 11. Testing and Validation

```bash
python -m track3_validation_engine.test_diagnostics
# or
pytest track3_validation_engine/test_diagnostics.py
```

The validation engine provides ARGO and RAMA co-location, climatology skill score, depth-wise skill, inversion Brier score, and automated report and plot generation.

---

## 12. Deployment

| Component | Platform | Notes |
|---|---|---|
| Inference service and gateway | Render | `render.yaml` defines `oceanembed-fastapi` and `oceanembed-gateway`. Set `GEMINI_API_KEY` in the Render dashboard. |
| Frontend | Vercel | Deploy `track5_frontend/` and set `NEXT_PUBLIC_API_URL` to the gateway URL. |
| Keep-alive | GitHub Actions | `.github/workflows/keep_alive.yml` pings both services every 10 minutes to avoid free-tier sleep. |

---

## 13. Data Sources

| Variable | Provider | Product |
|---|---|---|
| Sea surface temperature | NOAA | OISST v2.1 |
| Sea surface salinity | Copernicus Marine | Sea surface salinity product |
| Sea level anomaly, geostrophic currents | Copernicus Marine | SEALEVEL |
| Surface winds | ECMWF / EUMETSAT | ERA5 / ASCAT |
| Training target (θ, S) | Copernicus Marine | GLORYS12V1 reanalysis (1/12°, 50 levels) |
| Bathymetry | NOAA | ETOPO1 |
| Validation | Coriolis / INCOIS, NOAA PMEL | ARGO floats, RAMA moorings |

Dataset identifiers and endpoints are configured in `track1_data_engine/configs/data_sources.yaml`.

---

## 14. Roadmap

- Train on the full 2012–2021 North Indian Ocean dataset
- Extend ARGO and RAMA validation across seasons and sub-basins
- Add near-real-time ingestion of satellite products
- Add further tropical cyclone case studies
- Add authentication and rate limiting to the public gateway
- Include model and dependencies in the production Docker image

---

## 15. Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Commit your changes: `git commit -m "Describe your change"`
3. Push the branch and open a pull request

Please follow PEP 8 for Python, use TypeScript strict mode for frontend and gateway code, and add or update tests where relevant.

---

## 16. License

Released under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 17. Acknowledgements

- Ministry of Earth Sciences (MoES) and INCOIS for the problem statement
- Copernicus Marine Service for GLORYS12V1 reanalysis and satellite products
- NOAA, NASA, ESA, ECMWF and EUMETSAT for observational products
- ARGO Programme and RAMA array for in-situ validation data
- TEOS-10 and the Gibbs SeaWater (`gsw`) toolbox
- The PyTorch, FastAPI, Next.js and scientific Python open-source communities

<div align="center">

**OceanEmbed** · From satellite surface to ocean depth

</div>
