# Hotel Cancellation Prediction & Dynamic Overbooking Glide Path Engine

[![CI Pipeline](https://github.com/joshuaccarter/hotel-cancellation-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/joshuaccarter/hotel-cancellation-prediction/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](tests/)

---

## 1. Executive Summary

In the hospitality industry, room inventory is **strictly perishable**: an unsold room overnight generates zero revenue, and that revenue potential vanishes permanently at night audit. Conversely, walking an overbooked guest incurs relocation premiums, transport costs, and severe reputational damage.

This project delivers an end-to-end, production-grade Machine Learning and Revenue Management system that unifies two complementary decision layers:
1. **Individual Cancellation Likelihood Prediction**: A calibrated gradient-boosted decision engine that evaluates bookings strictly at the time of reservation creation (T0), predicting cancellation probabilities under an empirically derived **asymmetric loss matrix** (Cu unbooked room loss vs. Co overbooked walk loss).
2. **Dynamic Overbooking Glide Path Engine**: A portfolio-level engine that aggregates all on-the-books (OTB) reservations for a target arrival date. It enforces lead-time overbooking constraints and models physical arrivals using a **Bayesian Poisson-Binomial distribution** to bound capacity overflow probability $P(S > \text{Capacity}) \le \alpha$.

### Key Business Results
- **\$531,922 Loss Reduction**: Deploying the cost-optimal decision threshold ($\theta^* = 0.40$) instead of the naive default (0.50) cut operational losses by over 50%, capturing **86.41% of theoretical Perfect Oracle value**.
- **+\$55,500 Revenue Lift with 0.0% Walk Frequency**: In a 50-peak-date historical simulation, the dynamic Bayesian glide path achieved 100% capacity utilization while maintaining a **0.0% guest walk rate**, outperforming both naive fixed-capacity and static overbooking policies.
- **Zero Target & Operational Leakage**: Pruned post-booking indicators (including `required_car_parking_spaces`, which EDA proved had an artificial 0.00% cancellation rate due to front-desk check-in logging).

---

## 2. Description of the Problem

### 2.1 The Perishable Inventory & Cancellation Dilemma
Hotels face an aggregate cancellation rate of approximately **37%**. If a 200-room property strictly limits sales to 200 bookings, cancellations will leave an average of 74 rooms empty every night, causing massive revenue underperformance. However, naive overbooking creates severe tail risk: if fewer guests cancel than expected, arriving guests must be "walked" to competitor properties.

### 2.2 Asymmetric Economic Loss Formulation
Traditional ML classification minimizes symmetric error (e.g., accuracy at $\theta = 0.50$). In hospitality, false positives and false negatives carry drastically different costs:

$$\begin{aligned}
\text{Cu} &= \text{Cost of Underbooking (Unsold Room)} = \text{ADR(p50)} \\
\text{Co} &= \text{Cost of Overbooking (Walked Guest)} = (\text{ADR(p75)} - \text{ADR(p50)}) + \$25\text{ friction penalty}
\end{aligned}$$

Using empirical quantiles from the benchmark dataset:
- **Resort Hotel**: Cu = \$75.00, Co = \$75.00 ($\theta^* = 0.50$)
- **City Hotel**: Cu = \$100.00, Co = \$51.00 ($\theta^* = 0.338$)
- **Blended Property Baseline**: Cu = \$95.00, Co = \$56.00 ($\theta^* = 0.38 - 0.40$)

The cost-optimal Bayesian decision threshold is:

$$\theta^* = \frac{\text{Co}}{\text{Cu} + \text{Co}} \approx 0.40$$

Operating at $\theta^* = 0.40$ ensures that revenue managers accept additional reservations until the marginal expected walk penalty equals the marginal opportunity cost of an empty room.

### 2.3 The Dynamic Lead-Time Glide Path
Overbooking cannot be static. As an arrival date approaches:
1. Cancellation probabilities solidify.
2. The remaining booking window narrows, preventing replacement bookings if late cancellations occur.
3. Market capacity tightens, drastically increasing relocation costs.

The system enforces a **Lead-Time Glide Path** that steps down allowable inventory:

| Lead-Time Horizon (Days to Arrival) | Max Overbooking Factor | 200-Room Property Ceiling | Operational Rationale |
| :--- | :---: | :---: | :--- |
| **90+ days** | **1.25x** | 250 bookings | Maximum booking pace; broad cancellation runway. |
| **30 – 89 days** | **1.15x** | 230 bookings | Intermediate pace; cancellations steady. |
| **14 – 29 days** | **1.10x** | 220 bookings | Constraining inventory; rate optimization. |
| **4 – 13 days** | **1.05x** | 210 bookings | Pre-arrival lock-in; high certainty. |
| **0 – 3 days** | **1.00x** | 200 bookings | Zero overbooking allowance on arrival eve. |

---

## 3. Summary of Feature Selection & Governance

### 3.1 Strict Point-of-Booking (T0) Boundary
To guarantee real-world validity, only features irrevocably known at the precise moment of booking confirmation are permitted.

#### Purged Leakage Columns
1. **Terminal Status Indicators**:
   - `reservation_status` (Check-Out, Canceled, No-Show) — direct target proxy.
   - `reservation_status_date` — timestamp of cancellation event.
   - `assigned_room_type` — assigned at check-in; reflects room upgrade interventions.
2. **Operational Check-In Leakage**:
   - `required_car_parking_spaces`: Comprehensive EDA revealed that out of 7,416 historical records with parking > 0, **exactly 0 canceled (0.00% cancellation rate)**. Parking was logged upon car arrival at the front desk. Retaining this feature introduced an artificial "cheat rule" that collapsed in real production inference.
   - `total_of_special_requests`: Special requests (cribs, high floor) accumulate between booking and arrival, creating operational target drift.

### 3.2 Feature Engineering
- `total_stay_nights`: N(weekend) + N(week) (composite length of stay).
- `total_guests`: adults + children + babies (party volume).
- `includes_weekend`: Binary flag indicating presence of weekend nights.
- **Trigonometric Cyclical Encodings**:
  Calendar week numbers (1–53) possess continuous circular topology (week 53 connects to week 1). Linear encoding introduces an artificial distance of 52 units; we resolve this via sine and cosine transformations:
  - `week_sin` = $\sin\left(\frac{2\pi \cdot t}{53}\right)$
  - `week_cos` = $\cos\left(\frac{2\pi \cdot t}{53}\right)$

### 3.3 Production Feature Schema (19 Features)
- **Numerical (11)**: `lead_time`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes`, `days_in_waiting_list`, `adr`.
- **Categorical (8)**: `hotel`, `meal`, `country`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `reserved_room_type`, `deposit_type`, `customer_type`.

---

## 4. Summary of Model Evaluation & Benchmarks

### 4.1 Four-Model Benchmark (Notebook 06)
Models were evaluated on a stratified 20% holdout partition (23,878 reservations) across discrimination (ROC-AUC), calibration (Brier Score), and financial expected dollar loss:

| Model | ROC-AUC | Brier Score | Expected Loss / Reservation | Total Test Set Loss | % of Oracle Captured |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Policy ($\theta=0.50$)** | — | — | \$22.82 | \$544,833 | 0.00% (Baseline) |
| **Logistic Regression** | 0.8176 | 0.1601 | \$16.89 | \$403,299 | 57.65% |
| **K-Nearest Neighbors ($K=15$)** | 0.8654 | 0.1412 | \$15.22 | \$363,423 | 68.32% |
| **XGBoost** | 0.9085 | 0.1162 | \$13.08 | \$312,324 | 85.24% |
| **LightGBM (Selected)** | **0.9117** | **0.1140** | **\$12.91** | **\$308,266** | **86.41%** |
| **Perfect Oracle (Upper Bound)** | 1.0000 | 0.0000 | \$0.00 | \$0 | 100.00% |

> **KNN Diminishing Returns Finding**: K was evaluated iteratively from $K=3$ to $K=75$. Performance peaked at $K^* = 15$ and degraded steadily for $K > 25$ as neighborhood sparsity diluted local cancellation densities.

### 4.2 Dynamic Glide Path Simulation (Notebook 07)
Across 50 historical peak demand dates (Capacity = 200 rooms, Total Bookings = 240):

| Operating Policy | Mean Shows | Empty Rooms | Walked Guests | Net Realized Revenue | Walk Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Fixed Capacity (1.00x)** | 148.9 | 555 | 0 | \$394,500 | 0.0% |
| **Static Overbooking (1.15x)** | 196.2 | 0 | 32 | \$438,400 | 64.0% |
| **Dynamic Bayesian Glide Path** | **198.8** | **12** | **0** | **\$450,000** | **0.0%** |

The Dynamic Bayesian Glide Path achieved **+\$55,500 in net revenue lift** over naive capacity while ensuring **zero guests were walked** by verifying $P(S > \text{Capacity}) \le 0.05$.

---

## 5. Tech Stack & Architecture Review

### 5.1 Technology Stack
- **Language & Runtime**: Python 3.12 (CPython), Docker (Debian 12 slim)
- **Package & Virtualenv Management**: Astral [`uv`](https://github.com/astral-sh/uv) (deterministic locking via `uv.lock`)
- **API Framework**: FastAPI 0.162, Starlette, Uvicorn (ASGI)
- **Data Validation & Schemas**: Pydantic V2 (`BaseModel`, `Field` bounds validation)
- **Machine Learning**: LightGBM 4.7, XGBoost 3.4, Scikit-learn 1.9
- **Scientific Computing**: NumPy 2.5, Pandas 3.1, SciPy 1.18
- **Testing & Quality**: Pytest 9.1, Ruff 0.17 (Linter & Formatter)

### 5.2 System Architecture

```text
                                 [ Client / PMS Engine ]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │               FastAPI API Layer               │
                    │   - Pydantic V2 Input Validation              │
                    │   - Non-root Execution (UID 1000)             │
                    │   - Pre-warmed Lifespan Model In-Memory       │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
              POST /predict/reservation           POST /predict/glidepath
                            │                               │
                            ▼                               ▼
               ┌────────────────────────┐      ┌─────────────────────────┐
               │ CancellationPredictor  │      │    GlidePathEngine      │
               │  - Cyclical Encoding   │      │  - Lead-Time Horizon    │
               │  - LightGBM Pipeline   │◄────┤  - Poisson-Binomial S   │
               │  - Risk Categorization │      │  - Tail Risk P(S > C)   │
               └────────────────────────┘      └────────────┬────────────┘
                            │                               │
                            ▼                               ▼
                 [ Calibrated P(Cancel) ]         [ Operational Directive ]
                 [ Action: Low/Mod/High ]         [ ACCEPT / RESTRICT /   ]
                                                  [ CLOSE_INVENTORY       ]
```

### 5.3 Repository Tree

```text
hotel-cancellation-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI pipeline (lint, test, docker build)
├── config/
│   └── config.yaml                     # Central configuration (bounds, features, loss values, glide path)
├── data/
│   ├── raw/hotel_bookings.csv          # Raw benchmark dataset (119,390 records)
│   └── processed/                      # Cleaned parquet datasets
├── models/
│   └── artifacts/
│       └── lightgbm_cancellation_pipeline.joblib   # Serialized preprocessor + LightGBM pipeline
├── notebooks/
│   ├── 01_dataset_overview_and_cardinality.ipynb
│   ├── 02_categorical_distributions_and_leakage.ipynb
│   ├── 03_numerical_distributions_and_leakage.ipynb
│   ├── 04_pairwise_distributions_and_collinearity.ipynb
│   ├── 05_feature_and_model_exploration.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_glidepath_and_risk_aggregation.ipynb
├── src/hotel_cancellation_prediction/
│   ├── api/
│   │   ├── app.py                      # FastAPI factory & lifespan pre-warming
│   │   ├── routes.py                   # REST endpoints (/health, /model/info, /predict/*)
│   │   └── schemas.py                  # Pydantic V2 request/response contracts
│   ├── data/
│   │   └── loader.py                   # Ingestion, cleaning, and leakage pruning
│   ├── features/
│   │   └── pipeline.py                 # Feature engineering & scikit-learn transformers
│   ├── models/
│   │   ├── cancellation_model.py       # LightGBM predictor class
│   │   └── glidepath_model.py          # Poisson-Binomial tail risk aggregation engine
│   └── utils/
│       └── logger.py                   # Structured console logging
├── tests/
│   ├── test_api.py                     # Integration tests for FastAPI endpoints
│   ├── test_cancellation_model.py      # Predictor unit tests
│   ├── test_config.py                  # Config parser & economics tests
│   ├── test_data_loader.py             # Data cleaning tests
│   ├── test_features.py                # Feature pipeline tests
│   └── test_glidepath.py               # Glide path & tail risk math tests
├── Dockerfile                          # Multi-stage production container definition
├── docker-compose.yml                  # Service orchestration with healthcheck
├── pyproject.toml                      # Project metadata & tool configurations
└── uv.lock                             # Fully pinned dependency lockfile
```

---

## 6. Quick Start Guide

### 6.1 Prerequisites
- Python 3.12+
- [Astral `uv`](https://github.com/astral-sh/uv) (recommended) or Docker Desktop

### 6.2 Local Execution via `uv`

```bash
# 1. Clone repository
git clone https://github.com/joshuaccarter/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

# 2. Sync virtual environment and dependencies
uv sync

# 3. Launch FastAPI development server
uv run uvicorn hotel_cancellation_prediction.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

### 6.3 Production Container Execution via Docker

```bash
# Build and run containerized API in the background
docker compose up -d --build

# View container logs
docker compose logs -f

# Verify automated health status (reports 'healthy')
docker compose ps

# Tear down container
docker compose down
```

---

### 6.4 Running Automated Tests & Quality Checks

```bash
# Run full pytest suite (17 passed)
uv run pytest -v

# Run Ruff lint check
uv run ruff check .

# Verify code formatting
uv run ruff format --check .
```

---

### 6.5 Example API Requests

#### A. Predict Individual Reservation
```bash
curl -X POST http://localhost:8000/predict/reservation \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "Resort Hotel",
    "lead_time": 45,
    "arrival_date_year": 2026,
    "arrival_date_month": "October",
    "arrival_date_week_number": 44,
    "arrival_date_day_of_month": 27,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 110.50
  }'
```
**Response:**
```json
{
  "cancellation_probability": 0.15,
  "prediction": 0,
  "risk_level": "LOW",
  "threshold_used": 0.4
}
```

#### B. Evaluate Date-Level Overbooking Glide Path
```bash
curl -X POST http://localhost:8000/predict/glidepath \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "City Hotel",
    "days_to_arrival": 30,
    "hotel_capacity": 200,
    "target_risk_alpha": 0.05,
    "current_bookings": [
      {
        "hotel": "City Hotel",
        "lead_time": 30,
        "adr": 115.0,
        "market_segment": "Online TA",
        "deposit_type": "No Deposit"
      }
    ]
  }'
```
**Response:**
```json
{
  "hotel": "City Hotel",
  "days_to_arrival": 30,
  "hotel_capacity": 200,
  "total_current_bookings": 1,
  "expected_cancellations": 0.1,
  "expected_show_ups": 0.9,
  "max_allowed_overbooking_factor": 1.15,
  "max_allowed_bookings": 230,
  "remaining_booking_allowance": 229,
  "overflow_probability": 0.0,
  "is_within_glide_path": true,
  "glide_path_status": "ACCEPTING_RESERVATIONS",
  "recommended_action": "ACCEPT_RESERVATIONS"
}
```

---

## 7. Continuous Integration (CI) Workflow

The repository includes a production-grade GitHub Actions CI pipeline defined in [`.github/workflows/ci.yml`](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/.github/workflows/ci.yml). Every push or pull request to the `main` branch triggers automated quality gates:

```text
 ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────────┐
 │  1. Lint & Format    │ ───► │  2. Run Test Suite   │ ───► │  3. Docker Build & Smoke │
 │  - Ruff Linter Check │      │  - Pytest (17 tests) │      │  - Multi-stage image     │
 │  - Ruff Format Check │      │  - Unit & Integration│      │  - /health & /model/info │
 └──────────────────────┘      └──────────────────────┘      └──────────────────────────┘
```

1. **`lint-and-format`**:
   - Sets up Python 3.12 and installs Astral `uv` with automatic dependency caching (`astral-sh/setup-uv@v5`).
   - Runs `uv run ruff check .` to verify zero linting errors.
   - Runs `uv run ruff format --check .` to ensure consistent code styling.
2. **`test`**:
   - Executes all 17 automated tests (`uv run pytest -v`) covering API data contracts, LightGBM inference, economic loss configurations, data loading bounds, feature engineering pipelines, and Poisson-Binomial tail risk math.
3. **`docker-build`**:
   - Validates the multi-stage [`Dockerfile`](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/Dockerfile) using Buildx.
   - Spins up the container and runs automated smoke checks against `http://localhost:8000/health` and `http://localhost:8000/model/info` to guarantee deployability.

