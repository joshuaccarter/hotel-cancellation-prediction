# Exploratory Data Analysis & Modeling Notebooks

This directory contains the analytical and modeling notebooks analyzing `data/raw/hotel_bookings.csv`:

### [01_dataset_overview_and_cardinality.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/01_dataset_overview_and_cardinality.ipynb)
- **Problem Formulation:** Binary classification framework for `is_canceled` (prior baseline cancellation rate: 37.04%).
- **Summary Statistics:** Complete numerical and categorical descriptive statistics across all 119,390 records and 32 columns.
- **Missing Value Diagnostics:** Audits missing values in `company` (94.3%), `agent` (13.7%), `country` (0.41%), and `children` (4 rows).
- **Cardinality & Data Type Reclassification:** Identifies numeric features that should be reclassified as categorical IDs or discrete factors (`agent`, `company`, `is_repeated_guest`, `arrival_date_year`, `arrival_date_week_number`).

### [02_categorical_distributions_and_leakage.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/02_categorical_distributions_and_leakage.ipynb)
- **Categorical Distributions:** Univariate distributions across all nominal and ordinal categorical features.
- **Bivariate Associations:** Cross-tabulations and Cramér's V ranking against `is_canceled`.
- **Target Leakage Discovery:**
  - `reservation_status`: Cramér's V = 1.000 (100% target leakage; must be excluded from feature pipeline).
  - `reservation_status_date`: Post-event timestamp leakage.
  - `deposit_type` Anomaly: Non-refundable deposit bookings have an inverted ~99.4% cancellation rate due to high-risk group reservation policies.
  - `assigned_room_type` vs `reserved_room_type`: Identifies check-in timing leakage when rooms are reassigned/upgraded upon arrival.

### [03_numerical_distributions_and_leakage.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/03_numerical_distributions_and_leakage.ipynb)
- **Numerical Distributions & Outliers:** Evaluation of continuous and count variables, skewness, and extreme values (ADR < 0, ADR > 1000, zero-guest records).
- **Target Relationships:** Spearman rank and Pearson correlations with `is_canceled`.
  - `lead_time`: Strongest legitimate predictive feature ($\rho = +0.325$); cancellation rate scales monotonically from ~10% (<7 days out) to >45% (>90 days out), forming the operational basis for the overbooking glide path.
  - `total_of_special_requests` & `required_car_parking_spaces`: Strong negative predictors of cancellation.
- **Timing Leakage Audit:** Investigates `booking_changes` as a post-booking dynamic variable vs initial-booking static variable.

### [04_pairwise_distributions_and_collinearity.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/04_pairwise_distributions_and_collinearity.ipynb)
- **Numerical Collinearity:** Pearson and Spearman pairwise correlation heatmaps.
- **Variance Inflation Factor (VIF):** Confirms all numerical features have $VIF < 2.0$, demonstrating that multicollinearity does not destabilize Logistic Regression.
- **Categorical Pairwise Association:** Cramér's V heatmap highlighting significant redundancy between `market_segment` and `distribution_channel` (Cramér's V = 0.54).
- **Feature Engineering Recommendations:** Derived features for stay length (`stays_in_weekend_nights + stays_in_week_nights`) and total party size (`adults + children + babies`).

### [05_feature_and_model_exploration.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/05_feature_and_model_exploration.ipynb)
- **Data Cleaning & Leakage Exclusion:** Dropped the 2 out-of-range ADR records (-$6.38 and $5,400); strictly excluded `reservation_status`, `reservation_status_date`, and `assigned_room_type`.
- **Composite Features:** Engineered `total_stay_nights` and `total_guests`.
- **Cyclical Seasonality (`week_sin`, `week_cos` vs `month_sin`, `month_cos`):** Evaluated circular transforms across calendar intervals; confirmed week-level cyclical encoding captures granular holiday/summer peaks and eliminates Dec 31 to Jan 1 discontinuity.
- **Weekend Flag (`includes_weekend`):** Examined weekend vs weekday stay behavior across City and Resort hotels.
- **Partial Leakage Audit (Parking Spaces & Special Requests):** Discovered that across 7,416 reservations with `required_car_parking_spaces > 0`, exactly 0 cancellations occurred (0.00% cancellation rate), indicating post-booking check-in operational logging.
- **Ablation Study & Model Benchmarks:** Benchmarked Calibrated Logistic Regression and LightGBM across Scenario A (with parking & special requests) vs Scenario B (leak-free conservative baseline). LightGBM achieves 0.874 ROC-AUC even without suspect features.

### [06_model_comparison.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/06_model_comparison.ipynb)
- **Model Comparison:** Head-to-head evaluation of Logistic Regression, K-Nearest Neighbors (KNN), XGBoost, and LightGBM for individual reservation cancellation probability.
- **KNN Diminishing Returns Grid:** Tested $K \in [3, 7, 15, 25, 45, 75, 125, 201]$. Identified optimal performance at $K^* = 15$ with steady decay beyond $K=25$.
- **Standard Metrics Benchmark:** Evaluated ROC-AUC, PR-AUC, Brier score (calibration), Accuracy, Precision, Recall, F1, training time, and inference latency.
- **Asymmetric Cost Evaluation:** Applied economic loss matrix ($C_u = \$95$ unsold room loss vs $C_o = \$56$ walk penalty). Swept decision thresholds $\theta \in [0.10, 0.85]$, confirming the theoretical optimal threshold $\theta^* \approx 0.37-0.40$.
- **Economic Benchmarks:** Compared all models against Naive Baseline ($FN = \sum y_i$, \$840,275 loss, 62.96% of perfect value) and Perfect Information Oracle (\$0 loss, 100% value). LightGBM captures 86.41% of perfect value and recovers 63.30% of avoidable loss.

### [07_glidepath_and_risk_aggregation.ipynb](file:///c:/Users/joshu/Documents/Project%20Portfolio/hotel-cancellation-prediction/notebooks/07_glidepath_and_risk_aggregation.ipynb)
- **Overbooking Glide Path:** Formulated the lead-time overbooking constraint curve tapering from 125% capacity at 90+ days out down to 100% capacity at check-in.
- **Bayesian Risk Aggregation:** Modeled show-ups as a Poisson-Binomial random variable $S = \sum B_i$ with mean $\mu_S = \sum (1 - p_i)$ and variance $\sigma_S^2 = \sum p_i(1 - p_i)$. Calculated continuity-corrected overflow probability $P(S > \text{Capacity})$.
- **Historical Case Studies:** Traced on-the-books booking pace and expected show-ups across lead-time horizons for actual high-demand dates at City Hotel and Resort Hotel.
- **Multi-Date Policy Benchmark:** Compared No Overbooking (Naive), Unconstrained Overbooking, and Dynamic Bayesian Glide Path across 50 high-demand dates. The Glide Path achieved a **\$55,500 revenue lift** over Naive with **0.0% walk frequency** (zero walked guests).
- **Interactive Production Function:** Implemented and validated `evaluate_date_glidepath()` matching the FastAPI endpoint schema.
