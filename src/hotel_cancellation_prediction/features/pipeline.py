from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_categorical_features() -> List[str]:
    """Returns the configured legitimate booking-time categorical features."""
    return list(settings.data.booking_features.categorical)


def get_raw_numerical_features() -> List[str]:
    """Returns the configured legitimate raw numerical features."""
    return list(settings.data.booking_features.raw_numerical)


def get_engineered_features() -> List[str]:
    """Returns the names of engineered composite and cyclical numerical features."""
    return list(settings.data.booking_features.engineered_numerical)


def get_all_numerical_features() -> List[str]:
    """Returns all numerical features (raw + engineered) expected by the model."""
    return get_raw_numerical_features() + get_engineered_features()


def get_model_features() -> List[str]:
    """Returns the full ordered list of all model input features (numerical + categorical)."""
    return get_all_numerical_features() + get_categorical_features()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes booking records, creates composite and cyclical features, and ensures

    no excluded or leakage features are present in the output.
    """
    data = df.copy()

    # 1. Composite Stay Nights: stays_in_weekend_nights + stays_in_week_nights
    weekend_nights = data.get("stays_in_weekend_nights", 0)
    week_nights = data.get("stays_in_week_nights", 0)
    data["total_stay_nights"] = weekend_nights + week_nights

    # 2. Composite Guests: adults + children + babies
    adults = data.get("adults", 0)
    children = data.get("children", 0)
    if isinstance(children, pd.Series):
        children = children.fillna(0.0)
    babies = data.get("babies", 0)
    data["total_guests"] = adults + children + babies

    # 3. Weekend Stay Indicator: stays_in_weekend_nights >= 1
    if isinstance(weekend_nights, pd.Series):
        data["includes_weekend"] = (weekend_nights >= 1).astype(int)
    else:
        data["includes_weekend"] = int(weekend_nights >= 1)

    # 4. Cyclical Week Number Encodings (Week 1..53)
    if "arrival_date_week_number" in data.columns:
        week_num = data["arrival_date_week_number"].astype(float)
    else:
        # Fallback approximation from month if week number not directly provided
        month_map = {
            "January": 2,
            "February": 6,
            "March": 10,
            "April": 15,
            "May": 19,
            "June": 24,
            "July": 28,
            "August": 33,
            "September": 37,
            "October": 42,
            "November": 46,
            "December": 50,
        }
        if "arrival_date_month" in data.columns:
            week_num = data["arrival_date_month"].map(month_map).fillna(26.0)
        else:
            week_num = 26.0  # Neutral mid-year fallback

    data["week_sin"] = np.sin(2 * np.pi * week_num / 53.0)
    data["week_cos"] = np.cos(2 * np.pi * week_num / 53.0)

    # 5. Filter to strictly required features (guaranteeing no excluded/leakage features)
    model_cols = get_model_features()
    available_cols = [c for c in model_cols if c in data.columns]

    # Verify no excluded features leak in
    for forbidden in settings.data.excluded_features:
        if forbidden in data.columns and forbidden in available_cols:
            available_cols.remove(forbidden)
            logger.warning("Stripped forbidden feature from features output: %s", forbidden)

    return data[available_cols]


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for all model features."""
    num_cols = get_all_numerical_features()
    cat_cols = get_categorical_features()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )
    return preprocessor
