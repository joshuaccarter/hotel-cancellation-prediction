from pathlib import Path
from typing import List, Optional

import pandas as pd

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_raw_data(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads raw hotel bookings CSV data with missing-value handling."""
    path = Path(filepath) if filepath else Path(settings.paths.raw_data_path)
    if not path.is_file():
        root_path = Path(__file__).resolve().parents[3] / path
        if root_path.is_file():
            path = root_path
        else:
            raise FileNotFoundError(f"Raw data file not found at: {path}")

    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path, na_values=["NULL", "null", "NA", ""])
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame, drop_excluded: bool = True) -> pd.DataFrame:
    """Cleans dataset: filters out-of-range ADR anomalies, imputes baseline missing values,

    and optionally drops excluded leakage features.
    """
    initial_len = len(df)
    clean_df = df.copy()

    # 1. Filter ADR anomalies
    adr_min = settings.data.cleaning.adr_min
    adr_max = settings.data.cleaning.adr_max
    if "adr" in clean_df.columns:
        clean_df = clean_df[(clean_df["adr"] >= adr_min) & (clean_df["adr"] <= adr_max)]
        dropped_adr = initial_len - len(clean_df)
        if dropped_adr > 0:
            logger.info("Filtered %d records with ADR outside [%.1f, %.1f]", dropped_adr, adr_min, adr_max)

    # 2. Impute baseline missing values
    if "children" in clean_df.columns:
        clean_df["children"] = clean_df["children"].fillna(0.0)
    if "country" in clean_df.columns:
        clean_df["country"] = clean_df["country"].fillna("Unknown")

    # 3. Drop excluded leakage and partial leakage features
    if drop_excluded:
        excluded = settings.data.excluded_features
        existing_excluded = [col for col in excluded if col in clean_df.columns]
        if existing_excluded:
            clean_df = clean_df.drop(columns=existing_excluded)
            logger.info("Dropped excluded features: %s", existing_excluded)

    return clean_df


def load_clean_data(filepath: Optional[str | Path] = None, drop_excluded: bool = True) -> pd.DataFrame:
    """Loads raw data and immediately applies cleaning and anomaly filtering."""
    df_raw = load_raw_data(filepath=filepath)
    return clean_data(df_raw, drop_excluded=drop_excluded)


def validate_clean_schema(df: pd.DataFrame, check_target: bool = True) -> bool:
    """Validates that all required raw booking features are present and no excluded features exist."""
    errors: List[str] = []

    # Check for forbidden leakage features
    for col in settings.data.excluded_features:
        if col in df.columns:
            errors.append(f"Forbidden excluded feature '{col}' found in dataset!")

    # Check for target column if training data
    if check_target and settings.data.target_column not in df.columns:
        errors.append(f"Target column '{settings.data.target_column}' is missing!")

    # Check required categorical and raw numerical columns
    required_cols = settings.data.booking_features.categorical + settings.data.booking_features.raw_numerical
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required booking features: {missing_cols}")

    if errors:
        for err in errors:
            logger.error("Schema validation error: %s", err)
        return False

    return True
