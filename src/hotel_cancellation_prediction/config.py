from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Hotel Cancellation & Overbooking Glide Path API"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class PathsConfig(BaseModel):
    raw_data_path: Path = Path("data/raw/hotel_bookings.csv")
    processed_data_dir: Path = Path("data/processed")
    model_artifacts_dir: Path = Path("models/artifacts")


class CleaningConfig(BaseModel):
    adr_min: float = 0.0
    adr_max: float = 1000.0


class FeatureGroups(BaseModel):
    categorical: List[str] = Field(default_factory=list)
    raw_numerical: List[str] = Field(default_factory=list)
    engineered_numerical: List[str] = Field(default_factory=list)

    @property
    def numerical(self) -> List[str]:
        return self.raw_numerical + self.engineered_numerical


class DataConfig(BaseModel):
    target_column: str = "is_canceled"
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    excluded_features: List[str] = Field(default_factory=list)
    booking_features: FeatureGroups = Field(default_factory=FeatureGroups)
    temporal_features: List[str] = Field(default_factory=list)


class EconomicsConfig(BaseModel):
    cost_unbooked_p50: Dict[str, float] = Field(default_factory=dict)
    cost_overbook_premium_p75_minus_p50: Dict[str, float] = Field(default_factory=dict)
    walk_friction_penalty: float = 25.00

    def get_cost_unbooked(self, hotel: str) -> float:
        """Returns the cost of an empty/unsold room (Cu) for a property."""
        return self.cost_unbooked_p50.get(hotel, 95.00)

    def get_cost_overbook(self, hotel: str) -> float:
        """Returns total net walk loss (Co = room premium + walk friction) for a property."""
        premium = self.cost_overbook_premium_p75_minus_p50.get(hotel, 30.00)
        return premium + self.walk_friction_penalty

    def get_critical_fractile(self, hotel: str) -> float:
        """Computes the critical fractile (optimal target probability of not overflowing capacity):
        tau* = Cu / (Cu + Co)
        """
        cu = self.get_cost_unbooked(hotel)
        co = self.get_cost_overbook(hotel)
        return cu / (cu + co)


class IndividualCancellationModelConfig(BaseModel):
    model_type: str = "lightgbm"
    threshold: float = 0.40
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 200
    learning_rate: float = 0.08
    num_leaves: int = 31
    artifact_filename: str = "lightgbm_cancellation_pipeline.joblib"


class LeadTimeBucket(BaseModel):
    min_days: int
    max_days: Optional[int] = None
    max_overbooking_factor: float


class GlidePathConfig(BaseModel):
    hotel_capacity: Dict[str, int] = Field(default_factory=dict)
    lead_time_buckets: List[LeadTimeBucket] = Field(default_factory=list)
    default_target_service_level: float = 0.95


class ModelsConfig(BaseModel):
    individual_cancellation: IndividualCancellationModelConfig = Field(
        default_factory=IndividualCancellationModelConfig
    )
    glide_path: GlidePathConfig = Field(default_factory=GlidePathConfig)


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    economics: EconomicsConfig = Field(default_factory=EconomicsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config/config.yaml") -> "Settings":
        path = Path(config_path)
        if not path.is_file():
            # Fall back to root directory if called from subdirectory
            root_candidate = Path(__file__).resolve().parents[2] / config_path
            if root_candidate.is_file():
                path = root_candidate
            else:
                return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# Global settings singleton
settings = Settings.from_yaml()
