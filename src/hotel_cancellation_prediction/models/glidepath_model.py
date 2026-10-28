from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.models.cancellation_model import CancellationPredictor
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_poisson_binomial_risk(show_up_probs: np.ndarray, capacity: int) -> Dict[str, float]:
    """Computes expected show-ups, variance, and tail overflow probability P(S > capacity)

    for independent non-identical Bernoulli show-up trials.
    """
    q = np.array(show_up_probs, dtype=float)
    n = len(q)
    if n == 0:
        return {
            "expected_show_ups": 0.0,
            "std_show_ups": 0.0,
            "expected_cancellations": 0.0,
            "overflow_probability": 0.0,
        }

    mu_S = float(np.sum(q))
    var_S = float(np.sum(q * (1.0 - q)))
    sigma_S = float(np.sqrt(max(var_S, 1e-6)))

    # Continuity-corrected Gaussian approximation to Poisson-Binomial distribution
    z = (capacity + 0.5 - mu_S) / sigma_S
    p_overflow = float(1.0 - stats.norm.cdf(z))

    return {
        "expected_show_ups": round(mu_S, 2),
        "std_show_ups": round(sigma_S, 2),
        "expected_cancellations": round(n - mu_S, 2),
        "overflow_probability": round(float(np.clip(p_overflow, 0.0, 1.0)), 4),
    }


class GlidePathEngine:
    """Calculates overbooking constraints and Bayesian tail risk for a target arrival date."""

    def __init__(self, cancellation_predictor: Optional[CancellationPredictor] = None):
        self.cancellation_predictor = cancellation_predictor or CancellationPredictor(auto_load=True)
        self.config = settings.models.glide_path

    def get_overbooking_factor(self, days_to_arrival: int) -> float:
        """Determines the max overbooking multiplier allowed for the given lead time."""
        for bucket in self.config.lead_time_buckets:
            if days_to_arrival >= bucket.min_days:
                if bucket.max_days is None or days_to_arrival <= bucket.max_days:
                    return bucket.max_overbooking_factor
        return 1.0  # Safe default: no overbooking allowed

    def evaluate_date_glidepath(
        self,
        hotel: str,
        days_to_arrival: int,
        current_bookings: List[Dict[str, Any]] | pd.DataFrame,
        hotel_capacity: Optional[int] = None,
        target_risk_alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Evaluates overbooking risk and constraints for all current bookings of a specific date."""
        if isinstance(current_bookings, list):
            df_bookings = pd.DataFrame(current_bookings)
        else:
            df_bookings = current_bookings.copy()

        capacity = hotel_capacity or self.config.hotel_capacity.get(hotel, 200)
        total_booked = len(df_bookings)

        # 1. Lead-time Glide Path overbooking threshold
        overbooking_factor = self.get_overbooking_factor(days_to_arrival)
        max_allowed_bookings = int(round(capacity * overbooking_factor))

        # 2. Compute individual cancellation & show-up probabilities via LightGBM
        if total_booked > 0:
            cancellation_probs = self.cancellation_predictor.predict_proba_batch(df_bookings).values
            show_up_probs = 1.0 - cancellation_probs
            risk_stats = compute_poisson_binomial_risk(show_up_probs, capacity)
        else:
            risk_stats = {
                "expected_show_ups": 0.0,
                "std_show_ups": 0.0,
                "expected_cancellations": 0.0,
                "overflow_probability": 0.0,
            }

        expected_show_ups = risk_stats["expected_show_ups"]
        expected_cancels = risk_stats["expected_cancellations"]
        overflow_prob = risk_stats["overflow_probability"]

        # 3. Constraint checking
        is_within_glide_path = total_booked <= max_allowed_bookings
        is_safe_risk = overflow_prob <= target_risk_alpha
        remaining_allowance = max(0, max_allowed_bookings - total_booked)

        # 4. Status and recommendation determination
        if total_booked > max_allowed_bookings:
            action_status = "OVER_GLIDE_LIMIT"
            recommended_action = "CLOSE_INVENTORY"
        elif not is_safe_risk:
            action_status = "RISK_EXCEEDS_TOLERANCE"
            recommended_action = "RESTRICT_SALES"
        elif expected_show_ups >= capacity * 0.95:
            action_status = "NEAR_CAPACITY"
            recommended_action = "MONITOR_CLOSELY"
        else:
            action_status = "ACCEPTING_RESERVATIONS"
            recommended_action = "ACCEPT_RESERVATIONS"

        return {
            "hotel": hotel,
            "days_to_arrival": days_to_arrival,
            "hotel_capacity": capacity,
            "total_current_bookings": total_booked,
            "expected_cancellations": expected_cancels,
            "expected_show_ups": expected_show_ups,
            "max_allowed_overbooking_factor": overbooking_factor,
            "max_allowed_bookings": max_allowed_bookings,
            "remaining_booking_allowance": remaining_allowance,
            "overflow_probability": overflow_prob,
            "is_within_glide_path": is_within_glide_path,
            "glide_path_status": action_status,
            "recommended_action": recommended_action,
        }
