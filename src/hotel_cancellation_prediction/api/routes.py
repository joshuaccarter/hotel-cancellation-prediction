import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from hotel_cancellation_prediction.api.schemas import (
    BatchReservationPredictionRequest,
    BatchReservationPredictionResponse,
    GlidePathRequest,
    GlidePathResponse,
    HealthResponse,
    ModelInfoResponse,
    ReservationPredictionRequest,
    ReservationPredictionResponse,
)
from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.models.cancellation_model import CancellationPredictor
from hotel_cancellation_prediction.models.glidepath_model import GlidePathEngine
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Singletons for dependency injection
_cancellation_predictor = CancellationPredictor(auto_load=True)
_glidepath_engine = GlidePathEngine(_cancellation_predictor)


def get_cancellation_predictor() -> CancellationPredictor:
    return _cancellation_predictor


def get_glidepath_engine() -> GlidePathEngine:
    return _glidepath_engine


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
)
async def health_check() -> HealthResponse:
    """Returns the operational status, application name, and deployed version."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app.name,
        version=settings.app.version,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model Metadata & Economics Parameters",
)
async def get_model_info(
    predictor: CancellationPredictor = Depends(get_cancellation_predictor),
) -> ModelInfoResponse:
    """Returns metadata regarding the deployed cancellation model, active cost-optimal

    threshold, and property-level asymmetric loss parameters.
    """
    return ModelInfoResponse(
        model_name="LightGBM Cancellation Pipeline",
        model_type=settings.models.individual_cancellation.model_type,
        model_version=settings.app.version,
        threshold=predictor.threshold,
        cost_unbooked_Cu={
            "Resort Hotel": settings.economics.get_cost_unbooked("Resort Hotel"),
            "City Hotel": settings.economics.get_cost_unbooked("City Hotel"),
        },
        cost_overbooked_Co={
            "Resort Hotel": settings.economics.get_cost_overbook("Resort Hotel"),
            "City Hotel": settings.economics.get_cost_overbook("City Hotel"),
        },
        critical_fractile={
            "Resort Hotel": round(settings.economics.get_critical_fractile("Resort Hotel"), 4),
            "City Hotel": round(settings.economics.get_critical_fractile("City Hotel"), 4),
        },
    )


@router.post(
    "/predict/reservation",
    response_model=ReservationPredictionResponse,
    tags=["Predictions"],
    summary="Predict cancellation likelihood for an individual booking",
)
async def predict_individual_reservation(
    payload: ReservationPredictionRequest,
    predictor: CancellationPredictor = Depends(get_cancellation_predictor),
) -> ReservationPredictionResponse:
    """Computes calibrated cancellation probability strictly using information available

    at the time of booking creation. Excludes all terminal status and check-in leakage.
    """
    try:
        result = predictor.predict(payload.model_dump())
        return ReservationPredictionResponse(**result)
    except Exception as e:
        logger.error("Error predicting individual cancellation: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/reservations",
    response_model=BatchReservationPredictionResponse,
    tags=["Predictions"],
    summary="Batch predict cancellation likelihood across multiple bookings",
)
async def predict_reservations_batch(
    payload: BatchReservationPredictionRequest,
    predictor: CancellationPredictor = Depends(get_cancellation_predictor),
) -> BatchReservationPredictionResponse:
    """Vectorized scoring of multiple individual reservations."""
    try:
        bookings_data = [b.model_dump() for b in payload.reservations]
        df_bookings = pd.DataFrame(bookings_data)
        batch_results = predictor.predict_batch(df_bookings)
        predictions = [ReservationPredictionResponse(**r) for r in batch_results]
        return BatchReservationPredictionResponse(
            total_reservations=len(predictions),
            predictions=predictions,
        )
    except Exception as e:
        logger.error("Error in batch prediction: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post(
    "/predict/glidepath",
    response_model=GlidePathResponse,
    tags=["Glide Path"],
    summary="Evaluate overbooking glide path constraints across all bookings for a target arrival date",
)
async def evaluate_glidepath(
    payload: GlidePathRequest,
    engine: GlidePathEngine = Depends(get_glidepath_engine),
) -> GlidePathResponse:
    """Takes all current on-the-books reservations for a target arrival date and applies the

    lead-time glide path constraint and Bayesian Poisson-Binomial overflow tail risk evaluation.
    """
    try:
        bookings_data = [b.model_dump() for b in payload.current_bookings]
        result = engine.evaluate_date_glidepath(
            hotel=payload.hotel,
            days_to_arrival=payload.days_to_arrival,
            current_bookings=bookings_data,
            hotel_capacity=payload.hotel_capacity,
            target_risk_alpha=payload.target_risk_alpha or 0.05,
        )
        return GlidePathResponse(**result)
    except Exception as e:
        logger.error("Error evaluating glide path: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Glide path evaluation failed: {str(e)}")
