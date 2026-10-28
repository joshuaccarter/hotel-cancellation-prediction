from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ReservationPredictionRequest(BaseModel):
    hotel: str = Field(
        ..., description="Hotel property type ('Resort Hotel' or 'City Hotel')", examples=["Resort Hotel"]
    )
    lead_time: int = Field(..., ge=0, description="Number of days between booking creation and arrival", examples=[45])
    arrival_date_year: Optional[int] = Field(None, description="Year of arrival date", examples=[2026])
    arrival_date_month: Optional[str] = Field(None, description="Month of arrival date", examples=["October"])
    arrival_date_week_number: Optional[int] = Field(
        None, ge=1, le=53, description="Week number of arrival date (1-53)", examples=[44]
    )
    arrival_date_day_of_month: Optional[int] = Field(
        None, ge=1, le=31, description="Day of month of arrival date", examples=[27]
    )
    stays_in_weekend_nights: int = Field(
        0, ge=0, description="Number of weekend nights (Saturday or Sunday) booked", examples=[1]
    )
    stays_in_week_nights: int = Field(
        1, ge=0, description="Number of week nights (Monday through Friday) booked", examples=[2]
    )
    adults: int = Field(2, ge=0, description="Number of adults", examples=[2])
    children: float = Field(0.0, ge=0.0, description="Number of children", examples=[0.0])
    babies: int = Field(0, ge=0, description="Number of babies", examples=[0])
    meal: str = Field("BB", description="Meal package booked (BB, FB, HB, SC, Undefined)", examples=["BB"])
    country: str = Field("PRT", description="Country of origin ISO code", examples=["PRT"])
    market_segment: str = Field("Direct", description="Market segment designation", examples=["Direct"])
    distribution_channel: str = Field("Direct", description="Booking distribution channel", examples=["Direct"])
    is_repeated_guest: int = Field(0, ge=0, le=1, description="1 if repeated guest, 0 otherwise", examples=[0])
    previous_cancellations: int = Field(
        0, ge=0, description="Number of previous bookings canceled by the customer", examples=[0]
    )
    previous_bookings_not_canceled: int = Field(
        0, ge=0, description="Number of previous bookings not canceled", examples=[0]
    )
    reserved_room_type: str = Field("A", description="Code of room type reserved", examples=["A"])
    booking_changes: int = Field(0, ge=0, description="Number of changes made to the booking", examples=[0])
    deposit_type: str = Field(
        "No Deposit", description="Deposit type ('No Deposit', 'Non Refund', 'Refundable')", examples=["No Deposit"]
    )
    days_in_waiting_list: int = Field(
        0, ge=0, description="Days booking was on waiting list before confirmation", examples=[0]
    )
    customer_type: str = Field("Transient", description="Customer booking category", examples=["Transient"])
    adr: float = Field(100.0, ge=0.0, description="Average Daily Rate in currency units", examples=[110.50])


class ReservationPredictionResponse(BaseModel):
    cancellation_probability: float = Field(
        ..., description="Estimated probability of cancellation P(is_canceled = 1)", examples=[0.2345]
    )
    prediction: int = Field(
        ..., description="Binary outcome: 1 if predicted to cancel, 0 if predicted to show up", examples=[0]
    )
    risk_level: str = Field(..., description="Risk categorization ('LOW', 'MODERATE', 'HIGH')", examples=["LOW"])
    threshold_used: float = Field(..., description="Decision threshold applied for classification", examples=[0.40])


class BatchReservationPredictionRequest(BaseModel):
    reservations: List[ReservationPredictionRequest] = Field(..., description="List of reservations to score in batch")


class BatchReservationPredictionResponse(BaseModel):
    total_reservations: int = Field(..., description="Total number of reservations evaluated", examples=[10])
    predictions: List[ReservationPredictionResponse] = Field(..., description="Prediction results for each reservation")


class GlidePathRequest(BaseModel):
    hotel: str = Field(..., description="Hotel property type ('Resort Hotel' or 'City Hotel')", examples=["City Hotel"])
    days_to_arrival: int = Field(..., ge=0, description="Days remaining until the target arrival date", examples=[30])
    hotel_capacity: Optional[int] = Field(
        None, description="Physical room capacity (defaults from YAML config if omitted)", examples=[200]
    )
    target_risk_alpha: Optional[float] = Field(
        0.05,
        ge=0.001,
        le=0.50,
        description="Maximum allowable tail probability of overflowing physical capacity",
        examples=[0.05],
    )
    current_bookings: List[ReservationPredictionRequest] = Field(
        ...,
        description="List of all active on-the-books reservations for the target arrival date",
    )


class GlidePathResponse(BaseModel):
    hotel: str = Field(..., examples=["City Hotel"])
    days_to_arrival: int = Field(..., examples=[30])
    hotel_capacity: int = Field(..., description="Physical capacity of the property", examples=[200])
    total_current_bookings: int = Field(
        ..., description="Total active reservations currently on the books", examples=[180]
    )
    expected_cancellations: float = Field(
        ..., description="Sum of expected cancellations for these reservations", examples=[58.4]
    )
    expected_show_ups: float = Field(
        ..., description="Expected physical arrivals (Total Booked - Expected Cancellations)", examples=[121.6]
    )
    max_allowed_overbooking_factor: float = Field(
        ..., description="Allowable overbooking multiplier for this lead-time horizon", examples=[1.15]
    )
    max_allowed_bookings: int = Field(
        ..., description="Maximum allowable on-the-books reservations under the glide path", examples=[230]
    )
    remaining_booking_allowance: int = Field(
        ..., description="Additional reservations that can safely be sold before reaching the limit", examples=[50]
    )
    overflow_probability: float = Field(
        ...,
        description="Poisson-Binomial probability that actual arrivals exceed physical capacity P(S > Capacity)",
        examples=[0.0002],
    )
    is_within_glide_path: bool = Field(
        ..., description="True if current bookings do not exceed the glide path limit", examples=[True]
    )
    glide_path_status: str = Field(
        ...,
        description="Operational status code ('ACCEPTING_RESERVATIONS', 'NEAR_CAPACITY', 'OVER_GLIDE_LIMIT', 'RISK_EXCEEDS_TOLERANCE')",
        examples=["ACCEPTING_RESERVATIONS"],
    )
    recommended_action: str = Field(
        ..., description="Actionable recommendation for revenue management / PMS", examples=["ACCEPT_RESERVATIONS"]
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    threshold: float
    cost_unbooked_Cu: Dict[str, float]
    cost_overbooked_Co: Dict[str, float]
    critical_fractile: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
