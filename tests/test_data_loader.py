import pandas as pd

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.data.loader import clean_data, validate_clean_schema


def test_clean_data_removes_adr_anomalies_and_excluded_features():
    # Construct synthetic raw test data with anomalies
    sample_data = pd.DataFrame(
        [
            {
                "hotel": "Resort Hotel",
                "is_canceled": 0,
                "lead_time": 30,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": None,  # Should be imputed
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Direct",
                "distribution_channel": "Direct",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "assigned_room_type": "A",  # Forbidden
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 100.0,
                "required_car_parking_spaces": 1,  # Forbidden
                "total_of_special_requests": 2,  # Forbidden
                "reservation_status": "Check-Out",  # Forbidden
                "reservation_status_date": "2026-10-01",  # Forbidden
            },
            {
                # Out-of-range negative ADR record
                "hotel": "Resort Hotel",
                "is_canceled": 1,
                "lead_time": 10,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 1,
                "adults": 1,
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
                "assigned_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": -6.38,  # Out of range
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
                "reservation_status": "Canceled",
                "reservation_status_date": "2026-10-01",
            },
            {
                # Out-of-range extreme ADR record
                "hotel": "City Hotel",
                "is_canceled": 1,
                "lead_time": 15,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 2,
                "adults": 1,
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
                "assigned_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 5400.0,  # Out of range
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
                "reservation_status": "Canceled",
                "reservation_status_date": "2026-10-01",
            },
        ]
    )

    cleaned = clean_data(sample_data, drop_excluded=True)

    # Assert 2 anomalous ADR records were dropped
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["adr"] == 100.0
    assert cleaned.iloc[0]["children"] == 0.0

    # Assert all excluded features were dropped
    for col in settings.data.excluded_features:
        assert col not in cleaned.columns

    # Verify schema validation passes
    assert validate_clean_schema(cleaned, check_target=True) is True
