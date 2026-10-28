import pandas as pd

from hotel_cancellation_prediction.models.cancellation_model import CancellationPredictor


def test_cancellation_predictor_loads_and_predicts():
    predictor = CancellationPredictor(auto_load=True)
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor.threshold == 0.40

    sample_booking = {
        "hotel": "City Hotel",
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
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 115.0,
    }

    result = predictor.predict(sample_booking)
    assert "cancellation_probability" in result
    assert 0.0 <= result["cancellation_probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["threshold_used"] == 0.40


def test_batch_prediction():
    predictor = CancellationPredictor(auto_load=True)
    df_sample = pd.DataFrame(
        [
            {
                "hotel": "Resort Hotel",
                "lead_time": 5,
                "arrival_date_week_number": 20,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Direct",
                "distribution_channel": "Direct",
                "is_repeated_guest": 1,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 2,
                "reserved_room_type": "A",
                "booking_changes": 1,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 90.0,
            },
            {
                "hotel": "City Hotel",
                "lead_time": 250,
                "arrival_date_week_number": 40,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Groups",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 1,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "Non Refund",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 100.0,
            },
        ]
    )

    batch_probs = predictor.predict_proba_batch(df_sample)
    assert len(batch_probs) == 2
    # Short lead time repeat guest should have lower cancellation probability than 250-day group with past cancellation
    assert batch_probs.iloc[0] < batch_probs.iloc[1]
