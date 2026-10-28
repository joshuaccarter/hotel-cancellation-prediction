import pandas as pd

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.features.pipeline import (
    build_preprocessor,
    create_features,
    get_model_features,
)


def test_create_features_composites_and_cyclical():
    raw_df = pd.DataFrame(
        [
            {
                "hotel": "Resort Hotel",
                "lead_time": 45,
                "arrival_date_week_number": 27,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 5,
                "adults": 2,
                "children": 1.0,
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
                "adr": 120.0,
                # Included forbidden column to test exclusion
                "required_car_parking_spaces": 1,
                "reservation_status": "Check-Out",
            },
            {
                "hotel": "City Hotel",
                "lead_time": 5,
                "arrival_date_week_number": 1,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 3,
                "adults": 1,
                "children": 0.0,
                "babies": 0,
                "meal": "BB",
                "country": "ESP",
                "market_segment": "Corporate",
                "distribution_channel": "Corporate",
                "is_repeated_guest": 1,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 1,
                "reserved_room_type": "D",
                "booking_changes": 1,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 85.0,
            },
        ]
    )

    feats = create_features(raw_df)

    # 1. Total stay nights: 2+5=7 and 0+3=3
    assert feats.loc[0, "total_stay_nights"] == 7
    assert feats.loc[1, "total_stay_nights"] == 3

    # 2. Total guests: 2+1+0=3 and 1+0+0=1
    assert feats.loc[0, "total_guests"] == 3
    assert feats.loc[1, "total_guests"] == 1

    # 3. Includes weekend flag: 1 when weekend >= 1, else 0
    assert feats.loc[0, "includes_weekend"] == 1
    assert feats.loc[1, "includes_weekend"] == 0

    # 4. Cyclical week encodings: within [-1, 1]
    assert -1.0 <= feats.loc[0, "week_sin"] <= 1.0
    assert -1.0 <= feats.loc[0, "week_cos"] <= 1.0

    # 5. Guaranteed exclusion of forbidden features
    for forbidden in settings.data.excluded_features:
        assert forbidden not in feats.columns

    # 6. Verify preprocessor can fit and transform
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(feats)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(get_model_features())  # One-hot expansion
