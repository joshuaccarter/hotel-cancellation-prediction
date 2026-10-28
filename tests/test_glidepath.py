import numpy as np

from hotel_cancellation_prediction.models.glidepath_model import (
    GlidePathEngine,
    compute_poisson_binomial_risk,
)


def test_poisson_binomial_risk():
    # 100 bookings with 0.8 show-up probability each -> mu_S = 80, capacity = 85
    show_ups = np.full(100, 0.8)
    stats = compute_poisson_binomial_risk(show_ups, capacity=85)
    assert stats["expected_show_ups"] == 80.0
    assert stats["expected_cancellations"] == 20.0
    assert 0.0 <= stats["overflow_probability"] <= 0.15

    # 100 bookings with 0.95 show-up probability -> mu_S = 95, capacity = 85
    show_ups_high = np.full(100, 0.95)
    stats_high = compute_poisson_binomial_risk(show_ups_high, capacity=85)
    assert stats_high["overflow_probability"] > 0.95


def test_glidepath_engine_evaluation():
    engine = GlidePathEngine()

    sample_bookings = [
        {
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
            "adr": 100.0,
        }
    ] * 20

    # Test at 45 days out (lead time factor 1.15)
    res_45 = engine.evaluate_date_glidepath(
        hotel="City Hotel",
        days_to_arrival=45,
        current_bookings=sample_bookings,
        hotel_capacity=200,
    )

    assert res_45["hotel"] == "City Hotel"
    assert res_45["total_current_bookings"] == 20
    assert res_45["max_allowed_overbooking_factor"] == 1.15
    assert res_45["max_allowed_bookings"] == 230
    assert res_45["is_within_glide_path"] is True
    assert res_45["glide_path_status"] == "ACCEPTING_RESERVATIONS"
    assert 0.0 <= res_45["overflow_probability"] <= 1.0
