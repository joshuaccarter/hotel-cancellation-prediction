from fastapi.testclient import TestClient

from hotel_cancellation_prediction.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "LightGBM Cancellation Pipeline"
    assert data["threshold"] == 0.40
    assert "Resort Hotel" in data["cost_unbooked_Cu"]
    assert "City Hotel" in data["cost_unbooked_Cu"]
    assert "Resort Hotel" in data["cost_overbooked_Co"]
    assert "City Hotel" in data["cost_overbooked_Co"]
    assert "Resort Hotel" in data["critical_fractile"]
    assert "City Hotel" in data["critical_fractile"]
    # Check that critical fractile matches Cu / (Cu + Co)
    assert 0.30 <= data["critical_fractile"]["Resort Hotel"] <= 0.75
    assert 0.30 <= data["critical_fractile"]["City Hotel"] <= 0.75


def test_predict_reservation_single():
    sample_booking = {
        "hotel": "Resort Hotel",
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
        "adr": 110.5,
    }
    response = client.post("/predict/reservation", json=sample_booking)
    assert response.status_code == 200
    data = response.json()
    assert "cancellation_probability" in data
    assert 0.0 <= data["cancellation_probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["risk_level"] in ("LOW", "MODERATE", "HIGH")
    assert data["threshold_used"] == 0.40


def test_predict_reservation_validation_errors():
    # Negative lead time
    bad_payload = {
        "hotel": "Resort Hotel",
        "lead_time": -10,
        "adr": 100.0,
    }
    response = client.post("/predict/reservation", json=bad_payload)
    assert response.status_code == 422

    # Negative adr
    bad_payload2 = {
        "hotel": "Resort Hotel",
        "lead_time": 10,
        "adr": -25.0,
    }
    response = client.post("/predict/reservation", json=bad_payload2)
    assert response.status_code == 422

    # Invalid week number (>53)
    bad_payload3 = {
        "hotel": "Resort Hotel",
        "lead_time": 10,
        "arrival_date_week_number": 60,
    }
    response = client.post("/predict/reservation", json=bad_payload3)
    assert response.status_code == 422


def test_predict_reservations_batch():
    sample_booking_1 = {
        "hotel": "Resort Hotel",
        "lead_time": 10,
        "adr": 85.0,
        "market_segment": "Direct",
        "deposit_type": "No Deposit",
    }
    sample_booking_2 = {
        "hotel": "City Hotel",
        "lead_time": 250,
        "adr": 120.0,
        "market_segment": "Online TA",
        "deposit_type": "Non Refund",
    }
    batch_payload = {
        "reservations": [sample_booking_1, sample_booking_2, sample_booking_1],
    }
    response = client.post("/predict/reservations", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reservations"] == 3
    assert len(data["predictions"]) == 3
    for p in data["predictions"]:
        assert 0.0 <= p["cancellation_probability"] <= 1.0
        assert p["prediction"] in (0, 1)
        assert p["risk_level"] in ("LOW", "MODERATE", "HIGH")
        assert p["threshold_used"] == 0.40


def test_predict_glidepath_accepting_reservations():
    # Horizon: 45 days (overbooking factor 1.15)
    # Capacity 200 -> max allowed 230
    # Current bookings: 50 -> well below capacity, status ACCEPTING_RESERVATIONS
    sample_booking = {
        "hotel": "Resort Hotel",
        "lead_time": 45,
        "adr": 100.0,
        "market_segment": "Direct",
    }
    glidepath_request = {
        "hotel": "Resort Hotel",
        "days_to_arrival": 45,
        "hotel_capacity": 200,
        "target_risk_alpha": 0.05,
        "current_bookings": [sample_booking] * 50,
    }
    response = client.post("/predict/glidepath", json=glidepath_request)
    assert response.status_code == 200
    data = response.json()
    assert data["hotel"] == "Resort Hotel"
    assert data["total_current_bookings"] == 50
    assert data["days_to_arrival"] == 45
    assert data["hotel_capacity"] == 200
    assert data["max_allowed_bookings"] == 230
    assert data["remaining_booking_allowance"] == 180
    assert data["is_within_glide_path"] is True
    assert data["glide_path_status"] == "ACCEPTING_RESERVATIONS"
    assert data["recommended_action"] == "ACCEPT_RESERVATIONS"
    assert data["expected_show_ups"] < 50
    assert 0.0 <= data["overflow_probability"] <= 0.05


def test_predict_glidepath_over_glide_limit():
    # Horizon: 2 days (overbooking factor 1.00)
    # Capacity 100 -> max allowed 100
    # Current bookings: 110 -> exceeds max allowed bookings
    sample_booking = {
        "hotel": "City Hotel",
        "lead_time": 2,
        "adr": 100.0,
    }
    glidepath_request = {
        "hotel": "City Hotel",
        "days_to_arrival": 2,
        "hotel_capacity": 100,
        "target_risk_alpha": 0.05,
        "current_bookings": [sample_booking] * 110,
    }
    response = client.post("/predict/glidepath", json=glidepath_request)
    assert response.status_code == 200
    data = response.json()
    assert data["is_within_glide_path"] is False
    assert data["glide_path_status"] == "OVER_GLIDE_LIMIT"
    assert data["recommended_action"] == "CLOSE_INVENTORY"
    assert data["remaining_booking_allowance"] == 0


def test_predict_glidepath_risk_exceeds_tolerance():
    # When current bookings are within the raw overbooking multiplier,
    # but actual show-up distribution tail risk exceeds a strict alpha tolerance.
    sample_booking = {
        "hotel": "Resort Hotel",
        "lead_time": 5,
        "adr": 100.0,
        "market_segment": "Direct",
    }
    # Horizon: 10 days (factor 1.05). Capacity: 50 -> max_allowed = 52.
    # Current bookings: 52 (within glide limit).
    # With a super tight target_risk_alpha (0.001), tail overflow risk will exceed alpha.
    glidepath_request = {
        "hotel": "Resort Hotel",
        "days_to_arrival": 10,
        "hotel_capacity": 50,
        "target_risk_alpha": 0.001,
        "current_bookings": [sample_booking] * 52,
    }
    response = client.post("/predict/glidepath", json=glidepath_request)
    assert response.status_code == 200
    data = response.json()
    assert data["is_within_glide_path"] is True
    assert data["glide_path_status"] == "RISK_EXCEEDS_TOLERANCE"
    assert data["recommended_action"] == "RESTRICT_SALES"


def test_predict_glidepath_empty_bookings():
    glidepath_request = {
        "hotel": "City Hotel",
        "days_to_arrival": 60,
        "hotel_capacity": 150,
        "current_bookings": [],
    }
    response = client.post("/predict/glidepath", json=glidepath_request)
    assert response.status_code == 200
    data = response.json()
    assert data["total_current_bookings"] == 0
    assert data["expected_cancellations"] == 0.0
    assert data["expected_show_ups"] == 0.0
    assert data["overflow_probability"] == 0.0
    assert data["is_within_glide_path"] is True
    assert data["glide_path_status"] == "ACCEPTING_RESERVATIONS"
    assert data["recommended_action"] == "ACCEPT_RESERVATIONS"
