from hotel_cancellation_prediction.config import settings


def test_config_loaded():
    assert settings.app.name == "Hotel Cancellation & Overbooking Glide Path API"
    assert settings.paths.raw_data_path.name == "hotel_bookings.csv"
    assert len(settings.data.booking_features.raw_numerical) > 0
    assert len(settings.data.booking_features.engineered_numerical) > 0
    assert len(settings.data.booking_features.categorical) > 0
    assert len(settings.data.excluded_features) >= 5
    assert "reservation_status" in settings.data.excluded_features
    assert "assigned_room_type" in settings.data.excluded_features
    assert "required_car_parking_spaces" in settings.data.excluded_features
    assert "total_of_special_requests" in settings.data.excluded_features
    assert settings.data.cleaning.adr_min == 0.0
    assert settings.data.cleaning.adr_max == 1000.0


def test_economics_config():
    assert settings.economics.cost_unbooked_p50["Resort Hotel"] == 75.00
    assert settings.economics.cost_unbooked_p50["City Hotel"] == 100.00
    assert settings.economics.cost_overbook_premium_p75_minus_p50["Resort Hotel"] == 50.00
    assert settings.economics.cost_overbook_premium_p75_minus_p50["City Hotel"] == 26.00
    assert settings.economics.walk_friction_penalty == 25.00

    # Test Cu, Co and critical fractile calculation
    # Resort: Cu = 75, Co = 50 + 25 = 75 -> tau* = 75 / (75 + 75) = 0.50
    assert settings.economics.get_cost_unbooked("Resort Hotel") == 75.00
    assert settings.economics.get_cost_overbook("Resort Hotel") == 75.00
    assert round(settings.economics.get_critical_fractile("Resort Hotel"), 2) == 0.50

    # City: Cu = 100, Co = 26 + 25 = 51 -> tau* = 100 / (100 + 51) ~ 0.662
    assert settings.economics.get_cost_unbooked("City Hotel") == 100.00
    assert settings.economics.get_cost_overbook("City Hotel") == 51.00
    assert round(settings.economics.get_critical_fractile("City Hotel"), 2) == 0.66
