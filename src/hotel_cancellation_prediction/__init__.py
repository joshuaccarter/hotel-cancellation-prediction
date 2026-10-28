"""Hotel Cancellation Prediction and Overbooking Glide Path Package."""

__version__ = "0.1.0"


def main():
    """CLI entrypoint to launch the FastAPI server."""
    from hotel_cancellation_prediction.api.app import start

    start()
