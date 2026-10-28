from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hotel_cancellation_prediction.api.routes import get_cancellation_predictor, get_glidepath_engine, router
from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for pre-warming models on startup and teardown."""
    logger.info("Initializing Hotel Cancellation & Overbooking Glide Path API v%s...", settings.app.version)
    # Pre-load LightGBM model and preprocessor into memory
    predictor = get_cancellation_predictor()
    _ = get_glidepath_engine()
    logger.info(
        "Model pre-warmed successfully (Type: %s, Threshold: %.2f)",
        settings.models.individual_cancellation.model_type,
        predictor.threshold,
    )
    yield
    logger.info("Shutting down API service...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "Production FastAPI service for individual hotel cancellation likelihood prediction "
            "and dynamic overbooking glide path constraints using Bayesian risk aggregation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for web dashboards or external consumers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


def start():
    """CLI entrypoint for launching Uvicorn."""
    import uvicorn

    uvicorn.run(
        "hotel_cancellation_prediction.api.app:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
