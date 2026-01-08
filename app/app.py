from xmlrpc import client
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from app.inference_pipeline import predict, model, ohe, scaler
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="Flight Delay Prediction API",
    version="0.0.2"
)

# --------------------------------------------------
# PROMETHEUS METRICS
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction endpoint"
)

# --------------------------------------------------
# PROMETHEUS METRICS
# --------------------------------------------------

class PredictionInput(BaseModel):
    aerolinea: str = Field(..., json_schema_extra={"example": "AZ"})
    origen: str = Field(..., json_schema_extra={"example": "GIG"})
    destino: str = Field(..., json_schema_extra={"example": "GRU"})
    fecha_partida: str = Field(..., json_schema_extra={"example": "2025-11-10T14:30:00"})
    distancia_km: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prevision: str
    probabilidad: float
    latencia_ms: float

# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.post("/predict", response_model=PredictionOutput)
def predict_delay(data: PredictionInput):
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    start = time.perf_counter()

    try:
        result = predict(data.model_dump())      
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    latency_ms = (time.perf_counter() - start) * 1000  # in milliseconds
    PREDICTION_LATENCY.observe(latency_ms / 1000)

    return {
        **result,
        "latencia_ms": round(latency_ms, 2)
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        #"model_version": getattr(model, "version", "unknown"),
        "artifacts": {
            "encoder": ohe is not None,
            "scaler": scaler is not None
        }}

@app.get("/metrics")
def metrics() -> Response:
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
   