from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.weather.fallback import apply_fallbacks
from app.inference_pipeline import predict, model

# -----------------------
# APP
# -----------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0"
)

# -----------------------
# METRICS
# -----------------------
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

FALLBACK_COUNT = Counter(
    "fallback_total",
    "Total times any fallback was applied"
)

# -----------------------
# SCHEMAS
# -----------------------
class PredictionInput(BaseModel):
    aerolinea: str
    origen: str
    destino: str
    fecha_partida: str

    distancia_km: float = Field(..., gt=0)
    temperatura: Optional[float] = Field(None, gt=-50, lt=60)
    velocidad_viento: Optional[float] = Field(None, ge=0)
    visibilidad: Optional[float] = Field(None, ge=0)
    

class PredictionOutput(BaseModel):
    prevision: str
    probabilidad: float
    latencia_ms: float
    explicabilidad: Optional[dict]

# -----------------------
# ENDPOINTS
# -----------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_delay(data: PredictionInput, explain: bool = False):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start = time.perf_counter()

    try:
        payload = apply_fallbacks(data.model_dump())

        if payload.pop("_fallback_used", False):
            FALLBACK_COUNT.inc()

        result = predict(payload, explain=explain)

    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000
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
        "model_type": type(model).__name__ if model else None
    }

@app.get("/metrics")
def metrics() -> Response:
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
