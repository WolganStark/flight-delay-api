from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.inference_pipeline import predict, model
from fastapi import HTTPException
import time

app = FastAPI(
    title="Flight Delay Prediction API",
    version="0.0.1"
)

metrics = {
    "total_predictions": 0,
    "total_latency_ms": 0.0,
    "errors": 0
}

class PredictionInput(BaseModel):
    aerolinea: str = Field(..., json_schema_extra={"example": "AZ"})
    origen: str = Field(..., json_schema_extra={"example": "GIG"})
    destino: str = Field(..., json_schema_extra={"example": "GRU"})
    fecha_partida: str = Field(..., json_schema_extra={"example": "2025-11-10T14:30:00"})
    distancia_km: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prevision: str
    probabilidad: float


@app.post("/predict", response_model=PredictionOutput)
def predict_delay(data: PredictionInput):
    start = time.perf_counter()

    try:
        result = predict(data.model_dump())

        latency = (time.perf_counter() - start) * 1000  # in milliseconds
        metrics["total_predictions"] += 1
        metrics["total_latency_ms"] += latency

        return result
    except Exception as e:
        metrics["errors"] += 1
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None        
        }

@app.get("/metrics")
def metrics_endpoint():
    total = metrics["total_predictions"]
    avg_latency = (
        metrics["total_latency_ms"] / total
        if total > 0 else 0
    )

    return {
        "total_predictions": total,
        "average_latency_ms": round(avg_latency, 2),
        "errors": metrics["errors"]
    }
   