from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.inference_pipeline import predict
from fastapi import HTTPException

app = FastAPI(
    title="Flight Delay Prediction API",
    version="0.0.1"
)

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
    try:
        return predict(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
