# Flight Delay API

Python version 3.11.9

## Run locally

```bash
cd flight-delay-api
python -m venv venv
venv\Scripts\activate #Windows
pip install -r requirements.txt
uvicorn app.app:app --reload
```
## Available Endpoints

### Health check

```bash
GET /health
```
Devuelve el estado del servicio e información básica de las dependencias

Ejemplo de respuesta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "LogisticRegression",
  "database": {
    "enabled": false,
    "connected": false
  }
}
```

### Metrics

```bash
GET /metrics
```

Devuelve metricas basica en memoria del servicio

Ejemplo de respuesta:

```json
{
  "total_predictions": 12,
  "average_latency_ms": 4.82,
  "errors": 0
}
```

### Prediction

```bash
POST /predict
```

Cuerpo de la solicitud:

```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```

Respuesta: 

```json
{
  "prevision": "Retrasado",
  "probabilidad": 0.78
}
```

### API Docs

Interfaz de Swagger disponible en:
```bash
http://127.0.0.1:8000/docs
```