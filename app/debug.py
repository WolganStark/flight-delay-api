from app.inference_pipeline import model, CATEGORICAL_FEATURES, NUMERIC_FEATURES

def get_debug_info():
    return {
        "api_name": "Flight Delay Prediction API",
        "version": "2.0.2",
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES
        },
        "prediction_threshold": 0.35,
        "example_payload": {
            "aerolinea": "AZ",
            "origen": "GIG",
            "destino": "GRU",
            "fecha_partida": "2025-11-10T14:30:00",
            "distancia_km": 350
        },
        "links": ["/predict", "/health", "/metrics"]
    }
