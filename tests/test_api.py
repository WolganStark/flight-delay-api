from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_predict_ok():
    payload = {
        "aerolinea": "AZ",
        "origen": "GIG",
        "destino": "GRU",
        "fecha_partida": "2025-11-10T14:30:00",
        "distancia_km": 350
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prevision" in data
    assert "probabilidad" in data

    assert data["prevision"] in ["Retrasado", "No Retrasado"]
    assert 0.0 <= data["probabilidad"] <= 1.0