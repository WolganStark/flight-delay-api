from app.inference_pipeline import predict

payload = {
    "aerolinea": "AZ",
    "origen": "GIG",
    "destino": "GRU",
    "fecha_partida": "2025-11-10T14:30:00",
    "distancia_km": 350,
    "temperatura": 22,
    "velocidad_viento": 5,
    "visibilidad": 10000
}

result = predict(payload, explain=True)
print(result)
