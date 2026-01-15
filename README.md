# Flight Delay Prediction API

API de inferencia para predecir retrasos en vuelos, construida con **FastAPI** y un modelo de **Machine Learning** entrenado previamente.  
El servicio expone endpoints de predicción, health check y métricas, y está preparado para ejecución local, Docker y despliegue en la nube (OCI).

---

## Stack Tecnológico

- Python 3.11
- FastAPI
- scikit-learn
- Pandas / NumPy
- Prometheus Client
- Docker

---

## Ejecución Local

### 1. Crear y activar entorno virtual

```bash
cd flight-delay-api
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / Mac
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Levantar la API
```bash
uvicorn app.app:app --reload
```
La API estará disponible en:

```text
http://127.0.0.1:8000
```

## Documentación interactiva (Swagger)
FastAPI genera automáticamente la documentación:

```text
http://127.0.0.1:8000/docs
```

## Endpoints Disponibles

### Health Ckeck

```http
GET /health
```

Verifica el estado del servicio y la carga del modelo.

Ejemplo de respuesta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "GradientBoostingClassifier"
}
```

### Predicción de retraso

```http
POST /predict
```

Payload minimo

```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```

Payload completo (opcional)

```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350,
  "temperatura": 20,
  "velocidad_viento": 2,
  "visibilidad": 10000
}
```

Respuesta

```json
{
  "prevision": "Retrasado",
  "probabilidad": 0.37,
  "latencia_ms": 2.15,
  "explicabilidad": "En proceso de desarrollo..."
}
```

## Metricas

```http
GET /metrics
```

Expone métricas en formato Prometheus, incluyendo:

- Total de requests

- Errores

- Latencia de inferencia

- Uso de fallbacks

Este endpoint está pensado para monitoreo y observabilidad en producción.

## Features del Modelo

### Variables de entrada:
- Aerolínea

- Aeropuerto de origen

- Aeropuerto de destino

- Fecha y hora de partida

- Distancia del vuelo

- Variables climáticas (opcional)

### Features derivadas en inferencia:

Algunas variables se reconstruyen de forma determinística en producción:

- `hora_decimal` (derivada de `fecha_partida`)

- `dia_semana` (derivada de `fecha_partida`)

Esto garantiza coherencia total entre entrenamiento e inferencia.

### Manejo de Valores Faltantes

La API implementa un sistema de **fallbacks seguros** para variables opcionales.
Si una variable no es enviada en el request, se asigna un valor por defecto antes del preprocesamiento.

El uso de fallbacks es registrado como métrica.

## Docker
### Build de la imagen
```bash
docker build -t flight-delay-api .
```

### Ejecutar contenedor
```bash
docker run -p 8000:8000 flight-delay-api
```

## CI/CD y Despliegue

- **CI**: se ejecuta automáticamente en cada push a `main`

- **CD**: se activa al crear un release y despliega la nueva imagen en una VM en **Oracle Cloud Infrastructure (OCI)**

El despliegue utiliza contenedores Docker y reinicio controlado del servicio.

## Estado del Proyecto

- Inferencia local validada ✅

- Docker validado ✅

- Codespaces validado ✅

- CI/CD activo ✅

- Servicio desplegado en OCI ✅

## Notas Finales

Este proyecto corresponde a una versión evolucionada respecto al MVP inicial, incorporando:

- Pipeline de inferencia robusto

- Validación estricta de features

- Observabilidad

- Automatización de despliegue