import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List
from app.weather.fallback import apply_fallbacks
from app.explainability.lime_service import get_top_3_influential_features

# -----------------------------
# RUTAS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "current"

# -----------------------------
# CARGA DE ARTEFACTOS
# -----------------------------
model = joblib.load(ARTIFACTS_DIR / "champion_model_v2.pkl")
ohe = joblib.load(ARTIFACTS_DIR / "onehot_encoder_v2.pkl")
num_imputer = joblib.load(ARTIFACTS_DIR / "num_imputer_v2.pkl")

# -----------------------------
# DEFINICIÓN DE FEATURES
# -----------------------------
CATEGORICAL_FEATURES = [
    "aerolinea",
    "origen",
    "destino",
    "dia_semana"
]

NUMERIC_FEATURES = [
    "distancia_km",
    "hora_decimal",
    "temperatura",
    "velocidad_viento",
    "visibilidad"    
]

# -----------------------------
# PREPROCESAMIENTO BATCH
# -----------------------------
def preprocess_batch(payloads):
    if isinstance(payloads, list):
        df = pd.DataFrame(payloads)
    elif isinstance(payloads, pd.DataFrame):
        df = payloads.copy()
    else:
        raise ValueError("payloads debe ser lista de dicts o DataFrame")

    # Aplicar fallback fila por fila
    df = df.apply(lambda row: apply_fallbacks(row.to_dict()), axis=1, result_type='expand')
    df = df.drop(columns=["_fallback_used"], errors="ignore")  # CAMBIO: fix error 400

    # Fecha → hora_decimal y dia_semana
    dt = pd.to_datetime(df["fecha_partida"], errors="coerce")
    df["hora_decimal"] = dt.dt.hour + dt.dt.minute / 60
    df["dia_semana"] = dt.dt.dayofweek

    # Columnas faltantes
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "UNKNOWN"

    # Imputación numérica
    df[NUMERIC_FEATURES] = num_imputer.transform(df[NUMERIC_FEATURES])

    # OHE categórico
    X_cat = ohe.transform(df[CATEGORICAL_FEATURES])
    X_cat = pd.DataFrame(
        X_cat,
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df.index
    )

    X_num = df[NUMERIC_FEATURES]
    X = pd.concat([X_num, X_cat], axis=1)
    return X

# -----------------------------
# PREDICCIÓN SINGLE RECORD
# -----------------------------
def predict(payload: Dict, explain: bool = False):
    X = preprocess_batch([payload])  # Reusar batch prep
    proba = model.predict_proba(X)[0, 1]

    threshold = 0.35
    prediction = "Retrasado" if proba >= threshold else "No Retrasado"

    result = {
        "prevision": prediction,
        "probabilidad": round(float(proba), 2)
    }

    if explain:
        lime_result = get_top_3_influential_features(X)
        result['explicabilidad'] = {
            'metodo': 'LIME',
            'top_3_features': lime_result['top_3_features_influyentes']
        }

    return result

# -----------------------------
# PREDICCIÓN BATCH
# -----------------------------
def predict_batch(payloads: List[Dict]):
    X = preprocess_batch(payloads)
    probas = model.predict_proba(X)[:, 1]

    threshold = 0.35
    predictions = ["Retrasado" if p >= threshold else "No Retrasado" for p in probas]

    results = []
    for i, p in enumerate(probas):
        result = {
            "prevision": predictions[i],
            "probabilidad": round(float(p), 2)
            # No LIME en batch
        }
        results.append(result)
    return results
