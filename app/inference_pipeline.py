import joblib
import pandas as pd
from pathlib import Path
from typing import Dict
from app.weather.fallback import apply_fallbacks

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
# PREPROCESAMIENTO
# -----------------------------
def preprocess(payload: Dict) -> pd.DataFrame:
    """
    Convierte un payload validado en el dataset final
    esperado por el modelo. Nunca falla por columnas faltantes.
    """
    # Defensa MLOps: fallback incluso fuera de FastAPI
    payload = apply_fallbacks(payload)
    payload.pop("_fallback_used", None)

    df = pd.DataFrame([payload])

    # -------------------------
    # FECHA → HORA DECIMAL
    # -------------------------
    dt = pd.to_datetime(df["fecha_partida"], errors="coerce")
    df["hora_decimal"] = dt.dt.hour + dt.dt.minute / 60
    df["dia_semana"] = dt.dt.dayofweek

    # -------------------------
    # GARANTIZAR COLUMNAS
    # -------------------------
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "UNKNOWN"

    # -------------------------
    # NUMÉRICAS (IMPUTACIÓN)
    # -------------------------
    df[NUMERIC_FEATURES] = num_imputer.transform(df[NUMERIC_FEATURES])

    # -------------------------
    # CATEGÓRICAS (OHE)
    # -------------------------
    X_cat = ohe.transform(df[CATEGORICAL_FEATURES])
    X_cat = pd.DataFrame(
        X_cat,
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df.index
    )

    # -------------------------
    # DATASET FINAL
    # ORDEN CRÍTICO
    # -------------------------
    X_num = df[NUMERIC_FEATURES]
    X = pd.concat([X_num, X_cat], axis=1)

    return X

# -----------------------------
# PREDICCIÓN
# -----------------------------
def predict(payload: Dict) -> Dict:
    """
    Ejecuta inferencia completa y devuelve predicción
    en formato API-friendly.
    """
    X = preprocess(payload)

    proba = model.predict_proba(X)[0, 1]

    threshold = 0.35
    prediction = "Retrasado" if proba >= threshold else "No Retrasado"

    return {
        "prevision": prediction,
        "probabilidad": round(float(proba), 2)
    }
