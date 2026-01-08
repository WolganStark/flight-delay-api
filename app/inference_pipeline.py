import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path

# -----------------------------
# CARGA DE ARTEFACTOS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "current"

def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "champion.pkl")
    ohe = joblib.load(ARTIFACTS_DIR / "onehot_encoder.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_logreg.pkl")

    if not hasattr(model, "predict_proba"):
        raise ValueError("El modelo cargado no tiene el método 'predict_proba'.")
    
    return model, ohe, scaler

model, ohe, scaler = load_artifacts()

# -----------------------------
# DEFINICIÓN DE FEATURES
# -----------------------------
# dia_semana VA COMO CATEGÓRICA
CATEGORICAL_FEATURES = ["aerolinea", "origen", "destino", "dia_semana"]

# SOLO las numéricas escaladas
NUMERIC_FEATURES = ["distancia_km"]

# Variables cíclicas (NO se escalan)
CYCLIC_FEATURES = ["hora_sin", "hora_cos"]

# -----------------------------
# PREPROCESAMIENTO
# -----------------------------
def preprocess(payload: dict):
    df = pd.DataFrame([payload])

    # -------------------------
    # PARSE DATETIME
    # -------------------------
    dt = pd.to_datetime(df["fecha_partida"])

    # Día de la semana (0=lunes, 6=domingo)
    df["dia_semana"] = dt.dt.dayofweek.astype("int8")

    # -------------------------
    # HORA CÍCLICA
    # -------------------------
    # Hora fraccional
    hora_frac = (dt.dt.hour + dt.dt.minute / 60.0) / 24.0

    df["hora_sin"] = np.sin(2 * np.pi * hora_frac).astype("float32")
    df["hora_cos"] = np.cos(2 * np.pi * hora_frac).astype("float32")

    # -------------------------
    # CATEGÓRICAS (One-Hot)
    # -------------------------
    X_cat = ohe.transform(df[CATEGORICAL_FEATURES])
    X_cat = sparse.csr_matrix(X_cat)

    # -------------------------
    # NUMÉRICAS (ESCALADAS)
    # -------------------------
    X_num = df[NUMERIC_FEATURES].astype("float32")
    X_num_sparse = sparse.csr_matrix(X_num.values)
    X_num_scaled = scaler.transform(X_num_sparse)

    # -------------------------
    # CÍCLICAS (NO ESCALADAS)
    # -------------------------
    X_cyc = df[CYCLIC_FEATURES].astype("float32")
    X_cyc = sparse.csr_matrix(X_cyc.values)

    # -------------------------
    # CONCATENACIÓN FINAL
    # IMPORTANTE:
    # El orden de concatenación debe coincidir EXACTAMENTE
    # con el usado durante el entrenamiento
    # ORDEN CRÍTICO
    # -------------------------
    X = sparse.hstack([
        X_num_scaled,  # distancia_km
        X_cyc,         # hora_sin, hora_cos
        X_cat          # categóricas (incluye dia_semana)
    ])

    return X

# -----------------------------
# PREDICCIÓN
# -----------------------------
def predict(payload: dict):
    X = preprocess(payload)
    proba = model.predict_proba(X)[0, 1]

    prediction = "Retrasado" if proba >= 0.3 else "No Retrasado"

    return {
        "prevision": prediction,
        "probabilidad": round(float(proba), 2)
    }
