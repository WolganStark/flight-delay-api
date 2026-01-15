# -----------------------------
# build_lime_background.py
# SCRIPT OFFLINE
# NO USAR EN PRODUCCIÓN
# EJECUTAR MANUALMENTE
# -----------------------------

import sys
from pathlib import Path
import types
import joblib
import pandas as pd

# -----------------------------
# CONFIGURACIÓN DEL PATH
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# -----------------------------
# IMPORT DEL PIPELINE
# -----------------------------
try:
    import app.inference_pipeline as inf
    from app.inference_pipeline import preprocess
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"No se pudo importar inference_pipeline: {e}. "
        "Verifica que la carpeta 'app' esté en la raíz del proyecto."
    )

# -----------------------------
# MOCK DEL MODELO (para evitar carga de champion_model_v2.pkl)
# -----------------------------
inf.model = types.SimpleNamespace(
    predict=lambda X: [0]*len(X),
    predict_proba=lambda X: [[1.0, 0.0]]*len(X)
)

# -----------------------------
# RUTAS DE ARTEFACTOS
# -----------------------------
ARTIFACTS_DIR = BASE_DIR / "artifacts/current"
RAW_TRAIN_PATH = ARTIFACTS_DIR / "raw_train_v2.pkl"
OUTPUT_PATH = ARTIFACTS_DIR / "lime_background_v2.pkl"

# -----------------------------
# VALIDAR EXISTENCIA DEL RAW
# -----------------------------
if not RAW_TRAIN_PATH.exists():
    raise FileNotFoundError(f"No se encontró raw_train_v2.pkl en {RAW_TRAIN_PATH}")

# -----------------------------
# CARGAR DATA RAW
# -----------------------------
df_raw = joblib.load(RAW_TRAIN_PATH)
print(f"✅ raw_train_v2 cargado: shape {df_raw.shape}")

# -----------------------------
# TOMAR SUBSET REPRESENTATIVO
# -----------------------------
sample_size = min(5000, len(df_raw))
df_sample = df_raw.sample(sample_size, random_state=42).reset_index(drop=True)
print(f"✅ Subset para LIME creado: shape {df_sample.shape}")

# -----------------------------
# PREPROCESAMIENTO OFICIAL
# -----------------------------
# preprocess devuelve un DF por fila → concatenamos
X_lime_list = df_sample.apply(lambda row: preprocess(row.to_dict()), axis=1)
X_lime = pd.concat(X_lime_list.tolist(), ignore_index=True)
print(f"✅ Preprocesamiento completado: shape {X_lime.shape}")

# -----------------------------
# VALIDACIONES CRÍTICAS
# -----------------------------
if X_lime.isnull().any().any():
    raise ValueError("❌ Hay NaNs en X_lime")
if X_lime.shape[0] != sample_size:
    raise ValueError(f"❌ Filas incorrectas: {X_lime.shape[0]} != {sample_size}")

# -----------------------------
# SERIALIZAR LIME BACKGROUND
# -----------------------------
joblib.dump(X_lime, OUTPUT_PATH)
print(f"✅ lime_background serializado en: {OUTPUT_PATH}, shape: {X_lime.shape}")
