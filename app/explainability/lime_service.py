import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# RUTAS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "current"

# -----------------------------
# CARGA ARTEFACTOS
# -----------------------------
model = joblib.load(ARTIFACTS_DIR / "champion_model_v2.pkl")

# Dataset de background (MISMO orden que X_train)
# ⚠️ Este archivo debe existir (subset del train ya preprocesado)
X_TRAIN_LIME = joblib.load(ARTIFACTS_DIR / "lime_background_v2.pkl")

FEATURE_NAMES = X_TRAIN_LIME.columns.tolist()

# -----------------------------
# LIME EXPLAINER (GLOBAL)
# -----------------------------
lime_explainer = LimeTabularExplainer(
    training_data=X_TRAIN_LIME.values,
    feature_names=FEATURE_NAMES,
    class_names=["No Retrasado", "Retrasado"],
    mode="classification",
    discretize_continuous=True
)

# -----------------------------
# FUNCIÓN PRINCIPAL (DA → PROD)
# -----------------------------
def get_top_3_influential_features(
    instance: pd.DataFrame,
    n_features: int = 12
) -> dict:
    """
    Implementación productiva EXACTA del protocolo DA.
    """

    instance_values = instance.values[0]

    explanation = lime_explainer.explain_instance(
        instance_values,
        model.predict_proba,
        num_features=n_features
    )

    pred_prob = model.predict_proba(instance)[0]
    pred_class = model.predict(instance)[0]

    prob_retraso = pred_prob[1]

    top_contributions = explanation.as_list()[:3]

    top_3 = []
    for feature_desc, weight in top_contributions:
        direction = (
            "a favor del retraso"
            if weight > 0
            else "en contra del retraso"
        )

        top_3.append({
            "feature": feature_desc,
            "weight": round(float(weight), 4),
            "direction": direction
        })

    return {
        "prediccion": int(pred_class),
        "prevision": "Retrasado" if pred_class == 1 else "No Retrasado",
        "probabilidad_retraso": round(float(prob_retraso), 4),
        "top_3_features_influyentes": top_3
    }
