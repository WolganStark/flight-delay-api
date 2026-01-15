from typing import Dict

# Todas las features que el modelo espera
DEFAULT_NUMERIC_FEATURES = {
    "distancia_km": 0.0,
    "temperatura": 0.0,
    "velocidad_viento": 5.0,
    "visibilidad": 10000.0,    
}

def apply_fallbacks(data: Dict) -> Dict:
    """
    Garantiza que todas las features numéricas requeridas
    por el modelo existan antes de crear el DataFrame.
    """
    enriched = data.copy()
    fallback_used = False

    for feature, default_value in DEFAULT_NUMERIC_FEATURES.items():
        if feature not in enriched or enriched[feature] is None:
            enriched[feature] = default_value
            fallback_used = True

    enriched["_fallback_used"] = fallback_used
    return enriched