from typing import Dict
#import logging

DEFAULT_WEATHER = {
    "temperatura": 0.0,
    "velocidad_viento": 5.0,
    "visibilidad": 10000.0
}

def apply_weather_fallback(data: Dict) -> Dict:
    """
    Garantiza que las variables climáticas existan.
    Si faltan, se completan con valores seguros.
    """

    enriched = data.copy()
    fallback_used = False
    for key, default_value in DEFAULT_WEATHER.items():
        if key not in enriched or enriched[key] is None:
            enriched[key] = default_value
            fallback_used = True
        
    enriched["_weather_fallback_used"] = fallback_used
    return enriched