from app.inference_pipeline import model, ohe, scaler

def test_artifacts_loaded():
    """Testea que el modelo, scaler y encoder estan cargados correctamente."""
    assert model is not None, "Model is not loaded"
    assert ohe is not None, "One-Hot Encoder is not loaded"
    assert scaler is not None, "Scaler is not loaded"