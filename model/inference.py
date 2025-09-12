import joblib
import pandas as pd
import os
import logging

log = logging.getLogger("inference")
log.setLevel(logging.INFO)

MODEL_PATH = os.path.join("model", "preprocesing_model.pkl")

model = joblib.load(MODEL_PATH)

def predict(features: dict) -> str:
    """
    Predict CPU or GPU given a feature dictionary.
    Logs every feature before passing it to the model.
    """
    log.info("===== FEATURE SNAPSHOT BEFORE PREDICTION =====")
    for k, v in features.items():
        log.info(f"  {k:<25} : {v}")
    log.info("===============================================")

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]
    decision = "gpu" if prediction == 1 else "cpu"

    log.info(f"MODEL PREDICTION : {decision.upper()}")

    return decision
