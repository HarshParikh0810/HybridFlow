import numpy as np

def build_feature_dict(operation_type: str, log_size: float, sys_state: dict) -> dict:
    """
    Build a clean feature dictionary for the model.
    - operation_type: str (e.g., 'matmul', 'conv2d', 'fft', 'elementwise', 'reduce', 'sort', etc.)
    - log_size: float (log10 of total elements or bytes)
    - sys_state: dict returned from collector.sys_state.get_system_state()
    """

    return {
        "operation_type": operation_type,
        "log_size": log_size,

        "cpu_load_pct": sys_state.get("cpu_load_pct", 0.0),

        "gpu_load_pct": sys_state.get("gpu_load_pct", 0.0),
        "gpu_temp_C": sys_state.get("gpu_temp_C", 0.0),
        "gpu_mem_pressure": sys_state.get("gpu_mem_pressure", 0.0),

        "is_battery_powered": sys_state.get("is_battery_powered", 0),
    }


def to_dataframe(features: dict):
    """
    Convert feature dict to a 1-row DataFrame for model inference.
    """
    import pandas as pd
    return pd.DataFrame([features])
