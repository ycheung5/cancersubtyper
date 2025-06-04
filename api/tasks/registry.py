# Register preprocessing & execution functions dynamically
from tasks.subtyping_models.bctypefinder import preprocess_bctypefinder, run_bctypefinder
from tasks.subtyping_models.cancersubminer import preprocess_cancersubminer, run_cancersubminer

MODEL_REGISTRY = {
    "BCtypeFinder": {
        "preprocess": preprocess_bctypefinder,
        "execute": run_bctypefinder
    },
    "CancerSubminer": {
        "preprocess":preprocess_cancersubminer,
        "execute": run_cancersubminer
    }
    # "new_model": { "preprocess": preprocess_new_model, "execute": run_new_model }
}

def get_model_pipeline(model_name: str):
    """Returns the preprocessing and execution functions for the specified model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    return MODEL_REGISTRY[model_name]["preprocess"], MODEL_REGISTRY[model_name]["execute"]