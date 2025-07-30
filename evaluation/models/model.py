"""
Model registry and base class for models to evaluate.
Provides registration and lookup utilities for model classes.
"""
import pkgutil
import importlib
import os
import logging

# Model registry for all available models
MODEL_REGISTRY = {}
logger = logging.getLogger(__name__)

def register_model(cls):
    """
    Decorator to register a model class with the global MODEL_REGISTRY.
    The class must have a 'model_name' class attribute.
    """
    if not hasattr(cls, 'model_name'):
        logger.error(f"Class {cls.__name__} must define a class attribute 'model_name' to be registered.")
        raise AttributeError(f"Class {cls.__name__} must define a class attribute 'model_name' to be registered.")
    MODEL_REGISTRY[cls.model_name] = cls
    logger.info(f"Registered model: {cls.model_name}")
    return cls

class Model:
    """
    Base class for all evaluation models.
    Subclasses should implement the __call__ method.
    """
    def __init__(self):
        pass

    def __call__(self, system_prompt, user_prompt):
        pass

# Lookup function for model class by model_name

def get_model_class(model_name):
    """
    Retrieve a registered model class by its model_name.
    Raises ValueError if not found.
    """
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        logger.error(f"No model registered under name '{model_name}'")
        raise ValueError(f"No model registered under name '{model_name}'")

# Import all model modules to ensure registration (for multiprocessing)
def import_all_models():
    package_dir = os.path.dirname(__file__)
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        importlib.import_module(f"evaluation.models.{module_name}")

# Call this LAST, after all definitions above
import_all_models()