from .register_config_hook import register_all_config_hooks
from .register_models import register_all_models

def register_all():
    """
    Register all configuration hooks and models.

    This function should be called before using any of the Auto classes or configuration resolvers to ensure that the custom models and hooks are recognized and can be instantiated correctly.
    """
    register_all_config_hooks()
    register_all_models()