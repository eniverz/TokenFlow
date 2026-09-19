from . import BaseTransform


class InjectEmbodiment(BaseTransform):
    """Injects the embodiment into the data dictionary."""

    def __init__(self, embodiment: str):
        super().__init__()
        self.embodiment = embodiment

    def __call__(self, data):
        data["embodiment"] = self.embodiment
        return data