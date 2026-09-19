
class BaseTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data):
        raise NotImplementedError("Subclasses must implement the __call__ method.")
