from transformers import PretrainedConfig

class ActionCodecConfig(PretrainedConfig):
    model_type = "vq_actioncodec"

    def __init__(self):
        pass
