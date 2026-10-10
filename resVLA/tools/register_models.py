from transformers import AutoModel, AutoConfig, AutoProcessor

from ..models.resVLA import ResVLA, ResVLAConfig, ResVLAProcessor
from ..models.tokenizer.vq_actioncodec.configuration_vq_codec import ActionCodecConfig
from ..models.tokenizer.vq_actioncodec.modeling_vq_codec import ActionCodec


def register_resvla():
    AutoConfig.register("resvla", ResVLAConfig)
    AutoModel.register(ResVLAConfig, ResVLA)
    AutoProcessor.register(ResVLAConfig, ResVLAProcessor)


def register_all_models():
    """
    Register all models into the AutoModel, AutoConfig, and AutoProcessor classes.

    This function should be called before using any of the Auto classes to ensure that the custom models are recognized and can be instantiated correctly.
    """
    register_resvla()
    AutoConfig.register("vq_actioncodec", ActionCodecConfig)
    AutoModel.register(ActionCodecConfig, ActionCodec)
