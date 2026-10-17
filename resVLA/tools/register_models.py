from transformers import AutoModel, AutoConfig, AutoProcessor

from ..models.resVLA import ResVLA, ResVLAConfig, ResVLAProcessor
from ..models.tokenizer.vq_actioncodec.configuration_vq_codec import ActionCodecConfig
from ..models.tokenizer.vq_actioncodec.modeling_vq_codec import ActionCodec
from ..models.tokenizer.vq_actioncodec_v2.configuration_vq_codec import ActionCodecConfig as ActionCodecV2Config
from ..models.tokenizer.vq_actioncodec_v2.modeling_vq_codec import ActionCodec as ActionCodecV2
from ..models.tokenizer.vq_actioncodec_v3.configuration_vq_codec import ActionCodecConfig as ActionCodecV3Config
from ..models.tokenizer.vq_actioncodec_v3.modeling_vq_codec import ActionCodec as ActionCodecV3
from resVLA.models.tokenflow_stage2.v1.configuration_tokenflow_ar import TokenFlowARConfig
from resVLA.models.tokenflow_stage2.v1.modeling_tokenflow_ar import TokenFlowAR
from resVLA.models.tokenflow_stage2.v1.processing_tokenflow_ar import TokenFlowARProcessor


def register_resvla():
    AutoConfig.register("resvla", ResVLAConfig)
    AutoModel.register(ResVLAConfig, ResVLA)
    AutoProcessor.register(ResVLAConfig, ResVLAProcessor)


def register_tokenflow_stage1():
    # v1
    AutoConfig.register("vq_actioncodec", ActionCodecConfig)
    AutoModel.register(ActionCodecConfig, ActionCodec)
    # v2
    AutoConfig.register("vq_actioncodec_v2", ActionCodecV2Config)
    AutoModel.register(ActionCodecV2Config, ActionCodecV2)
    # v3
    AutoConfig.register("vq_actioncodec_v3", ActionCodecV3Config)
    AutoModel.register(ActionCodecV3Config, ActionCodecV3)


def register_tokenflow_stage2():
    # v1
    AutoConfig.register("tokenflow_ar", TokenFlowARConfig)
    AutoModel.register(TokenFlowARConfig, TokenFlowAR)
    AutoProcessor.register(TokenFlowARConfig, TokenFlowARProcessor)


def register_all_models():
    """
    Register all models into the AutoModel, AutoConfig, and AutoProcessor classes.

    This function should be called before using any of the Auto classes to ensure that the custom models are recognized and can be instantiated correctly.
    """
    register_resvla()
    register_tokenflow_stage1()
    register_tokenflow_stage2()
