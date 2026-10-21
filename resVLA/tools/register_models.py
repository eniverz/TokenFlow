from transformers import AutoModel, AutoConfig, AutoProcessor


def register_resvla():
    from ..models.resVLA import ResVLA, ResVLAConfig, ResVLAProcessor

    AutoConfig.register("resvla", ResVLAConfig)
    AutoModel.register(ResVLAConfig, ResVLA)
    AutoProcessor.register(ResVLAConfig, ResVLAProcessor)


def register_tokenflow_stage1():
    # v1
    from ..models.tokenizer.vq_actioncodec import ActionCodec, ActionCodecConfig

    AutoConfig.register("vq_actioncodec", ActionCodecConfig)
    AutoModel.register(ActionCodecConfig, ActionCodec)

    # v2
    from ..models.tokenizer.vq_actioncodec_v2 import ActionCodec as ActionCodecV2, ActionCodecConfig as ActionCodecV2Config

    AutoConfig.register("vq_actioncodec_v2", ActionCodecV2Config)
    AutoModel.register(ActionCodecV2Config, ActionCodecV2)

    # v3
    from ..models.tokenizer.vq_actioncodec_v3 import ActionCodec as ActionCodecV3, ActionCodecConfig as ActionCodecV3Config

    AutoConfig.register("vq_actioncodec_v3", ActionCodecV3Config)
    AutoModel.register(ActionCodecV3Config, ActionCodecV3)

    # g05_tokenizer
    from ..models.tokenizer.g05_tokenizer import G05Tokenizer, G05TokenizerConfig

    AutoConfig.register("g05_tokenizer", G05TokenizerConfig)
    AutoModel.register(G05TokenizerConfig, G05Tokenizer)


def register_tokenflow_stage2():
    # v1
    from ..models.tokenflow_stage2.v1 import TokenFlowARConfig, TokenFlowAR, TokenFlowARProcessor

    AutoConfig.register("tokenflow_ar", TokenFlowARConfig)
    AutoModel.register(TokenFlowARConfig, TokenFlowAR)
    AutoProcessor.register(TokenFlowARConfig, TokenFlowARProcessor)

    # v2
    from ..models.tokenflow_stage2.v2 import TokenFlowARConfig as TokenFlowARV2Config, TokenFlowAR as TokenFlowARV2, TokenFlowARProcessor as TokenFlowARV2Processor

    AutoConfig.register("tokenflow_ar_v2", TokenFlowARV2Config)
    AutoModel.register(TokenFlowARV2Config, TokenFlowARV2)
    AutoProcessor.register(TokenFlowARV2Config, TokenFlowARV2Processor)


def register_all_models():
    """
    Register all models into the AutoModel, AutoConfig, and AutoProcessor classes.

    This function should be called before using any of the Auto classes to ensure that the custom models are recognized and can be instantiated correctly.
    """
    register_resvla()
    register_tokenflow_stage1()
    register_tokenflow_stage2()
