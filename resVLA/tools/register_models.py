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

    # v4
    from ..models.tokenizer.vq_actioncodec_v4 import ActionCodec as ActionCodecV4, ActionCodecConfig as ActionCodecV4Config

    AutoConfig.register("vq_actioncodec_v4", ActionCodecV4Config)
    AutoModel.register(ActionCodecV4Config, ActionCodecV4)

    # oat
    from ..models.tokenizer.oat import OAT, OATConfig

    AutoConfig.register("oat", OATConfig)
    AutoModel.register(OATConfig, OAT)

    # oat_v2
    from ..models.tokenizer.oat_v2 import OAT as OATV2, OATConfig as OATV2Config

    AutoConfig.register("oat_v2", OATV2Config)
    AutoModel.register(OATV2Config, OATV2)

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

    # v3
    from ..models.tokenflow_stage2.v3 import TokenFlowARConfig as TokenFlowARV3Config, TokenFlowAR as TokenFlowARV3, TokenFlowARProcessor as TokenFlowARV3Processor

    AutoConfig.register("tokenflow_ar_v3", TokenFlowARV3Config)
    AutoModel.register(TokenFlowARV3Config, TokenFlowARV3)
    AutoProcessor.register(TokenFlowARV3Config, TokenFlowARV3Processor)

    # parallel decoding
    from ..models.tokenflow_stage2.pd_v1 import TokenFlowPDConfig, TokenFlowPD, TokenFlowPDProcessor

    AutoConfig.register("tokenflow_pd_v1", TokenFlowPDConfig)
    AutoModel.register(TokenFlowPDConfig, TokenFlowPD)
    AutoProcessor.register(TokenFlowPDConfig, TokenFlowPDProcessor)


def register_tokenflow_stage3():
    from ..models.tokenflow_stage3.v1 import TokenFlowConfig, TokenFlow, TokenFlowProcessor

    AutoConfig.register("tokenflow_stage3", TokenFlowConfig)
    AutoModel.register(TokenFlowConfig, TokenFlow)
    AutoProcessor.register(TokenFlowConfig, TokenFlowProcessor)

def register_all_models():
    """
    Register all models into the AutoModel, AutoConfig, and AutoProcessor classes.

    This function should be called before using any of the Auto classes to ensure that the custom models are recognized and can be instantiated correctly.
    """
    register_resvla()
    register_tokenflow_stage1()
    register_tokenflow_stage2()
    register_tokenflow_stage3()
