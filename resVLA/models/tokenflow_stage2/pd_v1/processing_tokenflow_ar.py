from transformers import Qwen3VLProcessor


class TokenFlowPDProcessor(Qwen3VLProcessor):
    @classmethod
    def from_transformer_pretrained(cls, path, config):
        processor = Qwen3VLProcessor.from_pretrained(path)
        processor.tokenizer.add_tokens([
            f"<|tokenflow_reserved_{i}|>"
            for i in range(len(processor.tokenizer), config.token_ids["<action_0000>"])
        ])
        processor.tokenizer.add_tokens(config.action_tokens)
        processor.tokenizer.pad_token = processor.tokenizer.convert_ids_to_tokens(config.pad_token_id)
        processor.tokenizer.padding_side = "left"
        return cls(
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            video_processor=processor.video_processor,
            chat_template=processor.chat_template,
        )

    def save_pretrained(self, *args, **kwargs):
        self.register_for_auto_class("AutoProcessor")
        return super().save_pretrained(*args, **kwargs)
