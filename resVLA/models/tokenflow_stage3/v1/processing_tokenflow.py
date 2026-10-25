from transformers import Qwen3VLProcessor


class TokenFlowProcessor(Qwen3VLProcessor):
    def save_pretrained(self, *args, **kwargs):
        self.register_for_auto_class("AutoProcessor")
        return super().save_pretrained(*args, **kwargs)
