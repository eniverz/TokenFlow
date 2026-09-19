from typing import Any, Dict

from PIL.Image import Image
from transformers import Qwen3VLProcessor


class ResVLAProcessor(Qwen3VLProcessor):
    """Qwen3-VL processor with a ResVLA observation encoding helper."""

    def encode_input(
        self,
        images: Dict[str, Image],
        instruction: str,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> Any:
        content = [{"type": "image", "image": image} for image in images.values()]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        return self.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors=return_tensors,
            **kwargs,
        )

    @classmethod
    def from_transformer_processor(cls, processor: Qwen3VLProcessor) -> "ResVLAProcessor":
        """Convert an already-loaded Qwen3-VL processor to ResVLAProcessor."""

        return cls(
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            video_processor=getattr(processor, "video_processor", None),
            chat_template=getattr(processor, "chat_template", None),
        )

    @classmethod
    def from_transformer_pretrained(cls, path: str, **kwargs: Any) -> "ResVLAProcessor":
        """Load a Qwen3-VL processor and return it as a ResVLAProcessor."""

        processor = Qwen3VLProcessor.from_pretrained(path, **kwargs)
        return cls.from_transformer_processor(processor)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Save the processor and make AutoProcessor restore this subclass."""

        self.register_for_auto_class("AutoProcessor")
        return super().save_pretrained(*args, **kwargs)
