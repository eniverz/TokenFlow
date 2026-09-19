from typing import Any, Dict, List

import torch
from PIL.Image import Image
from torch.utils.data import default_collate
from torch.utils.data._utils.collate import default_collate_fn_map
from torch.nn.utils.rnn import pad_sequence
from transformers import ProcessorMixin


default_collate_fn_map.update({
    Image: lambda batch, *, collate_fn_map=None: list(batch),
})


class BatchCollator:
    def __init__(
        self,
        processor: ProcessorMixin,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
        return_tensors: str = "pt",
        padding_side: str | None = None,
        **processor_kwargs: Any,
    ):
        self.processor = processor
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.return_dict = return_dict
        self.return_tensors = return_tensors
        self.padding_side = padding_side
        self.processor_kwargs = processor_kwargs

        self.encode_fn = getattr(processor, "encode_input", self._encode_input)

    def _encode_input(
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

        return self.processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors=return_tensors,
            **kwargs,
        )

    def _collate_and_pad(self, encoded_vlm_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        padding_side = self.padding_side or self.processor.tokenizer.padding_side
        padding_values = {
            "input_ids": self.processor.tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
            "mm_token_type_ids": 0,
            "labels": -100,
        }

        vlm_inputs = {}
        for key in encoded_vlm_inputs[0]:
            values = [encoded[key] for encoded in encoded_vlm_inputs]
            if key in padding_values:
                # Each encoded sequence is [1, L]; pad to [B, L_max].
                vlm_inputs[key] = pad_sequence(
                    [value.squeeze(0) for value in values],
                    batch_first=True,
                    padding_value=padding_values[key],
                    padding_side=padding_side,
                )
            else:
                # Qwen packs visual patches and grid rows across all samples.
                vlm_inputs[key] = torch.cat(values, dim=0)

        return vlm_inputs

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "vlm_inputs" in samples[0]:
            raise KeyError("The sample key 'vlm_inputs' is reserved by BatchCollator")
        encoded_vlm_inputs = [
            self.encode_fn(
                images=sample["images"],
                instruction=sample["prompt"],
                tokenize=self.tokenize,
                add_generation_prompt=self.add_generation_prompt,
                return_dict=self.return_dict,
                return_tensors=self.return_tensors,
                **self.processor_kwargs,
            )
            for sample in samples
        ]
        batch = default_collate(samples)
        batch["vlm_inputs"] = self._collate_and_pad(encoded_vlm_inputs)
        return batch
