from typing import Tuple

import numpy as np
from PIL import Image
import torch

from . import BaseTransform


class ResizeImage(BaseTransform):
    """
    Resize the images in the input data to the specified size.
    
    :param data: A dictionary containing the images to be resized. The images should be in the "images" key, and type should be a list of PIL.Image.Image objects.
    :return: The input data with the images resized.
    """
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def _to_pil_image(self, img) -> Image.Image:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            if np.issubdtype(arr.dtype, np.floating):
                scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]

            img = Image.fromarray(arr)

        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img.convert("RGB")

    def resize_with_pad(self, img) -> Image.Image:
        img = self._to_pil_image(img)
        target_w, target_h = self.size

        scale = min(target_w / img.width, target_h / img.height)
        resized_w = max(1, int(round(img.width * scale)))
        resized_h = max(1, int(round(img.height * scale)))
        resized = img.resize((resized_w, resized_h), resample=Image.BILINEAR)

        canvas = Image.new("RGB", (target_w, target_h))
        offset_x = (target_w - resized_w) // 2
        offset_y = (target_h - resized_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def __call__(self, data):
        data["images"] = {cam: self.resize_with_pad(img) for cam, img in data["images"].items()}
        return data
