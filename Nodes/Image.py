from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from comfy.model_management import InterruptProcessingException
from folder_paths import get_input_directory, get_output_directory
from PIL import Image
from torchvision.transforms import functional as TF


class Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_dir": ("STRING", {"default": get_input_directory()}),
            },
        }

    CATEGORY = "Zuellni/Image"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, input_dir):
        input_dir = Path(input_dir)
        files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

        if not files:
            raise InterruptProcessingException()

        min_height = float("inf")
        min_width = float("inf")
        images = []

        for file in files:
            image = Image.open(file)
            image = TF.to_tensor(image)
            image = image[:3, :, :]
            min_height = min(min_height, image.shape[1])
            min_width = min(min_width, image.shape[2])
            images.append(image)

        if len(images) > 1:
            min_dim = min(min_height, min_width)
            images = [TF.resize(v, min_dim) for v in images]
            images = [TF.center_crop(v, (min_height, min_width)) for v in images]

        images = torch.stack(images)
        images = images.permute(0, 2, 3, 1)
        return (images,)


class Share:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"default": get_output_directory()}),
                "optimize": ([False, True], {"default": False}),
            },
        }

    CATEGORY = "Zuellni/Image"
    FUNCTION = "process"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def process(self, images, output_dir, optimize):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image in images:
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image.save(output_dir / f"{uuid4().hex[:16]}.png", optimize=optimize)

        return (None,)
