from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from comfy.model_management import InterruptProcessingException
from folder_paths import get_input_directory, get_output_directory
from PIL import Image, ImageSequence
from torchvision.transforms import functional as TF


class Loader:
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
        files = []

        for file in ["gif", "jpeg", "jpg", "png"]:
            files.extend(input_dir.glob(f"*.{file}"))

        if not files:
            raise InterruptProcessingException()

        pil_images = []

        for file in files:
            image = Image.open(file)

            if getattr(image, "is_animated", True):
                for frame in ImageSequence.Iterator(image):
                    pil_images.append(frame.copy().convert("RGBA"))
            else:
                pil_images.append(image.convert("RGBA"))

        min_height = float("inf")
        min_width = float("inf")
        images = []

        for image in pil_images:
            image = TF.to_tensor(image)
            image[:3, image[3, :, :] == 0] = 0
            image = image[:3, :, :]
            min_height = min(min_height, image.shape[1])
            min_width = min(min_width, image.shape[2])
            images.append(image)

        if len(images) > 1:
            min_dim = min(min_height, min_width)
            cropped_images = []

            for image in images:
                image = TF.resize(image, min_dim)
                min_height = min(min_height, image.shape[1])
                min_width = min(min_width, image.shape[2])
                image = TF.center_crop(image, (min_height, min_width))
                cropped_images.append(image)

            images = cropped_images

        images = torch.stack(images)
        images = images.permute(0, 2, 3, 1)
        return (images,)


class Saver:
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
