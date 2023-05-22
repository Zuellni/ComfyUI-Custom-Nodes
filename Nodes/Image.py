from pathlib import Path
from uuid import uuid4

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
    RETURN_NAMES = ("IMAGES", "MASKS")
    RETURN_TYPES = ("IMAGE", "MASK")

    def process(self, input_dir):
        input_dir = Path(input_dir)
        files = []

        for file in ["bmp", "gif", "jpeg", "jpg", "png", "webp"]:
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

        images = []

        for image in pil_images:
            image = TF.to_tensor(image)
            image[:3, image[3, :, :] == 0] = 0
            images.append(image)

        if len(images) > 1:
            min_height = min([i.shape[1] for i in images])
            min_width = min([i.shape[2] for i in images])
            min_dim = min(min_height, min_width)
            images = [TF.resize(i, min_dim) for i in images]

            min_height = min([i.shape[1] for i in images])
            min_width = min([i.shape[2] for i in images])
            images = [TF.center_crop(i, (min_height, min_width)) for i in images]

        images = torch.stack(images)
        images = images.permute(0, 2, 3, 1)
        masks = images[:, :, :, 3]
        images = images[:, :, :, :3]
        return (images, masks)


class Saver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"default": get_output_directory()}),
                "optimize": ([False, True], {"default": False}),
                "animate": ([False, True], {"default": False}),
                "fps": ("INT", {"default": 10, "min": 1, "max": 1000}),
            },
            "optional": {
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/Image"
    FUNCTION = "process"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def process(self, images, output_dir, optimize, animate, fps, masks=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if masks is not None:
            masks = masks.unsqueeze(-1)
            images = torch.cat((images, masks), dim=-1)

        images = images.permute(0, 3, 1, 2)
        images = torch.clamp(images * 255.0, 0, 255)
        images = images.cpu().to(torch.uint8)
        images = [TF.to_pil_image(i) for i in images]

        if animate:
            images[0].save(
                output_dir / f"{uuid4().hex[:16]}.gif",
                append_images=images[1:],
                duration=1 / fps * 1000,
                optimize=optimize,
                save_all=True,
                loop=0,
            )
        else:
            for image in images:
                image.save(
                    output_dir / f"{uuid4().hex[:16]}.png",
                    optimize=optimize,
                )

        return (None,)
