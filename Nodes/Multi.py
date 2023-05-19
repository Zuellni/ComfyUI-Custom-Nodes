import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF


class Crop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 8, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, width, height, images=None, latents=None):
        if images is not None:
            images = images.permute(0, 3, 1, 2)
            images = TF.center_crop(images, (height, width))
            images = images.permute(0, 2, 3, 1)

        if latents:
            latents = latents["samples"]
            latents = TF.center_crop(latents, (height // 8, width // 8))
            latents = {"samples": latents}

        return (images, latents)


class Repeat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, batch_size, images=None, latents=None):
        if batch_size > 1:
            if images is not None:
                images = images.repeat(batch_size, 1, 1, 1)

            if latents:
                latents = latents["samples"]
                latents = latents.repeat(batch_size, 1, 1, 1)
                latents = {"samples": latents}

        return (images, latents)


class Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "color": ([False, True], {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, strength, color, images=None, latents=None):
        if strength:
            if images is not None:
                shape = (images.shape[3] if color else 1,)
                noise = torch.randn(images.shape[:3] + shape)
                images = images + noise * strength

            if latents:
                latents = latents["samples"]
                shape = (latents.shape[1] if color else 1,)
                noise = torch.randn(latents.shape[:1] + shape + latents.shape[2:])
                latents = latents + noise * strength
                latents = {"samples": latents}

        return (images, latents)


class Resize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "mode": (
                    ["area", "bicubic", "bilinear", "nearest", "nearest-exact"],
                    {"default": "nearest-exact"},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, scale, mode, images=None, latents=None):
        if scale != 1.0:
            if images is not None:
                images = images.permute(0, 3, 1, 2)
                images = F.interpolate(images, mode=mode, scale_factor=scale)
                images = images.permute(0, 2, 3, 1)

            if latents:
                latents = latents["samples"]
                latents = F.interpolate(latents, mode=mode, scale_factor=scale)
                latents = {"samples": latents}

        return (images, latents)
