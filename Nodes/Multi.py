import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF


class Crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")

    def process(self, width, height, images=None, latents=None, masks=None):
        if images is not None:
            images = images.permute(0, 3, 1, 2)
            images = TF.center_crop(images, (height // 8 * 8, width // 8 * 8))
            images = images.permute(0, 2, 3, 1)

        if latents is not None:
            latents = latents["samples"]
            latents = TF.center_crop(latents, (height // 8, width // 8))
            latents = {"samples": latents}

        if masks is not None:
            masks = TF.center_crop(masks, (height // 8 * 8, width // 8 * 8))

        return (images, latents, masks)


class Repeat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")

    def process(self, batch_size, images=None, latents=None, masks=None):
        if batch_size > 1:
            if images is not None:
                images = images.repeat(batch_size, 1, 1, 1)

            if latents is not None:
                latents = latents["samples"]
                latents = latents.repeat(batch_size, 1, 1, 1)
                latents = {"samples": latents}

            if masks is not None:
                masks = masks.repeat(batch_size, 1, 1)

        return (images, latents, masks)


class Noise:
    @classmethod
    def INPUT_TYPES(cls):
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
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")

    def process(self, strength, color, images=None, latents=None, masks=None):
        if strength:
            if images is not None:
                shape = (images.shape[3] if color else 1,)
                noise = torch.randn(images.shape[:3] + shape)
                images = images + noise * strength

            if latents is not None:
                latents = latents["samples"]
                shape = (latents.shape[1] if color else 1,)
                noise = torch.randn(latents.shape[:1] + shape + latents.shape[2:])
                latents = latents + noise * strength
                latents = {"samples": latents}

            if masks is not None:
                noise = torch.randn(masks.shape)
                masks = masks + noise * strength

        return (images, latents, masks)


class Resize:
    @classmethod
    def INPUT_TYPES(cls):
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
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/Multi"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")

    def process(self, scale, mode, images=None, latents=None, masks=None):
        if scale != 1.0:
            if images is not None:
                images = images.permute(0, 3, 1, 2)
                images = F.interpolate(images, mode=mode, scale_factor=scale)
                images = TF.center_crop(
                    images, (images.shape[2] // 8 * 8, images.shape[3] // 8 * 8)
                )
                images = images.permute(0, 2, 3, 1)

            if latents is not None:
                latents = latents["samples"]
                latents = F.interpolate(latents, mode=mode, scale_factor=scale)
                latents = TF.center_crop(
                    latents, (latents.shape[2] // 8 * 8, latents.shape[3] // 8 * 8)
                )
                latents = {"samples": latents}

            if masks is not None:
                masks = masks.unsqueeze(1)
                masks = F.interpolate(masks, mode=mode, scale_factor=scale)
                masks = TF.center_crop(
                    masks, (masks.shape[2] // 8 * 8, masks.shape[3] // 8 * 8)
                )
                masks = masks.squeeze()

        return (images, latents, masks)
