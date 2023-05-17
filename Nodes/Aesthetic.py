from comfy.model_management import InterruptProcessingException, get_torch_device
from transformers import pipeline
from PIL import Image
import numpy as np
import torch


class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aesthetic": ([False, True], {"default": True}),
                "waifu": ([False, True], {"default": True}),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("MODEL",)
    RETURN_TYPES = ("AE_MODEL",)

    def process(self, aesthetic, waifu):
        if aesthetic:
            aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic", device=get_torch_device())

        if waifu:
            waifu = pipeline("image-classification", "cafeai/cafe_waifu", device=get_torch_device())

        return ({"aesthetic": aesthetic, "waifu": waifu},)


class Select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 0, "max": 64}),
                "images": ("IMAGE",),
            },
            "optional": {
                "latents": ("LATENT",),
                "model": ("AE_MODEL",),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENTS")

    def process(self, count, images, latents=None, model=None):
        if not count:
            raise InterruptProcessingException()

        aesthetic = model["aesthetic"] if model else None
        waifu = model["waifu"] if model else None
        scores = {}

        if not aesthetic and not waifu:
            if latents:
                latents = latents["samples"]
                latents = {"samples": latents[count - 1].unsqueeze(0)}

            return (images[count - 1].unsqueeze(0), latents)

        for index, image in enumerate(images):
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            score = 0.0

            if aesthetic:
                for item in aesthetic(image, top_k=2):
                    score += item["score"] * (1.0 if item["label"] == "aesthetic" else -1.0)

            if waifu:
                for item in waifu(image, top_k=2):
                    score += item["score"] * (1.0 if item["label"] == "waifu" else -1.0)

            scores[index] = score

        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)
        images = [images[score[0]] for score in scores[:count]]
        images = torch.stack(images)

        if latents:
            latents = latents["samples"]
            latents = [latents[score[0]] for score in scores[:count]]
            latents = {"samples": torch.stack(latents)}

        return (images, latents)
