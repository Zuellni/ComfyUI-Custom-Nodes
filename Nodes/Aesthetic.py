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
                "style": ([False, True], {"default": True}),
                "waifu": ([False, True], {"default": True}),
                "age": ([False, True], {"default": True}),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("MODELS",)
    RETURN_TYPES = ("AE_MODEL",)

    def process(self, aesthetic, style, waifu, age):
        def pipe(load, model):
            return pipeline(model=model, device=get_torch_device()) if load else None

        return (
            {
                "aesthetic": {
                    "pipe": pipe(aesthetic, "cafeai/cafe_aesthetic"),
                    "map": {
                        "not_aesthetic": -1.0,
                        "aesthetic": 1.0,
                    },
                },
                "style": {
                    "pipe": pipe(style, "cafeai/cafe_style"),
                    "map": {
                        "anime": 1.0,
                        "real_life": 1.0,
                        "3d": 1.0,
                        "manga_like": -1.0,
                        "other": -1.0,
                    },
                },
                "waifu": {
                    "pipe": pipe(waifu, "cafeai/cafe_waifu"),
                    "map": {
                        "not_waifu": -1.0,
                        "waifu": 1.0,
                    },
                },
                "age": {
                    "pipe": pipe(age, "nateraw/vit-age-classifier"),
                    "map": {
                        "0-2": -1.0,
                        "3-9": -1.0,
                        "10-19": 1.0,
                        "20-29": 1.0,
                        "30-39": -1.0,
                        "40-49": -1.0,
                        "50-59": -1.0,
                        "60-69": -1.0,
                        "more than 70": -1.0,
                    },
                },
            },
        )


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
                "models": ("AE_MODEL",),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, count, images, latents=None, models=None):
        if not count:
            raise InterruptProcessingException()

        if not models or all(not v["pipe"] for v in models.values()):
            if latents:
                latents = latents["samples"]
                latents = {"samples": latents[count - 1].unsqueeze(0)}

            return (images[count - 1].unsqueeze(0), latents)

        scores = {}

        for index, image in enumerate(images):
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            score = 0.0

            for value in models.values():
                pipe = value["pipe"]

                if pipe:
                    map = value["map"]
                    keys = len(map)
                    items = pipe(image, top_k=keys)
                    score += sum(v["score"] * map[v["label"]] / keys for v in items)

            scores[index] = score

        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)
        images = [images[score[0]] for score in scores[:count]]
        images = torch.stack(images)

        if latents:
            latents = latents["samples"]
            latents = [latents[score[0]] for score in scores[:count]]
            latents = {"samples": torch.stack(latents)}

        return (images, latents)
