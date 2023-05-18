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
                    "weights": [0.0, 1.0],
                    "positive": 1,
                },
                "style": {
                    "pipe": pipe(style, "cafeai/cafe_style"),
                    "weights": [1.0, 1.0, 0.5, 0.0, 0.0],
                    "positive": 3,
                },
                "waifu": {
                    "pipe": pipe(waifu, "cafeai/cafe_waifu"),
                    "weights": [0.0, 1.0],
                    "positive": 1,
                },
                "age": {
                    "pipe": pipe(age, "nateraw/vit-age-classifier"),
                    "weights": [0.0, 0.5, 1.0, 0.75, 0.5, 0.0, 0.0, 0.0, 0.0],
                    "positive": 4,
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

            for model in models.values():
                pipe = model["pipe"]

                if pipe:
                    labels = pipe.model.config.id2label
                    weights = model["weights"]
                    num = len(weights)
                    map = {labels[i]: weights[i] for i in range(num)}
                    items = pipe(image, top_k=num)
                    score += sum(v["score"] * map[v["label"]] / model["positive"] for v in items)

            scores[index] = score

        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)
        images = [images[score[0]] for score in scores[:count]]
        images = torch.stack(images)

        if latents:
            latents = latents["samples"]
            latents = [latents[score[0]] for score in scores[:count]]
            latents = {"samples": torch.stack(latents)}

        return (images, latents)
