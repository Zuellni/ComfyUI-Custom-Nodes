import numpy as np
import torch
from comfy.model_management import InterruptProcessingException, get_torch_device
from PIL import Image
from transformers import pipeline


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
                },
                "style": {
                    "pipe": pipe(style, "cafeai/cafe_style"),
                    "weights": [1.0, 0.75, 0.5, 0.0, 0.0],
                },
                "waifu": {
                    "pipe": pipe(waifu, "cafeai/cafe_waifu"),
                    "weights": [0.0, 1.0],
                },
                "age": {
                    "pipe": pipe(age, "nateraw/vit-age-classifier"),
                    "weights": [0.25, 0.5, 1.0, 0.75, 0.5, 0.0, 0.0, 0.0, 0.0],
                },
            },
        )


class Select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 0, "max": 64}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "models": ("AE_MODEL",),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS")
    RETURN_TYPES = ("IMAGE", "LATENT")

    def process(self, count, images=None, latents=None, models=None):
        if not count:
            raise InterruptProcessingException()

        if not models or images is None or all(not v["pipe"] for v in models.values()):
            if images is not None:
                images = images[count - 1].unsqueeze(0)

            if latents:
                latents = latents["samples"]
                latents = latents[count - 1].unsqueeze(0)
                latents = {"samples": latents}

            return (images, latents)

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
                    w_len = len(weights)
                    w_sum = sum(weights)
                    w_map = {labels[i]: weights[i] for i in range(w_len)}

                    items = pipe(image, top_k=w_len)
                    items = [v["score"] * w_map[v["label"]] / w_sum for v in items]
                    score += sum(items)

            scores[index] = score

        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)
        images = [images[v[0]] for v in scores[:count]]
        images = torch.stack(images)

        if latents:
            latents = latents["samples"]
            latents = [latents[v[0]] for v in scores[:count]]
            latents = torch.stack(latents)
            latents = {"samples": latents}

        return (images, latents)
