import torch
from comfy.model_management import InterruptProcessingException, get_torch_device
from torchvision.transforms import functional as TF
from transformers import pipeline


class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aesthetic": ([False, True], {"default": False}),
                "style": ([False, True], {"default": False}),
                "waifu": ([False, True], {"default": False}),
                "age": ([False, True], {"default": False}),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("MODELS",)
    RETURN_TYPES = ("LIST",)

    def process(self, aesthetic, style, waifu, age):
        def pipe(model):
            return pipeline(model=model, device=get_torch_device())

        models = []

        aesthetic and models.append(
            {
                "pipe": pipe("cafeai/cafe_aesthetic"),
                "weights": [0.0, 1.0],
            }
        )

        style and models.append(
            {
                "pipe": pipe("cafeai/cafe_style"),
                "weights": [1.0, 0.75, 0.5, 0.0, 0.0],
            }
        )

        waifu and models.append(
            {
                "pipe": pipe("cafeai/cafe_waifu"),
                "weights": [0.0, 1.0],
            }
        )

        age and models.append(
            {
                "pipe": pipe("nateraw/vit-age-classifier"),
                "weights": [0.25, 0.5, 1.0, 0.75, 0.5, 0.0, 0.0, 0.0, 0.0],
            }
        )

        return (models,)


class Select:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 0, "max": 64}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
                "models": ("LIST",),
            },
        }

    CATEGORY = "Zuellni/Aesthetic"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "MASKS", "SCORES")
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")

    def process(self, count, images=None, latents=None, masks=None, models=None):
        if not count:
            raise InterruptProcessingException()

        if images is None or not models:
            if images is not None:
                images = images[count - 1].unsqueeze(0)

            if latents is not None:
                latents = latents["samples"]
                latents = latents[count - 1].unsqueeze(0)
                latents = {"samples": latents}

            if masks is not None:
                masks = masks[count - 1].unsqueeze(0)

            return (images, latents, masks, "")

        pil_images = images.permute(0, 3, 1, 2)
        pil_images = torch.clamp(pil_images * 255, 0, 255)
        pil_images = pil_images.to("cpu", torch.uint8)
        pil_images = [TF.to_pil_image(i) for i in pil_images]
        scores = {i: 0.0 for i in range(images.shape[0])}

        for model in models:
            pipe = model["pipe"]
            weights = model["weights"]
            labels = pipe.model.config.id2label

            w_len = len(weights)
            w_sum = sum(weights)
            w_map = {labels[i]: weights[i] for i in range(w_len)}
            values = pipe(pil_images, top_k=w_len)

            for index, value in enumerate(values):
                score = [v["score"] * w_map[v["label"]] for v in value]
                scores[index] += sum(score) / w_sum

        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:count]
        scores_str = ", ".join([f"{v:.3f}" for k, v in scores])
        images = [images[v[0]] for v in scores]
        images = torch.stack(images)

        if latents is not None:
            latents = latents["samples"]
            latents = [latents[v[0]] for v in scores]
            latents = torch.stack(latents)
            latents = {"samples": latents}

        if masks is not None:
            masks = [masks[v[0]] for v in scores]
            masks = torch.stack(masks)

        return (images, latents, masks, scores_str)
