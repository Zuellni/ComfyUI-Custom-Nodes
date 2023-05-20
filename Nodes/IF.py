import torch
from comfy.model_management import (
    InterruptProcessingException,
    throw_exception_if_processing_interrupted,
)
from comfy.utils import ProgressBar
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel


class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["NONE", "T5-FP8", "T5-FP16", "I-M", "I-L", "I-XL", "II-M", "II-L", "III"],
                    {"default": "NONE"},
                ),
                "device": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_NAMES = ("MODEL",)
    RETURN_TYPES = ("IF_MODEL",)

    def process(self, model, device):
        if model == "NONE":
            raise InterruptProcessingException()

        config = {
            "variant": "fp16",
            "torch_dtype": torch.float16,
            "requires_safety_checker": False,
            "feature_extractor": None,
            "safety_checker": None,
            "watermarker": None,
        }

        if model.startswith("T5"):
            load_in_8bit = model == "T5-FP8"

            text_encoder = T5EncoderModel.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                subfolder="text_encoder",
                load_in_8bit=load_in_8bit,
                device_map="auto" if load_in_8bit else None,
                variant="8bit" if load_in_8bit else "fp16",
            )

            model = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                text_encoder=text_encoder,
                unet=None,
                **config,
            )

            if load_in_8bit:
                return (model,)
        elif model == "III":
            model = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                **config,
            )
        else:
            model = DiffusionPipeline.from_pretrained(
                f"DeepFloyd/IF-{model}-v1.0",
                text_encoder=None,
                **config,
            )

        if device:
            return (model.to(device),)

        model.enable_model_cpu_offload()
        return (model,)


class Encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_TYPES = ("POSITIVE", "NEGATIVE")

    def process(self, model, positive, negative):
        positive, negative = model.encode_prompt(
            prompt=positive,
            negative_prompt=negative,
        )

        return (positive, negative)


class Stage_I:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "width": ("INT", {"default": 64, "min": 16, "max": 128, "step": 16}),
                "height": ("INT", {"default": 64, "min": 16, "max": 128, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, positive, negative, width, height, batch_size, seed, steps, cfg):
        progress = ProgressBar(steps)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        images = model(
            prompt_embeds=positive,
            negative_prompt_embeds=negative,
            height=height,
            width=width,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().float().permute(0, 2, 3, 1)
        return (images,)


class Stage_II:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, images, positive, negative, seed, steps, cfg):
        images = images.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size = images.shape[0]

        if batch_size > 1:
            positive = positive.repeat(batch_size, 1, 1)
            negative = negative.repeat(batch_size, 1, 1)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        images = model(
            image=images,
            prompt_embeds=positive,
            negative_prompt_embeds=negative,
            height=images.shape[2] * 4,
            width=images.shape[3] * 4,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images

        images = images.cpu().float().permute(0, 2, 3, 1)
        return (images,)


class Stage_III:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "images": ("IMAGE",),
                "tile": ([False, True], {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "noise": ("INT", {"default": 20, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, images, tile, tile_size, noise, seed, steps, cfg, positive, negative):
        images = images.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size = images.shape[0]

        if batch_size > 1:
            positive = [positive] * batch_size
            negative = [negative] * batch_size

        if tile:
            model.vae.config.sample_size = tile_size
            model.vae.enable_tiling()

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        images = model(
            image=images,
            prompt=positive,
            negative_prompt=negative,
            noise_level=noise,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images

        images = images.cpu().float().permute(0, 2, 3, 1)
        return (images,)
