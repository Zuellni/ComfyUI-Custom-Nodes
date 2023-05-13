from comfy.model_management import throw_exception_if_processing_interrupted, xformers_enabled
from comfy.utils import ProgressBar

import torchvision.transforms.functional as TF
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import torch
import gc


class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["I-M", "I-L", "I-XL", "II-M", "II-L", "III"], {"default": "I-M"}),
                "device": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "Zuellni/DeepFloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IF_MODEL",)

    def process(self, model, device):
        if model == "III":
            model = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype = torch.float16,
                requires_safety_checker = False,
                feature_extractor = None,
                safety_checker = None,
                watermarker = None,
            )

            if xformers_enabled():
                model.enable_xformers_memory_efficient_attention()
        else:
            model = DiffusionPipeline.from_pretrained(
                f"DeepFloyd/IF-{model}-v1.0",
                variant = "fp16",
                torch_dtype = torch.float16,
                requires_safety_checker = False,
                feature_extractor = None,
                safety_checker = None,
                text_encoder = None,
                watermarker = None,
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
                "unload": ([False, True], {"default": True}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "Zuellni/DeepFloyd"
    FUNCTION = "process"
    MODEL = None
    RETURN_TYPES = ("POSITIVE", "NEGATIVE",)
    TEXT_ENCODER = None

    def process(self, unload, positive, negative):
        if not Encoder.MODEL:
            Encoder.TEXT_ENCODER = T5EncoderModel.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                subfolder = "text_encoder",
                variant = "8bit",
                load_in_8bit = True,
                device_map = "auto",
            )

            Encoder.MODEL = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                text_encoder = Encoder.TEXT_ENCODER,
                requires_safety_checker = False,
                feature_extractor = None,
                safety_checker = None,
                unet = None,
                watermarker = None,
            )

        positive, negative = Encoder.MODEL.encode_prompt(
            prompt = positive,
            negative_prompt = negative,
        )

        if unload:
            del Encoder.MODEL, Encoder.TEXT_ENCODER
            gc.collect()
            Encoder.MODEL = None
            Encoder.TEXT_ENCODER = None

        return (positive, negative,)


class StageI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "model": ("IF_MODEL",),
                "width": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "height": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            },
        }

    CATEGORY = "Zuellni/DeepFloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, positive, negative, width, height, batch_size, seed, steps, cfg):
        progress = ProgressBar(steps)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        image = model(
            prompt_embeds = positive,
            negative_prompt_embeds = negative,
            width = width,
            height = height,
            generator = torch.manual_seed(seed),
            guidance_scale = cfg,
            num_images_per_prompt = batch_size,
            num_inference_steps = steps,
            callback = callback,
            output_type = "pt",
        ).images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().permute(0, 2, 3, 1)
        return (image,)


class StageII:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "model": ("IF_MODEL",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            },
        }

    CATEGORY = "Zuellni/DeepFloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, image, positive, negative, seed, steps, cfg):
        image = image.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)

        # temp scale hack, pad the image to max dim
        batch_size, channels, height, width = image.shape
        max_dim = max(height, width)
        image = TF.center_crop(image, max_dim)
        model.unet.config.sample_size = max_dim * 4

        if batch_size > 1:
            positive = positive.repeat(batch_size, 1, 1)
            negative = negative.repeat(batch_size, 1, 1)

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        image = model(
            image = image,
            prompt_embeds = positive,
            negative_prompt_embeds = negative,
            generator = torch.manual_seed(seed),
            guidance_scale = cfg,
            num_inference_steps = steps,
            callback = callback,
            output_type = "pt",
        ).images.cpu().float()

        # crop the image back to init dims * 4
        image = TF.center_crop(image, (height * 4, width * 4))
        image = image.permute(0, 2, 3, 1)
        return (image,)


class StageIII:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IF_MODEL",),
                "image": ("IMAGE",),
                "tile": ([False, True], {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "noise": ("INT", {"default": 20, "min": 0, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "Zuellni/DeepFloyd"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)

    def process(self, model, image, tile, tile_size, noise, seed, steps, cfg, positive, negative):
        image = image.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size = image.shape[0]

        if batch_size > 1:
            positive = [positive] * batch_size
            negative = [negative] * batch_size

        if tile:
            model.vae.config.sample_size = tile_size
            model.vae.enable_tiling()

        def callback(step, time_step, latent):
            throw_exception_if_processing_interrupted()
            progress.update_absolute(step)

        image = model(
            image = image,
            prompt = positive,
            negative_prompt = negative,
            noise_level = noise,
            generator = torch.manual_seed(seed),
            guidance_scale = cfg,
            num_inference_steps = steps,
            callback = callback,
            output_type = "pt",
        ).images.cpu().float().permute(0, 2, 3, 1)

        return (image,)