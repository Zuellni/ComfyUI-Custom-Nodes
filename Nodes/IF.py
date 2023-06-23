import torch
from accelerate import cpu_offload, cpu_offload_with_hook
from comfy.model_management import (
    VRAMState,
    get_torch_device,
    soft_empty_cache,
    throw_exception_if_processing_interrupted,
    vram_state,
)
from comfy.utils import ProgressBar
from diffusers import DiffusionPipeline
from transformers import BitsAndBytesConfig, T5EncoderModel


class Load_Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        models = list(cls._MODELS.keys())

        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "device": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "Zuellni/IF"
    FUNCTION = "process"
    RETURN_NAMES = ("MODEL",)
    RETURN_TYPES = ("S0_MODEL",)

    _CONFIG = {
        "variant": "fp16",
        "torch_dtype": torch.float16,
        "requires_safety_checker": False,
        "feature_extractor": None,
        "safety_checker": None,
        "watermarker": None,
    }

    _MODELS = {
        "4-bit": BitsAndBytesConfig(
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        ),
        "8-bit": BitsAndBytesConfig(
            device_map="auto",
            load_in_8bit=True,
        ),
        "16-bit": None,
    }

    def offload(self, model, device):
        if device:
            return (model.to(device),)

        components = []
        device = get_torch_device()
        hook = None

        if device != "cpu":
            model = model.to("cpu")
            soft_empty_cache()

        for attr in ["unet", "text_encoder", "vae"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                components.append(getattr(model, attr))

        if vram_state.value > VRAMState.LOW_VRAM.value:
            for component in components:
                _, hook = cpu_offload_with_hook(component, device, hook)
                component.offload_hook = hook

            model.final_offload_hook = hook
        else:
            for component in components:
                cpu_offload(component, device)

        return (model,)

    def process(self, model, device):
        quantize = model != "16-bit"

        text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            subfolder="text_encoder",
            variant="fp16",
            quantization_config=__class__._MODELS[model],
        )

        model = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            text_encoder=text_encoder,
            unet=None,
            **__class__._CONFIG,
        )

        if quantize:
            return (model,)

        return self.offload(model, device)


class Load_Stage_I(Load_Encoder):
    RETURN_TYPES = ("S1_MODEL",)

    _MODELS = {
        "medium": "I-M",
        "large": "I-L",
        "extra large": "I-XL",
    }

    def process(self, model, device):
        model = DiffusionPipeline.from_pretrained(
            f"DeepFloyd/IF-{__class__._MODELS[model]}-v1.0",
            text_encoder=None,
            **__class__._CONFIG,
        )

        return self.offload(model, device)


class Load_Stage_II(Load_Encoder):
    RETURN_TYPES = ("S2_MODEL",)

    _MODELS = {
        "medium": "II-M",
        "large": "II-L",
    }

    def process(self, model, device):
        model = DiffusionPipeline.from_pretrained(
            f"DeepFloyd/IF-{__class__._MODELS[model]}-v1.0",
            text_encoder=None,
            **__class__._CONFIG,
        )

        return self.offload(model, device)


class Load_Stage_III(Load_Encoder):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("S3_MODEL",)

    def process(self, device):
        model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **__class__._CONFIG,
        )

        return self.offload(model, device)


class Encode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("S0_MODEL",),
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("S1_MODEL",),
                "positive": ("POSITIVE",),
                "negative": ("NEGATIVE",),
                "width": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "height": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
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

    def process(
        self, model, positive, negative, width, height, batch_size, seed, steps, cfg
    ):
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

        images = (images - images.min()) / (images.max() - images.min())
        images = images.clamp(0, 1)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.float32)
        return (images,)


class Stage_II:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("S2_MODEL",),
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
            height=images.shape[2] // 8 * 8 * 4,
            width=images.shape[3] // 8 * 8 * 4,
            generator=torch.manual_seed(seed),
            guidance_scale=cfg,
            num_inference_steps=steps,
            callback=callback,
            output_type="pt",
        ).images

        images = images.clamp(0, 1)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.float32)
        return (images,)


class Stage_III:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("S3_MODEL",),
                "images": ("IMAGE",),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 64}),
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

    def process(
        self, model, images, tile_size, noise, seed, steps, cfg, positive, negative
    ):
        images = images.permute(0, 3, 1, 2)
        progress = ProgressBar(steps)
        batch_size = images.shape[0]

        if batch_size > 1:
            positive = [positive] * batch_size
            negative = [negative] * batch_size

        if tile_size:
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

        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.float32)
        return (images,)
