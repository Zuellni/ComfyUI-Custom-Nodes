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
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IF_MODEL",)

	def process(self, model):
		if model == "III":
			model = DiffusionPipeline.from_pretrained(
				"stabilityai/stable-diffusion-x4-upscaler",
				torch_dtype = torch.float16,
				requires_safety_checker = False,
				feature_extractor = None,
				safety_checker = None,
				watermarker = None,
			)
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

		model.unet.to(torch.float16, memory_format = torch.channels_last)
		model.enable_model_cpu_offload()

		if xformers_enabled():
			model.enable_xformers_memory_efficient_attention()

		return (model,)


class Encoder:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"load_in_8bit": ([False, True], {"default": True}),
				"unload": ([False, True], {"default": True}),
				"positive": ("STRING", {"default": "", "multiline": True}),
				"negative": ("STRING", {"default": "", "multiline": True}),
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("POSITIVE", "NEGATIVE",)

	def process(self, load_in_8bit, unload, positive, negative):
		text_encoder = T5EncoderModel.from_pretrained(
			"DeepFloyd/IF-I-M-v1.0",
			subfolder = "text_encoder",
			variant = "fp16",
			torch_dtype = torch.float16,
			load_in_8bit = load_in_8bit,
			device_map = "auto",
		)

		model = DiffusionPipeline.from_pretrained(
			"DeepFloyd/IF-I-M-v1.0",
			text_encoder = text_encoder,
			requires_safety_checker = False,
			feature_extractor = None,
			safety_checker = None,
			unet = None,
			watermarker = None,
		)

		if xformers_enabled():
			model.enable_xformers_memory_efficient_attention()

		positive, negative = model.encode_prompt(
			prompt = positive,
			negative_prompt = negative,
		)

		if unload:
			del model, text_encoder
			gc.collect()

		return (positive, negative,)


class StageI:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("IF_MODEL",),
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"width": ("INT", {"default": 64, "min": 8, "max": 1024, "step": 8}),
				"height": ("INT", {"default": 64, "min": 8, "max": 1024, "step": 8}),
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
				"model": ("IF_MODEL",),
				"image": ("IMAGE",),
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"scale": ("INT", {"default": 4, "min": 1, "max": 10}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, model, image, positive, negative, scale, seed, steps, cfg):
		image = image.permute(0, 3, 1, 2)
		progress = ProgressBar(steps)
		batch_size, channels, height, width = image.shape
		max_dim = max(height, width)
		image = TF.center_crop(image, max_dim)
		model.unet.config.sample_size = max_dim * scale

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

		image = TF.center_crop(image, (height * scale, width * scale))
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
				"noise": ("INT", {"default": 100, "min": 0, "max": 100}),
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