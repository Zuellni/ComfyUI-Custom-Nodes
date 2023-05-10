from comfy.model_management import throw_exception_if_processing_interrupted, xformers_enabled
from comfy.utils import ProgressBar

from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import torch
import gc


class Loader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (["M", "L", "XL"], {"default": "M"}),
				"stage": (["I", "II", "III"], {"default": "I"}),
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IF_MODEL",)

	def process(self, model, stage):
		if_model = None

		if stage == "III":
			if_model = DiffusionPipeline.from_pretrained(
				"stabilityai/stable-diffusion-x4-upscaler",
				torch_dtype = torch.float16,
				requires_safety_checker = False,
				feature_extractor = None,
				safety_checker = None,
				watermarker = None,
			)
		else:
			if_model = DiffusionPipeline.from_pretrained(
				f"DeepFloyd/IF-{stage}-{model}-v1.0",
				variant = "fp16",
				torch_dtype = torch.float16,
				requires_safety_checker = False,
				feature_extractor = None,
				safety_checker = None,
				text_encoder = None,
				watermarker = None,
			)

		if_model.unet.to(torch.float16, memory_format = torch.channels_last)
		if_model.enable_model_cpu_offload()

		if xformers_enabled():
			if_model.enable_xformers_memory_efficient_attention()

		return (if_model,)


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
			f"DeepFloyd/IF-I-M-v1.0",
			subfolder = "text_encoder",
			variant = "fp16",
			torch_dtype = torch.float16,
			load_in_8bit = load_in_8bit,
			device_map = "auto",
		)

		if_model = DiffusionPipeline.from_pretrained(
			f"DeepFloyd/IF-I-M-v1.0",
			text_encoder = text_encoder,
			requires_safety_checker = False,
			feature_extractor = None,
			safety_checker = None,
			unet = None,
			watermarker = None,
		)

		if xformers_enabled():
			if_model.enable_xformers_memory_efficient_attention()

		positive, negative = if_model.encode_prompt(
			prompt = positive,
			negative_prompt = negative,
		)

		if unload:
			del if_model, text_encoder
			gc.collect()

		return (positive, negative,)


class StageI:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"if_model": ("IF_MODEL",),
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, if_model, positive, negative, batch_size, seed, steps, cfg):
		progress = ProgressBar(steps)

		def callback(step, time_step, latents):
			throw_exception_if_processing_interrupted()
			progress.update_absolute(step)

		image = if_model(
			prompt_embeds = positive,
			negative_prompt_embeds = negative,
			generator = torch.manual_seed(seed),
			guidance_scale = cfg,
			num_images_per_prompt = batch_size,
			num_inference_steps = steps,
			callback = callback,
			output_type = "pt",
		).images

		image = (image / 2 + 0.5).clamp(0, 1)
		image = image.cpu().permute(0, 2, 3, 1).float()
		return (image,)


class StageII:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"if_model": ("IF_MODEL",),
				"image": ("IMAGE",),
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			},
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, if_model, image, positive, negative, seed, steps, cfg):
		image = image.permute(0, 3, 1, 2)
		progress = ProgressBar(steps)
		batch_size = image.shape[0]

		if batch_size > 1:
			positive = positive.repeat(batch_size, 1, 1)
			negative = negative.repeat(batch_size, 1, 1)

		def callback(step, time_step, latents):
			throw_exception_if_processing_interrupted()
			progress.update_absolute(step)

		image = if_model(
			image = image,
			prompt_embeds = positive,
			negative_prompt_embeds = negative,
			generator = torch.manual_seed(seed),
			guidance_scale = cfg,
			num_inference_steps = steps,
			callback = callback,
			output_type = "pt",
		).images.cpu().permute(0, 2, 3, 1).float()

		return (image,)


class StageIII:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"if_model": ("IF_MODEL",),
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

	def process(self, if_model, image, tile, tile_size, noise, seed, steps, cfg, positive, negative):
		image = image.permute(0, 3, 1, 2)
		progress = ProgressBar(steps)
		batch_size = image.shape[0]

		if batch_size > 1:
			positive = [positive] * batch_size
			negative = [negative] * batch_size

		def callback(step, time_step, latents):
			throw_exception_if_processing_interrupted()
			progress.update_absolute(step)

		if tile:
			if_model.vae.config.sample_size = tile_size
			if_model.vae.enable_tiling()

		image = if_model(
			image = image,
			prompt = positive,
			negative_prompt = negative,
			noise_level = noise,
			generator = torch.manual_seed(seed),
			guidance_scale = cfg,
			num_inference_steps = steps,
			callback = callback,
			output_type = "pt",
		).images.cpu().permute(0, 2, 3, 1).float()

		return (image,)