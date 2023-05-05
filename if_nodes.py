from comfy.model_management import get_torch_device, xformers_enabled
from comfy.utils import ProgressBar
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import torch
import gc


class Encode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"load_in_8bit": ([False, True], {"default": False}),
				"unload": ([False, True], {"default": False}),
				"positive": ("STRING", {"default": "", "multiline": True}),
				"negative": ("STRING", {"default": "", "multiline": True}),
			}
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("POSITIVE", "NEGATIVE",)

	def process(self, batch_size, unload, load_in_8bit, positive, negative):
		text_encoder = T5EncoderModel.from_pretrained(
			"DeepFloyd/IF-I-XL-v1.0",
			subfolder = "text_encoder",
			variant = "fp16",
			torch_dtype = torch.float16,
			load_in_8bit = load_in_8bit,
			device_map = "auto",
		)

		pipe = DiffusionPipeline.from_pretrained(
			"DeepFloyd/IF-I-XL-v1.0",
			text_encoder = text_encoder,
			requires_safety_checker = False,
			safety_checker = None,
			unet = None,
			watermarker = None,
		)

		positive, negative = pipe.encode_prompt(
			prompt = positive,
			negative_prompt = negative,
			num_images_per_prompt = batch_size
		)

		if unload:
			del pipe, text_encoder
			gc.collect()

		return (positive, negative,)


class StageI:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"model": (["M", "L", "XL"], {"default": "XL"}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			}
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, positive, negative, model, seed, steps, cfg):
		progress = ProgressBar(steps)

		def callback(step, time_step, latents):
			progress.update_absolute(step)

		pipe = DiffusionPipeline.from_pretrained(
			f"DeepFloyd/IF-I-{model}-v1.0",
			variant = "fp16",
			torch_dtype = torch.float16,
			requires_safety_checker = False,
			safety_checker = None,
			text_encoder = None,
			watermarker = None,
		)

		pipe.to(get_torch_device())
		pipe.unet.to(memory_format = torch.channels_last)

		image = pipe(
			prompt_embeds = positive,
			negative_prompt_embeds = negative,
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			callback = callback,
			output_type = "pt",
		).images.permute(0, 2, 3, 1).cpu()

		return (image,)


class StageII:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"positive": ("POSITIVE",),
				"negative": ("NEGATIVE",),
				"model": (["M", "L"], {"default": "L"}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			}
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, image, positive, negative, model, seed, steps, cfg):
		image = image.permute(0, 3, 1, 2)
		progress = ProgressBar(steps)

		def callback(step, time_step, latents):
			progress.update_absolute(step)

		pipe = DiffusionPipeline.from_pretrained(
			f"DeepFloyd/IF-II-{model}-v1.0",
			variant = "fp16",
			torch_dtype = torch.float16,
			requires_safety_checker = False,
			safety_checker = None,
			text_encoder = None,
			watermarker = None,
		)

		pipe.to(get_torch_device())
		pipe.unet.to(memory_format = torch.channels_last)

		image = pipe(
			image = image,
			prompt_embeds = positive,
			negative_prompt_embeds = negative,
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			callback = callback,
			output_type = "pt",
		).images.permute(0, 2, 3, 1).cpu()

		return (image,)


class StageIII:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"tile": ([False, True], {"default": False}),
				"tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
				"positive": ("STRING", {"default": "", "multiline": True}),
				"negative": ("STRING", {"default": "", "multiline": True}),
			}
		}

	CATEGORY = "Zuellni/DeepFloyd"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, image, unload, tile, tile_size, positive, negative, seed, steps, cfg):
		image = image.permute(0, 3, 1, 2)
		batch_size = image.shape[0]
		progress = ProgressBar(steps)

		def callback(step, time_step, latents):
			progress.update_absolute(step)

		if batch_size > 1:
			positive = [positive] * batch_size
			negative = [negative] * batch_size

		pipe = DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-x4-upscaler",
			torch_dtype = torch.float16,
			requires_safety_checker = False,
			safety_checker = None,
			watermarker = None,
		)

		pipe.to(get_torch_device())
		pipe.unet.to(memory_format = torch.channels_last)

		if xformers_enabled():
			pipe.enable_xformers_memory_efficient_attention()

		if tile:
			pipe.vae.config.sample_size = tile_size
			pipe.vae.enable_tiling()

		image = pipe(
			image = image,
			prompt = positive,
			negative_prompt = negative,
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			callback = callback,
			output_type = "pt",
		).images.permute(0, 2, 3, 1).cpu()

		return (image,)