from transformers import T5EncoderModel, pipeline, logging as transformers_logging
from diffusers import DiffusionPipeline, logging as diffusers_logging
from comfy.model_management import get_torch_device, xformers_enabled
from comfy.utils import ProgressBar, common_upscale
from warnings import filterwarnings
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import torch
import gc


transformers_logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()
logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")
filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")


class ZDecode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latents": ("LATENT",),
				"vae": ("VAE",),
				"tile": ([False, True], {"default": False}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, vae, latents, tile):
		return (vae.decode_tiled(latents["samples"]) if tile else vae.decode(latents["samples"]),)


class ZEncode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"vae": ("VAE",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"tile": ([False, True], {"default": False}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, images, vae, batch_size, tile):
		images = images[:, :, :, :3]
		x = (images.shape[1] // 8) * 8
		y = (images.shape[2] // 8) * 8

		if images.shape[1] != x or images.shape[2] != y:
			x_offset = (images.shape[1] % 8) // 2
			y_offset = (images.shape[2] % 8) // 2
			images = images[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

		if batch_size > 1:
			images = images.repeat(batch_size, 1, 1, 1)

		return ({"samples": vae.encode_tiled(images) if tile else vae.encode(images)},)


class ZFilter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"indexes": ("INT", {"default": 1, "min": 1, "max": 64}),
				"aesthetic": ([False, True], {"default": True}),
				"waifu": ([False, True], {"default": True}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LIST",)

	def process(self, images, indexes, aesthetic, waifu):
		if not aesthetic and not waifu:
			return (images[indexes - 1].unsqueeze(0), [indexes - 1],)

		a_model = pipeline("image-classification", "cafeai/cafe_aesthetic", device = get_torch_device())
		w_model = pipeline("image-classification", "cafeai/cafe_waifu", device = get_torch_device())
		scores = {}

		for index, image in enumerate(images):
			image = 255.0 * image.cpu().numpy()
			image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
			score = 0.0

			if aesthetic:
				data = a_model(image, top_k = 2)

				for item in data:
					score += item["score"] * (1.0 if item["label"] == "aesthetic" else -1.0)

			if waifu:
				data = w_model(image, top_k = 2)

				for item in data:
					score += item["score"] * (1.0 if item["label"] == "waifu" else -1.0)

			scores[index] = score

		sorted_scores = sorted(scores.items(), key = lambda x: x[1], reverse = True)
		top_images = [images[score[0]] for score in sorted_scores[:indexes]]
		top_images = torch.stack(top_images)
		top_indexes = [score[0] for score in sorted_scores[:indexes]]
		return (top_images, top_indexes,)


class ZNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
				"color": ([False, True], {"default": False}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, images, amount, color):
		tensors = images.clone()

		if amount > 0.0:
			noise = torch.randn(tensors.shape[0], tensors.shape[1], tensors.shape[2], 3 if color else 1)
			tensors += noise * amount

		return (tensors,)


class ZPrompt:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"positive": ("STRING", {"default": "", "multiline": True}),
				"negative": ("STRING", {"default": "", "multiline": True}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("PROMPT",)

	def process(self, batch_size, positive, negative):
		text = T5EncoderModel.from_pretrained(
			"DeepFloyd/IF-I-XL-v1.0",
			subfolder = "text_encoder",
			variant = "fp16",
			device_map = "auto",
			load_in_8bit = True,
			torch_dtype = torch.float16,
		)

		model = DiffusionPipeline.from_pretrained(
			"DeepFloyd/IF-I-XL-v1.0",
			requires_safety_checker = False,
			safety_checker = None,
			text_encoder = text,
			unet = None,
			watermarker = None,
		)

		positive, negative = model.encode_prompt(prompt = positive, negative_prompt = negative, num_images_per_prompt = batch_size)
		del model, text
		gc.collect()
		return ((positive, negative),)


class ZRepeat:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latents": ("LATENT",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, latents, batch_size):
		tensors = latents["samples"]

		if batch_size > 1:
			tensors = tensors.repeat(batch_size, 1, 1, 1)

		return ({"samples": tensors},)


class ZResize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
				"mode": (["area", "bicubic", "bilinear", "nearest", "nearest-exact"], {"default": "nearest-exact"}),
				"crop": ([False, True], {"default": True}),
			},
			"optional": {
				"images": ("IMAGE",),
				"latents": ("LATENT",),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, scale, mode, crop, images = None, latents = None):
		if scale != 1.0:
			if images is not None:
				tensors = images.permute(0, 3, 1, 2)
				tensors = common_upscale(tensors, int(tensors.shape[3] * scale // 1), int(tensors.shape[2] * scale // 1), mode, "center" if crop else None)
				tensors = tensors.permute(0, 2, 3, 1).cpu()
				return (tensors, latents,)
			elif latents is not None:
				tensors = latents["samples"]
				tensors = common_upscale(tensors, int(tensors.shape[3] * scale // 1), int(tensors.shape[2] * scale // 1), mode, "center" if crop else None)
				return (images, {"samples": tensors},)
			else:
				raise ValueError("Invalid input.")
		return (images, latents,)


class ZSelect:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latents": ("LATENT",),
				"list": ("LIST",),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, latents, list):
		samples = latents["samples"]
		tensors = []

		for index in list:
			index = min(samples.shape[0] - 1, index)
			tensor = samples[index:index + 1].clone()
			tensors.append(tensor)

		tensors = torch.cat(tensors)
		return ({"samples": tensors},)


class ZShare:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"output": ("STRING", {"default": "D:\Downloads"}),
				"prefix": ("STRING", {"default": "share"}),
			}
		}

	CATEGORY = "Zuellni"
	COUNTER = 1
	FUNCTION = "process"
	OUTPUT_NODE = True
	RETURN_TYPES = ()

	def process(self, images, output, prefix):
		for image in images:
			image = 255.0 * image.cpu().numpy()
			image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
			image.save(Path(output) / f"{prefix}_{ZShare.COUNTER:05}.png", optimize = True)
			ZShare.COUNTER += 1

		return (None,)


class ZStage1:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"prompts": ("PROMPT",),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, prompts, seed, steps, cfg):
		progress = ProgressBar(steps)

		def callback(step, timestep, latents):
			progress.update_absolute(step)

		model = DiffusionPipeline.from_pretrained(
			"DeepFloyd/IF-I-XL-v1.0",
			requires_safety_checker = False,
			safety_checker = None,
			text_encoder = None,
			watermarker = None,
			variant = "fp16",
			torch_dtype = torch.float16,
		)

		model.to(get_torch_device())
		model.unet.to(memory_format = torch.channels_last)

		tensors = model(
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			prompt_embeds = prompts[0],
			negative_prompt_embeds = prompts[1],
			output_type = "pt",
			callback = callback,
		).images

		tensors = tensors.permute(0, 2, 3, 1).cpu()
		return (tensors,)


class ZStage2:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"prompts": ("PROMPT",),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, images, prompts, seed, steps, cfg):
		tensors = images.permute(0, 3, 1, 2)
		progress = ProgressBar(steps)

		def callback(step, timestep, latents):
			progress.update_absolute(step)

		model = DiffusionPipeline.from_pretrained(
			"DeepFloyd/IF-II-L-v1.0",
			requires_safety_checker = False,
			safety_checker = None,
			text_encoder = None,
			watermarker = None,
			variant = "fp16",
			torch_dtype = torch.float16,
		)

		model.to(get_torch_device())
		model.unet.to(memory_format = torch.channels_last)

		tensors = model(
			image = tensors,
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			prompt_embeds = prompts[0],
			negative_prompt_embeds = prompts[1],
			output_type = "pt",
			callback = callback,
		).images

		tensors = tensors.permute(0, 2, 3, 1).cpu()
		return (tensors,)


class ZUpscale:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"tile": ([False, True], {"default": True}),
				"tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
				"positive": ("STRING", {"default": "", "multiline": True}),
				"negative": ("STRING", {"default": "", "multiline": True}),
			}
		}

	CATEGORY = "Zuellni"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, images, tile_size, tile, seed, steps, cfg, positive, negative):
		tensors = images.permute(0, 3, 1, 2)
		batch_size = tensors.shape[0]
		progress = ProgressBar(steps)

		def callback(step, timestep, latents):
			progress.update_absolute(step)

		if batch_size > 1:
			positive = [positive] * batch_size
			negative = [negative] * batch_size

		model = DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-x4-upscaler",
			requires_safety_checker = False,
			safety_checker = None,
			watermarker = None,
			torch_dtype = torch.float16,
		)

		model.to(get_torch_device())
		model.unet.to(memory_format = torch.channels_last)

		if xformers_enabled():
			model.enable_xformers_memory_efficient_attention()

		if tile:
			model.vae.config.sample_size = tile_size
			model.vae.enable_tiling()

		tensors = model(
			image = tensors,
			generator = torch.manual_seed(seed),
			num_inference_steps = steps,
			guidance_scale = cfg,
			prompt = positive,
			negative_prompt = negative,
			output_type = "pt",
			callback = callback,
		).images

		tensors = tensors.permute(0, 2, 3, 1).cpu()
		return (tensors,)


NODE_CLASS_MAPPINGS = {
	"ZDecode": ZDecode,
	"ZEncode": ZEncode,
	"ZFilter": ZFilter,
	"ZNoise": ZNoise,
	"ZPrompt": ZPrompt,
	"ZRepeat": ZRepeat,
	"ZResize": ZResize,
	"ZSelect": ZSelect,
	"ZShare": ZShare,
	"ZStage1": ZStage1,
	"ZStage2": ZStage2,
	"ZUpscale": ZUpscale,
}