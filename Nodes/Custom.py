from comfy.model_management import get_torch_device
from folder_paths import get_output_directory
from nodes import VAEEncode

from transformers import pipeline
from pathlib import Path
from PIL import Image
import numpy as np
import torch


class Load:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (["aesthetic", "waifu"], {"default": "aesthetic"}),
			},
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("MODEL",)

	def process(self, model):
		return (pipeline("image-classification", f"cafeai/cafe_{model}", device = get_torch_device()),)


class Filter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"count": ("INT", {"default": 1, "min": 1, "max": 64}),
			},
			"optional": {
				"aesthetic": ("MODEL",),
				"waifu": ("MODEL",),
			},
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LIST",)

	def process(self, images, count, aesthetic = None, waifu = None):
		if not aesthetic and not waifu:
			return (images[count - 1].unsqueeze(0), [count - 1],)

		scores = {}

		for index, image in enumerate(images):
			image = 255.0 * image.cpu().numpy()
			image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
			score = 0.0

			if aesthetic:
				data = aesthetic(image, top_k = 2)

				for item in data:
					score += item["score"] * (1.0 if item["label"] == "aesthetic" else -1.0)

			if waifu:
				data = waifu(image, top_k = 2)

				for item in data:
					score += item["score"] * (1.0 if item["label"] == "waifu" else -1.0)

			scores[index] = score

		scores = sorted(scores.items(), key = lambda x: x[1], reverse = True)
		images = [images[score[0]] for score in scores[:count]]
		images = torch.stack(images)
		list = [score[0] for score in scores[:count]]
		return (images, list,)


class Select:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent": ("LATENT",),
				"list": ("LIST",),
			},
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, latent, list):
		latent = latent["samples"]
		samples = []

		for index in list:
			index = min(latent.shape[0] - 1, index)
			sample = latent[index:index + 1].clone()
			samples.append(sample)

		samples = torch.cat(samples)
		return ({"samples": samples},)


class Save:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"output_dir": ("STRING", {"default": get_output_directory()}),
				"prefix": ("STRING", {"default": "clean"}),
			},
		}

	CATEGORY = "Zuellni/Image"
	COUNTER = 1
	FUNCTION = "process"
	OUTPUT_NODE = True
	RETURN_TYPES = ()

	def process(self, images, output_dir, prefix):
		output_dir = Path(output_dir)
		output_dir.mkdir(parents = True, exist_ok = True)

		for image in images:
			image = 255.0 * image.cpu().numpy()
			image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
			image.save(output_dir / f"{prefix}_{Save.COUNTER:05}.png", optimize = True)
			Save.COUNTER += 1

		return (None,)


class Decode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent": ("LATENT",),
				"vae": ("VAE",),
				"tile": ([False, True], {"default": False}),
			},
		}

	CATEGORY = "Zuellni/Latent"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, latent, vae, tile):
		return (vae.decode_tiled(latent["samples"]) if tile else vae.decode(latent["samples"]),)


class Encode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"vae": ("VAE",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"tile": ([False, True], {"default": False}),
			},
		}

	CATEGORY = "Zuellni/Latent"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, image, vae, tile, batch_size):
		image = image[:, :, :, :3]
		image = VAEEncode.vae_encode_crop_pixels(image)

		if batch_size > 1:
			image = image.repeat(batch_size, 1, 1, 1)

		return ({"samples": vae.encode_tiled(image) if tile else vae.encode(image)},)


class Repeat:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			},
		}

	CATEGORY = "Zuellni/Multi"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, batch_size, image = None, latent = None):
		if batch_size > 1:
			if image is not None:
				image = image.repeat(batch_size, 1, 1, 1)

			if latent is not None:
				latent = latent["samples"]
				latent = latent.repeat(batch_size, 1, 1, 1)
				latent = {"samples": latent}

		return (image, latent,)


class Noise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			},
		}

	CATEGORY = "Zuellni/Multi"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, strength, image = None, latent = None):
		if amount > 0.0:
			if image is not None:
				noise = torch.randn(image.shape)
				image = image + noise * amount

			if latent is not None:
				latent = latent["samples"]
				noise = torch.randn(latent.shape)
				latent = {"samples": latent + noise * amount}

		return (image, latent,)


class Resize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.05}),
				"mode": (["area", "bicubic", "bilinear", "nearest", "nearest-exact"], {"default": "nearest-exact"}),
				"crop": ([False, True], {"default": False}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			},
		}

	CATEGORY = "Zuellni/Multi"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, scale, mode, crop, image = None, latent = None):
		if scale != 1.0:
			if image is not None:
				image = image.permute(0, 3, 1, 2)
				image = torch.nn.functional.interpolate(image, mode = mode, scale_factor = scale)
				image = image.permute(0, 2, 3, 1)

				if crop:
					image = VAEEncode.vae_encode_crop_pixels(image)

			if latent is not None:
				latent = latent["samples"]
				latent = torch.nn.functional.interpolate(latent, mode = mode, scale_factor = scale)

				if crop:
					latent = latent.permute(0, 3, 2, 1)
					latent = VAEEncode.vae_encode_crop_pixels(latent)
					latent = latent.permute(0, 3, 2, 1)

				latent = {"samples": latent}

		return (image, latent,)