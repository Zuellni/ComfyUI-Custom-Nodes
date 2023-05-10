from folder_paths import get_input_directory, get_output_directory
from comfy.model_management import get_torch_device

import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch

from transformers import pipeline
from pathlib import Path
from PIL import Image
import numpy as np


class AestheticLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"aesthetic": ([False, True], {"default": True}),
				"waifu": ([False, True], {"default": True}),
			},
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("AE_MODEL",)

	def process(self, aesthetic, waifu):
		if aesthetic:
			aesthetic = pipeline("image-classification", f"cafeai/cafe_aesthetic", device = get_torch_device())

		if waifu:
			waifu = pipeline("image-classification", f"cafeai/cafe_waifu", device = get_torch_device())

		return ({"aesthetic": aesthetic, "waifu": waifu},)


class AestheticFilter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("AE_MODEL",),
				"images": ("IMAGE",),
				"count": ("INT", {"default": 1, "min": 0, "max": 64}),
			},
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LIST",)

	def process(self, model, images, count):
		if count == 0:
			return (None, None,)

		aesthetic = model["aesthetic"]
		waifu = model["waifu"]

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


class AestheticSelect:
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


class LoadFolder:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"input_dir": ("STRING", {"default": get_input_directory()}),
				"file_type": (["gif", "jpg", "png"], {"default": "png"}),
			},
		}

	CATEGORY = "Zuellni/Image"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, input_dir, file_type):
		input_dir = Path(input_dir)
		min_height = None
		min_width = None
		images = []

		for image in list(input_dir.glob(f"*.{file_type}")):
			image = Image.open(image)
			image = TF.to_tensor(image)
			image = image[:3, :, :]
			min_height = min(min_height, image.shape[1]) if min_height else image.shape[1]
			min_width = min(min_width, image.shape[2]) if min_width else image.shape[2]
			images.append(image)

		if not images:
			return (None,)

		if len(images) > 1:
			min_dim = min(min_height, min_width)
			images = [TF.resize(image, min_dim) for image in images]
			images = [TF.center_crop(image, (min_height, min_width)) for image in images]

		images = torch.stack(images)
		images = images.permute(0, 2, 3, 1)
		return (images,)


class ShareImage:
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
			image.save(output_dir / f"{prefix}_{ShareImage.COUNTER:05}.png", optimize = True)
			ShareImage.COUNTER += 1

		return (None,)


class LatentDecoder:
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


class LatentEncoder:
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

		if batch_size > 1:
			image = image.repeat(batch_size, 1, 1, 1)

		return ({"samples": vae.encode_tiled(image) if tile else vae.encode(image)},)


class MultiRepeat:
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


class MultiNoise:
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
		if strength > 0.0:
			if image is not None:
				noise = torch.randn(image.shape)
				image = image + noise * strength

			if latent is not None:
				latent = latent["samples"]
				noise = torch.randn(latent.shape)
				latent = {"samples": latent + noise * strength}

		return (image, latent,)


class MultiResize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.05}),
				"mode": (["area", "bicubic", "bilinear", "nearest", "nearest-exact"], {"default": "nearest-exact"}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			},
		}

	CATEGORY = "Zuellni/Multi"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, scale, mode, image = None, latent = None):
		if scale != 1.0:
			if image is not None:
				image = image.permute(0, 3, 1, 2)
				image = F.interpolate(image, mode = mode, scale_factor = scale)
				image = image.permute(0, 2, 3, 1)

			if latent is not None:
				latent = latent["samples"]
				latent = F.interpolate(latent, mode = mode, scale_factor = scale)
				latent = {"samples": latent}

		return (image, latent,)