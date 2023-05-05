from comfy.model_management import get_torch_device
from transformers import pipeline
from pathlib import Path
from PIL import Image
import numpy as np
import torch


def resize(tensor):
	h = tensor.shape[1] // 8 * 8
	w = tensor.shape[2] // 8 * 8

	if tensor.shape[1] != h or tensor.shape[2] != w:
		h_offset = tensor.shape[1] % 8 // 2
		w_offset = tensor.shape[2] % 8 // 2
		tensor = tensor[:, h_offset:h + h_offset, w_offset:w + w_offset, :]

	return tensor


class Filter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"count": ("INT", {"default": 1, "min": 1, "max": 64}),
				"aesthetic": ([False, True], {"default": True}),
				"waifu": ([False, True], {"default": True}),
			}
		}

	CATEGORY = "Zuellni/Aesthetic"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LIST",)

	def process(self, images, count, aesthetic, waifu):
		if not aesthetic and not waifu:
			return (images[count - 1].unsqueeze(0), [count - 1],)

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
		top_images = [images[score[0]] for score in sorted_scores[:count]]
		top_images = torch.stack(top_images)
		top_list = [score[0] for score in sorted_scores[:count]]
		return (top_images, top_list,)


class Select:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent": ("LATENT",),
				"list": ("LIST",),
			}
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


class Share:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"output": ("STRING", {"default": "share"}),
				"prefix": ("STRING", {"default": "share"}),
			}
		}

	CATEGORY = "Zuellni/Image"
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


class VAEDecode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent": ("LATENT",),
				"vae": ("VAE",),
				"tile": ([False, True], {"default": False}),
			}
		}

	CATEGORY = "Zuellni/Latent"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE",)

	def process(self, latent, vae, tile):
		return (vae.decode_tiled(latent["samples"]) if tile else vae.decode(latent["samples"]),)


class VAEEncode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"vae": ("VAE",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"tile": ([False, True], {"default": False}),
			}
		}

	CATEGORY = "Zuellni/Latent"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, image, vae, tile, batch_size):
		image = image[:, :, :, :3]
		image = resize(image)

		if batch_size > 1:
			image = image.repeat(batch_size, 1, 1, 1)

		return ({"samples": vae.encode_tiled(image) if tile else vae.encode(image)},)


class Repeat:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent": ("LATENT",),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
			}
		}

	CATEGORY = "Zuellni/Latent"
	FUNCTION = "process"
	RETURN_TYPES = ("LATENT",)

	def process(self, samples, batch_size):
		latent = latent["samples"]

		if batch_size > 1:
			latent = latent.repeat(batch_size, 1, 1, 1)

		return ({"samples": latent},)


class Noise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
				"color": ([False, True], {"default": False}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			}
		}

	CATEGORY = "Zuellni/Multi"
	FUNCTION = "process"
	RETURN_TYPES = ("IMAGE", "LATENT",)

	def process(self, amount, color, image = None, latent = None):
		if amount > 0.0:
			if image is not None:
				noise = torch.randn(image.shape[0], image.shape[1], image.shape[2], 3 if color else 1)
				image = image + noise * amount

			if latent is not None:
				latent = latent["samples"]
				noise = torch.randn(latent.shape[0], latent.shape[1], latent.shape[2], 3 if color else 1)
				latent = {"samples": latent + noise * amount}

		return (image, latent,)


class Resize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.05}),
				"mode": (["area", "bicubic", "bilinear", "nearest", "nearest-exact"], {"default": "nearest-exact"}),
				"crop": ([False, True], {"default": True}),
			},
			"optional": {
				"image": ("IMAGE",),
				"latent": ("LATENT",),
			}
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
					image = resize(image)

			if latent is not None:
				latent = latent["samples"]
				latent = torch.nn.functional.interpolate(latent, mode = mode, scale_factor = scale)

				if crop:
					latent = latent.permute(0, 3, 2, 1)
					latent = resize(latent)
					latent = latent.permute(0, 3, 2, 1)

				latent = {"samples": latent}

		return (image, latent,)