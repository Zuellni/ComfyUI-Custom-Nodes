from folder_paths import get_input_directory, get_output_directory
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import numpy as np
import torch


class Batch:
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
	RETURN_NAMES = ("IMAGES",)
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


class Share:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"output_dir": ("STRING", {"default": get_output_directory()}),
				"prefix": ("STRING", {"default": ""}),
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

		if prefix:
			prefix = f"{prefix}_"

		for image in images:
			image = 255.0 * image.cpu().numpy()
			image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
			image.save(output_dir / f"{prefix}{Save.COUNTER:05}.png")
			Save.COUNTER += 1

		return (None,)