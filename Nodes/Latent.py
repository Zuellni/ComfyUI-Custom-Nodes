from torchvision.transforms import functional as TF


class Decoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "vae": ("VAE",),
                "tile": ([False, True], {"default": False}),
            },
        }

    CATEGORY = "Zuellni/Latent"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("IMAGE",)

    def process(self, latents, vae, tile):
        latents = latents["samples"]
        return (vae.decode_tiled(latents) if tile else vae.decode(latents),)


class Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "tile": ([False, True], {"default": False}),
            },
        }

    CATEGORY = "Zuellni/Latent"
    FUNCTION = "process"
    RETURN_NAMES = ("LATENTS",)
    RETURN_TYPES = ("LATENT",)

    def process(self, images, vae, tile, batch_size):
        images = images.permute(0, 3, 1, 2)
        height, width = images.shape[2:]
        images = TF.center_crop(images, (height // 8 * 8, width // 8 * 8))
        images = images.permute(0, 2, 3, 1)

        if batch_size > 1:
            images = images.repeat(batch_size, 1, 1, 1)

        return ({"samples": vae.encode_tiled(images) if tile else vae.encode(images)},)
