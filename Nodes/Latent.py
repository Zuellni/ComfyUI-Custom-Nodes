class Decoder:
    @classmethod
    def INPUT_TYPES(s):
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
    def INPUT_TYPES(s):
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
        images = images[:, :, :, :3]

        if batch_size > 1:
            images = images.repeat(batch_size, 1, 1, 1)

        return ({"samples": vae.encode_tiled(images) if tile else vae.encode(images)},)
