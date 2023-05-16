# Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory. A `config.json` file should be created on first run and requirements installed automatically.
```
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes custom_nodes\Zuellni
```
To update change the relevant setting in `config.json` or execute the following command:
```
git -C custom_nodes\Zuellni pull
```
## Aesthetic Nodes
Name | Description
:--- | :---
Aesthetic&nbsp;Loader | Loads models for use with `Aesthetic Select`.
Aesthetic&nbsp;Select | Returns `count` best images and latents based on [cafe_aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)/[cafe_waifu](https://huggingface.co/cafeai/cafe_waifu) scoring. If no models are loaded then acts like `LatentFromBatch` and returns 1 image/latent with 1-based index. Setting `count` to 0 stops processing for connected nodes.
## IF Nodes
A poor man's implementation of [DeepFloyd IF](https://huggingface.co/DeepFloyd). Text encoder requires more than 8GB of VRAM. To download the models you will have to agree to the terms of use on the site, create an access token and [log in](https://huggingface.co/docs/huggingface_hub/quick-start#login) with it.
Name | Description
:--- | :---
IF&nbsp;Loader | Loads models for use with other `IF` nodes. `Device` can be used to move the models to specific devices, eg `cuda:0`, `cuda:1`. Leaving it empty enables cpu offloading.
IF&nbsp;Encode | Encodes prompts for use with `IF Stage I` and `IF Stage II`. Setting `unload` to `True` removes the model from memory. Prompts can be reused without having to reload the encoder.
IF&nbsp;Stage&nbsp;I | Takes the prompt embeds from `IF Encode` and returns images which can be used with `IF Stage II` or other nodes.
IF&nbsp;Stage&nbsp;II | As above, but also takes `Stage I` or other images and upscales them x4.
IF&nbsp;Stage&nbsp;III | Upscales `Stage II` or other images using [Stable Diffusion x4 upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). Doesn't work with `IF Encode` embeds, has its own encoder accepting `string` prompts instead. Setting `tile` to `True` allows for upscaling larger images than normally possible.
## Other Nodes
Name | Description
:--- | :---
Latent&nbsp;Decode | Combines `VAEDecode` and `VAEDecodeTiled`. Probably not necessary since `VAEDecodeTiled` is now used on error, but just here for the sake of completeness.
Latent&nbsp;Encode | As above, but adds `batch_size`. Allows loading 1 image and denoising it `batch_size` times without having to create multiple sampler nodes.
Multi&nbsp;Crop | Center crops images/latents to specified dimensions.
Multi&nbsp;Noise | Adds random noise to images/latents.
Multi&nbsp;Repeat | Allows for repeating images/latents `batch_size` times, similar to `Latent Encode`.
Multi&nbsp;Resize | Similar to `LatentUpscale` but uses `scale` instead of width/height. Works with both images and latents.
Image&nbsp;Folder | Loads all images in a specified directory. The images will be cropped/resized if their dimensions aren't equal.
Image&nbsp;Share | Saves images without metadata in a specified directory. Counter resets on restart. Useful for sharing images without having to remove prompts manually.
