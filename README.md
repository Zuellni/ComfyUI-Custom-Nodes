# Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory:
```
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes custom_nodes\Zuellni
```

A `config.json` file will be created on first run in the extension's directory. Requirements should be installed automatically but if that doesn't happen you can install them with:
```
pip install -r custom_nodes\Zuellni\requirements.txt
```

To update set `Update Repository` to `true` in `config.json` or run:
```
git -C custom_nodes\Zuellni pull
```

## Aesthetic Nodes
Name | Description
:--- | :---
Aesthetic&nbsp;Loader | Loads models for use with `Aesthetic Selector`.
Aesthetic&nbsp;Selector | Returns `count` best images/latents based on [aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)/[style](https://huggingface.co/cafeai/cafe_style)/[waifu](https://huggingface.co/cafeai/cafe_waifu)/[age](https://huggingface.co/nateraw/vit-age-classifier) classifiers. If no models are selected then acts like `LatentFromBatch` and returns 1 image/latent with 1-based index. Setting `count` to 0 stops processing for connected nodes.

## IF Nodes
A poor man's implementation of [DeepFloyd IF](https://huggingface.co/DeepFloyd). Models will be downloaded automatically, but you will have to agree to the terms of use on the site, create an access token, and [log in](https://huggingface.co/docs/huggingface_hub/quick-start#login) with it.
Name | Description
:--- | :---
IF&nbsp;Loader | Loads models for use with other `IF` nodes. `Device` can be used to move the models to specific devices, eg `cpu`, `cuda:0`, `cuda:1`. Leaving it empty enables offloading.
IF&nbsp;Encoder | Encodes prompts for use with `IF Stage I` and `IF Stage II`.
IF&nbsp;Stage&nbsp;I | Takes the prompt embeds from `IF Encoder` and returns images which can be used with `IF Stage II` or other nodes.
IF&nbsp;Stage&nbsp;II | As above, but also takes `Stage I` or other images and upscales them x4.
IF&nbsp;Stage&nbsp;III | Upscales `Stage II` or other images using [Stable Diffusion x4 upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). Doesn't work with `IF Encoder` embeds, has its own encoder accepting `string` prompts instead. Setting `tile` to `True` allows for upscaling larger images than normally possible.

## Other Nodes
Name | Description
:--- | :---
Image&nbsp;Loader | Loads all images in a specified directory, including animated gifs, as a batch. The images will be cropped/resized if their dimensions aren't equal.
Image&nbsp;Saver | Saves images without metadata in a specified directory. Allows saving a batch of images as a grid or animated gif.
Latent&nbsp;Decoder | Combines `VAEDecode` and `VAEDecodeTiled`. Probably not necessary since `VAEDecodeTiled` is now used on error, but just here for the sake of completeness.
Latent&nbsp;Encoder | As above, but adds `batch_size`. Allows loading 1 image and denoising it `batch_size` times without having to create multiple sampler nodes.
Multi&nbsp;Crop | Center crops/pads images/latents to specified dimensions.
Multi&nbsp;Noise | Adds random noise to images/latents.
Multi&nbsp;Repeat | Allows for repeating images/latents `batch_size` times, similar to `Latent Encoder`.
Multi&nbsp;Resize | Similar to `LatentUpscale` but uses `scale` instead of width/height to upscale images/latents.
