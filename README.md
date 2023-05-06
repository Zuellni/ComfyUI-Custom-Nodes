# Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory and install the requirements.
```
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes custom_nodes\Zuellni
pip install -r custom_nodes\Zuellni\requirements.txt
```
## Custom Nodes
A bunch of custom/modded nodes. Most work with multiple images/latents. All required models are currently downloaded to the huggingface `.cache` directory.
### Aesthetic Filter
Returns `x` best images and a `list` of their indexes based on [cafe_aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)/[cafe_waifu](https://huggingface.co/cafeai/cafe_waifu) scoring. If both are `False` then it acts like `LatentFromBatch` and returns 1 image with 1-based index.
### Aesthetic Select
Takes `latents` and a `list` of indexes from `Aesthetic Filter` and returns only the selected `latents`.
### Share Image
Saves images without metadata in specified directory. Counter resets on restart. Good for sharing without having to remove prompts manually.
### VAE Decode
Combines `VAEDecode` and `VAEDecodeTiled`. Probably not necessary since `VAEDecodeTiled` is now used on error but just here for the sake of completeness.
### VAE Encode
As above, but adds `batch_size`. Allows for loading 1 image and denoising it `x` times without having to create multiple `KSampler` nodes.
### Multi Noise
Adds random black and white/color noise to image/latent.
### Multi Repeat
Allows for repeating image/latent `x` times, similar to `VAE Encode`.
### Multi Resize
Similar to `LatentUpscale` but takes `scale` instead of width/height. Works with images and latents.
## DeepFloyd Nodes
A poor man's implementation of [IF](https://huggingface.co/docs/diffusers/api/pipelines/if). All the stages with text encoder unloading enabled currently <ins>require more than 8GB of VRAM</ins>. 13GB should be enough to run without unloading the text encoder. 12GB isn't.
### IF Encode
Encodes positive/negative prompts for use with `IF Stage I` and `IF Stage II`. Higher `batch_size` results in more images. <ins>Requires more than 8GB of VRAM</ins>, as well as [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to load with 8-bit precision. CPU offloading currently doesn't seem to work. Setting `unload` to `True` removes the model from memory after it's finished. Prompts can be reused without having to reload it.
### IF Stage I
Takes the prompt embeds from `IF Encode`, as well as `seed`, `steps`, and `cfg`, and returns `64x64px` images which can be used with `IF Stage II` or other nodes.
### IF Stage II
As above, but also takes `Stage I` or other images. Returns `256x256px` images which can be used with `IF Stage III` or other nodes such as upscalers or samplers. Images larger than `64x64px` will still result in `256x256px` output.
### IF Stage III
Upscales `Stage II` or other images `4 times`, resulting in `1024x1024px` images for `Stage II`. Doesn't work with `IF Encode` embeds, has its own encoder accepting `string` prompts instead. Uses `xformers` to reduce memory by default if enabled. Setting `tile` to `True` additionally allows for upscaling larger images than normally possible, around `768x768px` base with 12GB of VRAM. Tile size can be a bit of a hit or miss, the seams are often quite visible.
