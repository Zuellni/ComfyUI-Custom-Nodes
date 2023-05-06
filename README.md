# Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory and install the requirements.
```
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes custom_nodes\Zuellni
pip install -r custom_nodes\Zuellni\requirements.txt
```
## Custom Nodes
### Aesthetic Filter
Returns `x` best images and a `list` of their indexes based on [cafe_aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)/[cafe_waifu](https://huggingface.co/cafeai/cafe_waifu) scoring. If both are `False` then it acts like `LatentFromBatch` and returns 1 image with 1-based index.
### Aesthetic Select
Takes `latents` and a `list` of indexes from `Aesthetic Filter` and returns only the selected `latents`.
### Share Image
Saves images without metadata in specified directory. Counter resets on restart. Good for sharing without having to remove prompts manually.
### VAE Decode
Combines `VAEDecode` and `VAEDecodeTiled`. Probably not necessary since `VAEDecodeTiled` is now used on error but just here for the sake of completeness.
### VAE Encode
As above but adds `batch_size`. Allows for loading 1 image and denoising it `x` times without having to create multiple `KSampler` nodes.
### Multi Noise
Adds random black and white/color noise to image/latent.
### Multi Repeat
Allows for repeating image/latent `x` times, similar to `VAE Encode`.
### Multi Resize
Similar to `LatentUpscale` but takes `scale` instead of width/height. Works with images and latents.
## DeepFloyd Nodes
//TODO
