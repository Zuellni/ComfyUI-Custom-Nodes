# Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory:
```
git clone https://github.com/Zuellni/ComfyUI-Custom-Nodes custom_nodes\Zuellni
```

A `config.json` file will be created on first run in the extension's directory.  
Requirements should be installed automatically but if that doesn't happen you can install them with:
```
pip install -r custom_nodes\Zuellni\requirements.txt
```
You can skip the installation if you don't wish to use the `IF` nodes.  
Run ComfyUI once, wait till the config file gets created, then quit and set `IF` to `false` under `Load Nodes` in `config.json`.

To enable automatic updates set `Update Repository` to `true` in the config. You can also update with:
```
git -C custom_nodes\Zuellni pull
```


## Aesthetic Nodes
Name | Description
:--- | :---
Aesthetic&nbsp;Loader | Loads models for use with `Aesthetic Select`.
Aesthetic&nbsp;Select | Returns `count` best tensors based on [aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)/[style](https://huggingface.co/cafeai/cafe_style)/[waifu](https://huggingface.co/cafeai/cafe_waifu)/[age](https://huggingface.co/nateraw/vit-age-classifier) classifiers. If no models are selected then acts like `LatentFromBatch` and returns a single tensor with 1-based index. Setting `count` to 0 stops processing for connected nodes.

## IF Nodes
A poor man's implementation of [DeepFloyd IF](https://huggingface.co/DeepFloyd). Models will be downloaded automatically, but you will have to agree to the terms of use on the site, create an access token, and [log in](https://huggingface.co/docs/huggingface_hub/quick-start#login) with it.

Name | Description
:--- | :---
IF&nbsp;Load | Loads models for use with other `IF` nodes. `Device` can be used to move the models to specific devices, eg `cpu`, `cuda`, `cuda:0`, `cuda:1`. Leaving it empty enables offloading.
IF&nbsp;Encode | Encodes prompts for use with `IF Stage I` and `IF Stage II`.
IF&nbsp;Stage&nbsp;I | Takes the prompt embeds from `IF Encode` and returns images which can be used with `IF Stage II` or other nodes.
IF&nbsp;Stage&nbsp;II | As above, but also takes `Stage I` or other images and upscales them x4.
IF&nbsp;Stage&nbsp;III | Upscales `Stage II` or other images using [Stable Diffusion x4 upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). Doesn't work with `IF Encoder` embeds, has its own encoder accepting `string` prompts instead. Setting `tile_size` allows for upscaling larger images than normally possible.

## Image Nodes
Name | Description
:--- | :---
Image&nbsp;Batch | Loads all images in a specified directory, including animated gifs, as a batch. The images will be cropped/resized if their dimensions aren't equal.
Image&nbsp;Saver | Saves images without metadata in a specified directory. Allows saving a batch of images as a grid or animated gif as well.

## Multi Nodes
Nodes that work with multiple types of tensors - images, latents, and masks.

Name | Description
:--- | :---
Multi&nbsp;Crop | Center crops/pads tensors to specified dimensions.
Multi&nbsp;Noise | Adds random noise to tensors.
Multi&nbsp;Repeat | Allows for repeating tensors `batch_size` times.
Multi&nbsp;Resize | Similar to `LatentUpscale` but uses `scale` instead of width/height to resize tensors.

## Text Nodes
Experimental nodes utilizing [text-generation-webui](https://github.com/oobabooga/text-generation-webui) to generate and manipulate prompts. Webui needs to be running with `--api` and a preloaded model since it's not possible to change it through the API currently.

Example startup command for [WizardLM](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GPTQ):
```
python server.py --api --model llama-7b-4bit-128g-wizard
```

Name | Description
:--- | :---
Text&nbsp;Loader | Used as initializer for `Text Prompt` so you don't have to specify the same params multiple times. Set your API endpoint with `api`, instruction template for your loaded model with `template` (might not be necessary), and the character used to generate prompts with `character` (format depends on your needs).
Text&nbsp;Prompt | Queries the API with params from `Text Loader` and returns a `string` you can use as input for other nodes like `CLIP Text Encode`.
Text&nbsp;Condition | Returns input tensors and `true` if variables match some condition, `false` otherwise. Will interrupt the generation if condition is not met and `interrupt` set to `true`.
Text&nbsp;Format | Joins input `string` with multiple variables and returns a single output `string`. Specifying `var_1-5` somewhere in the input field will replace it with said variable's value.
Text&nbsp;Print | Prints `string` input to console for debugging purposes (or just to see what sort of prompt your LLM came up with).
