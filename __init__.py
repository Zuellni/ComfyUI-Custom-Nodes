from transformers import logging as transformers_logging
from diffusers import logging as diffusers_logging
from warnings import filterwarnings
import logging

from .if_nodes import *
from .nodes import *


transformers_logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()
logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")
filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")


NODE_CLASS_MAPPINGS = {
	# Aesthetic
	"Filter": nodes.Filter,
	"Select": nodes.Select,

	# Image
	"Share": nodes.Share,

	# Latent
	"VAE Decode": nodes.VAEDecode,
	"VAE Encode": nodes.VAEEncode,
	"Repeat": nodes.Repeat,

	# Multi
	"Noise": nodes.Noise,
	"Resize": nodes.Resize,

	# DeepFloyd
	"Encode": if_nodes.Encode,
	"Stage I": if_nodes.StageI,
	"Stage II": if_nodes.StageII,
	"Stage III": if_nodes.StageIII,
}