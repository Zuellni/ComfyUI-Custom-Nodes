from transformers import logging as transformers_logging
from diffusers import logging as diffusers_logging
from warnings import filterwarnings
import logging

from .Nodes import Custom, DeepFloyd


transformers_logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()
logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")
filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")


NODE_CLASS_MAPPINGS = {
	# Aesthetic
	"Filter": Custom.Filter,
	"Select": Custom.Select,

	# Image
	"Save": Custom.Save,

	# Latent
	"VAE Decode": Custom.Decode,
	"VAE Encode": Custom.Encode,
	"Repeat": Custom.Repeat,

	# Multi
	"Noise": Custom.Noise,
	"Resize": Custom.Resize,

	# DeepFloyd
	"Encode": DeepFloyd.Encode,
	"Stage I": DeepFloyd.StageI,
	"Stage II": DeepFloyd.StageII,
	"Stage III": DeepFloyd.StageIII,
}