from transformers import logging as transformers_logging
from diffusers import logging as diffusers_logging
from warnings import filterwarnings
import logging

from .Nodes.Custom import *
from .Nodes.DeepFloyd import *


transformers_logging.set_verbosity_error()
diffusers_logging.set_verbosity_error()
logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")
filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")


NODE_CLASS_MAPPINGS = {
	# Aesthetic
	"Filter": Nodes.Custom.Filter,
	"Select": Nodes.Custom.Select,

	# Image
	"Save": Nodes.Custom.Save,

	# Latent
	"VAE Decode": Nodes.Custom.Decode,
	"VAE Encode": Nodes.Custom.Encode,
	"Repeat": Nodes.Custom.Repeat,

	# Multi
	"Noise": Nodes.Custom.Noise,
	"Resize": Nodes.Custom.Resize,

	# DeepFloyd
	"Encode": Nodes.DeepFloyd.Encode,
	"Stage I": Nodes.DeepFloyd.StageI,
	"Stage II": Nodes.DeepFloyd.StageII,
	"Stage III": Nodes.DeepFloyd.StageIII,
}