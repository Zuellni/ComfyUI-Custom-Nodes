from pathlib import Path
import importlib
import inspect
import json


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

config = {
	"Settings": {
		"Update": False,
		"Quiet Update": False,
		"Suppress Warnings": True,
	},
	"Load Nodes": {
		"Aesthetic": True,
		"IF": True,
		"Image": True,
		"Latent": True,
		"Multi": True,
	},
}

path = Path(__file__)
config_path = path.with_name("config.json")
req_path = path.with_name("requirements.txt")

if config_path.is_file():
	with open(config_path, "r") as f:
		dict = json.load(f)
		
		for key, val in dict.items():
			if key in config:
				for sub_key, sub_val in config[key].items():
					if sub_key in config:
						config[key][sub_key] = sub_val

with open(config_path, "w") as f:
	json.dump(config, f, indent = "\t", separators = (",", ": "))
		
if config["Settings"]["Update"]:
	import subprocess

	quiet = "-q" if config["Settings"]["Quiet Update"] else ""
	subprocess.run(f"git pull {quiet} -C {path}")
	subprocess.run(f"pip install {quiet} --upgrade-strategy only-if-needed -r {req_path}")

if config["Settings"]["Suppress Warnings"]:
	from transformers import logging as transformers_logging
	from diffusers import logging as diffusers_logging
	from warnings import filterwarnings
	import logging

	filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")
	filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")
	filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
	logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
	transformers_logging.set_verbosity_error()
	diffusers_logging.set_verbosity_error()

for key, val in config["Load Nodes"].items():
	if val:
		module = importlib.import_module(f".Nodes.{key}", package = __name__)

		for name, cls in inspect.getmembers(module, inspect.isclass):
			if cls.__module__ == module.__name__:
				node = f"{cls.__module__}.{name}"
				disp = f"{key} {name.replace('_', ' ')}"
				NODE_CLASS_MAPPINGS[node] = cls
				NODE_DISPLAY_NAME_MAPPINGS[node] = disp