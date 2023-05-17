from pathlib import Path
import subprocess
import importlib
import inspect
import json

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

config = {
	"Settings": {
		"Install Requirements": False,
		"Update Repository": False,
		"Quiet Update": True,
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
git_path = path.parent
config_path = path.with_name("config.json")
req_path = path.with_name("requirements.txt")
quiet = "-q" if config["Settings"]["Quiet Update"] else ""
first_run = False

if config_path.is_file():
	try:
		with open(config_path, "r") as f:
			try:
				dict = json.load(f)

				for key, val in dict.items():
					if key in config:
						for sub_key, sub_val in dict[key].items():
							if sub_key in config[key]:
								config[key][sub_key] = sub_val
			except:
				print("[\033[94mZuellni\033[0m]: Invalid config. Loading defaults...")
	except:
		print("[\033[94mZuellni\033[0m]: Couldn't open config. Loading defaults...")
else:
	first_run = True

try:
	with open(config_path, "w") as f:
		json.dump(config, f, indent = "\t", separators = (",", ": "))
except:
	print("[\033[94mZuellni\033[0m]: Couldn't save config. Proceeding...")

if config["Settings"]["Update Repository"]:
	print("[\033[94mZuellni\033[0m]: Updating repository...")
	subprocess.run(f"git -C {git_path} pull {quiet}")

if config["Settings"]["Install Requirements"] or first_run:
	print("[\033[94mZuellni\033[0m]: Installing requirements...")
	subprocess.run(f"pip install {quiet} --upgrade-strategy only-if-needed -r {req_path}")

if config["Settings"]["Suppress Warnings"]:
	from transformers import logging as transformers_logging
	from diffusers import logging as diffusers_logging
	from warnings import filterwarnings
	import logging

	filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")
	filterwarnings("ignore", category = UserWarning, message = "You seem to be using the pipelines sequentially on GPU")
	filterwarnings("ignore", category = FutureWarning, message = "The `reduce_labels` parameter is deprecated")

	logging.getLogger("xformers").addFilter(lambda r: "A matching Triton is not available" not in r.getMessage())
	transformers_logging.set_verbosity_error()
	diffusers_logging.set_verbosity_error()

for key, val in config["Load Nodes"].items():
	if val:
		module = importlib.import_module(f".Nodes.{key}", package = __name__)

		for name, cls in inspect.getmembers(module, inspect.isclass):
			if cls.__module__ == module.__name__:
				node = f"Zuellni.{key}.{name}"
				disp = f"{key} {name.replace('_', ' ')}"
				NODE_CLASS_MAPPINGS[node] = cls
				NODE_DISPLAY_NAME_MAPPINGS[node] = disp