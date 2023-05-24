import json
import requests


class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("STRING", {"default": "Example"}),
                "template": ("STRING", {"default": "WizardLM"}),
                "api": ("STRING", {"default": "http://localhost:5000/api/v1/chat"}),
                "min_tokens": ("INT", {"default": 32, "min": 1, "max": 2048}),
                "max_tokens": ("INT", {"default": 64, "min": 1, "max": 2048}),
                "penalty": (
                    "FLOAT",
                    {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_NAMES = ("PARAMS",)
    RETURN_TYPES = ("DICT",)

    def process(
        self,
        character,
        template,
        api,
        min_tokens,
        max_tokens,
        penalty,
        temperature,
        top_k,
        top_p,
    ):
        return (
            {
                "api": api,
                "request": {
                    "character": character,
                    "instruction_template": template,
                    "min_length": min_tokens,
                    "max_new_tokens": max_tokens,
                    "repetition_penalty": penalty,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "mode": "chat",
                    "history": {
                        "internal": [],
                        "visible": [],
                    },
                },
            },
        )


class Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "params": ("DICT",),
            },
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING",)

    def process(self, text, seed, params):
        params["request"]["user_input"] = text
        params["request"]["seed"] = seed
        response = requests.post(params["api"], json=params["request"])
        return (response.json()["results"][0]["history"]["visible"][-1][1],)


class Condition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"default": ""}),
                "condition": (
                    [
                        "==",
                        "!=",
                        "<",
                        "<=",
                        ">",
                        ">=",
                        "contains",
                        "starts with",
                        "ends with",
                    ],
                    {"default": "=="},
                ),
                "b": ("STRING", {"default": ""}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
            },
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_NAMES = ("IMAGES", "LATENTS", "RESULT")
    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")

    def process(self, a, condition, b, images=None, latents=None):
        result = False
        fa = a
        fb = b

        try:
            fa = float(a)
            fb = float(b)
        except:
            pass

        operations = {
            "==": lambda: fa == fb,
            "!=": lambda: fa != fb,
            "<": lambda: fa < fb,
            "<=": lambda: fa <= fb,
            ">": lambda: fa > fb,
            ">=": lambda: fa >= fb,
            "contains": lambda: b in a,
            "starts with": lambda: a.startswith(b),
            "ends with": lambda: a.endswith(b),
        }

        result = operations.get(condition, lambda: False)()

        if result:
            return (images, latents, "true")

        return (None, None, "false")


class Format:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "var_1": ("STRING", {"default": ""}),
                "var_2": ("STRING", {"default": ""}),
                "var_3": ("STRING", {"default": ""}),
                "var_4": ("STRING", {"default": ""}),
                "var_5": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING",)

    def process(self, text, **vars):
        for key, value in vars.items():
            text = text.replace(key, value)

        return (text,)


class Print:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": ""}),
                "text": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def process(self, prefix, text):
        prefix = f"[\033[94m{prefix}\033[0m]: " if prefix else ""
        print(f"{prefix}{text}")
        return (None,)
