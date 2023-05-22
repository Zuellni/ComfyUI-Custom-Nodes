import json
import requests


class Gen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True}),
                "character": ("STRING", {"default": "Example"}),
                "template": ("STRING", {"default": "WizardLM"}),
                "api": ("STRING", {"default": "http://localhost:5000/api/v1/chat"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "min_tokens": ("INT", {"default": 32, "min": 1, "max": 2048}),
                "max_tokens": ("INT", {"default": 64, "min": 1, "max": 2048}),
                "penalty": (
                    "FLOAT",
                    {"default": 1.15, "min": 1.0, "max": 1.5},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 200}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING",)

    def process(
        self,
        string,
        character,
        template,
        api,
        seed,
        min_tokens,
        max_tokens,
        penalty,
        temperature,
        top_k,
        top_p,
    ):
        request = {
            "user_input": string,
            "character": character,
            "instruction_template": template,
            "seed": seed,
            "min_length": min_tokens,
            "max_new_tokens": max_tokens,
            "repetition_penalty": penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_at_newline": True,
            "chat_prompt_size": 2048,
            "mode": "chat",
            "history": {
                "internal": [],
                "visible": [],
            },
        }

        response = requests.post(api, json=request)
        result = response.json()["results"][0]["history"]["visible"][-1][1]
        return (result,)


class Join:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_1": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": "", "multiline": True}),
                "string_2": ("STRING", {"default": "", "multiline": True}),
            }
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING",)

    def process(self, string_1, separator, string_2):
        return (separator.join((string_1, string_2)),)


class Print:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prefix": ("STRING", {"default": "Zuellni"}),
                "string": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "Zuellni/Text"
    FUNCTION = "process"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def process(self, prefix, string):
        prefix = f"[\033[94m{prefix}\033[0m]: " if prefix else ""
        print(prefix + string)
        return (None,)
