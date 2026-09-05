import json
from .models import Prompt, FunctionDef


def json_to_data(path: str) -> list[dict]:
        with open(path) as file:
            data: list[dict] = json.load(file)
        return data


def parse_data(
        promfile: str,
        funcfile: str
        ) -> tuple[list[Prompt], list[FunctionDef]]:
    tests = json_to_data(promfile)
    funcs = json_to_data(funcfile)
    prompts = [Prompt(**prom) for prom in tests]
    funcdefs = [
        FunctionDef(**f)
        for f in funcs
    ]
    return prompts, funcdefs