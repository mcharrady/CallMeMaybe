"""Parse and validate JSON input used."""
import json
import keyword
from .models import Prompt, FunctionDef


def json_check(pairs: list[tuple]) -> dict:
    """A 'object_pairs_hook' to validate JSON object key-value
    pairs while constructing a dictionary.

    Args:
        pairs: A list of key-value pairs extracted from a JSON object by
            'json.load'.

    Returns:
        A dictionary containing: 
            the validated key-value pairs.
    """
    seen = set()
    for key, value in pairs:
        if not value:
            raise ValueError(f"{key} has an empty value")
        if key in seen:
            raise ValueError(f"Duplicate key detected: {key}")
        seen.add(key)
    return dict(pairs)


def json_to_data(path: str) -> list[dict]:
    """Load and validate a JSON file containing a list of objects.
    
    Args:
        path: Path to the JSON file that should be loaded.

    Returns:
        data: A list containing the validated JSON objects from the
            input file.
    """
    with open(path) as file:
        data: list[dict] = json.load(file, object_pairs_hook=json_check)
    return data


def validate_name(name: str, kind: str, owner: str) -> None:
    """check if the fuction/parameter has a valid name.

    Args:
        name: name of the fuction/parameter
        kind: a flag for the error message
        owner: were we found the error
    """
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Invalid {kind} name in {owner}: '{name}'")


def parse_data(
        promfile: str,
        funcfile: str
        ) -> tuple[list[Prompt], list[FunctionDef]]:
    """Parse prompt and function-definition files into Pydantic models.

    Args:
        promfile: Path to the JSON file containing the input prompts.
        funcfile: Path to the JSON file containing the available function
            definitions.

    Returns:
        A tuple contaning:
            prompts: containing the parsed prompts
            funcdefs: the parsed function definitions.
    """
    tests = json_to_data(promfile)
    funcs = json_to_data(funcfile)
    prompts = [Prompt(**prom) for prom in tests]
    if not prompts:
        raise ValueError(f"No prompt in {promfile}")
    funcdefs = [FunctionDef(**f)for f in funcs]
    names = [f.name for f in funcdefs]
    if not names:
        raise ValueError(f"No function definitions in {funcfile}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate function names in {funcfile}")
    for f in funcdefs:
        validate_name(f.name, "function", funcfile)
        for pname in f.parameters:
            validate_name(pname, "parameter", f.name)
    return prompts, funcdefs
