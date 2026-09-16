"""Output the generated function calls to a JSON output file."""
import json
from pathlib import Path
from .models import FunCall

outfile = "data/output/function_calling_results.json"


def write_output(results: list[FunCall], path: str) -> None:
    """Wright generated function calls to a JSON file.

    Args:
        results: A list of validated function calls that should be written
            to the output file.
        path: Destination path where the generated JSON results should be
            stored.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
