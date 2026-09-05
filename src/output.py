import json
from pathlib import Path
from .models import FunCall

outfile = "data/output/function_calling_results.json"


def write_output(results: list[FunCall], path: str ) -> None:
    """."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)