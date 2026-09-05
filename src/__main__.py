"""."""
import argparse
import sys
from llm_sdk import Small_LLM_Model
from . import parser, generator, output, models


def parse_args() -> argparse.Namespace:
    """."""
    promfile = "data/input/function_calling_tests.json"
    funcfile = "data/input/functions_definition.json"
    outfile = "data/output/function_calling_results.json"
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--functions_definition", default=funcfile)
    arg_parser.add_argument("--input", default=promfile)
    arg_parser.add_argument("--output", default=outfile)
    return arg_parser.parse_args()


def main() -> None:
    """."""
    args = parse_args()##########
    prompts, func_defs = parser.parse_data(
        args.input, args.functions_definition
    )
    model = Small_LLM_Model()
    results : list[models.FunCall] = [
        generator.generate(model, prompt.prompt, func_defs)
        for prompt in prompts
    ]

    output.write_output(results, args.output)


if __name__ == "__main__":
    try:
        main()########################
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
