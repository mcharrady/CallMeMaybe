"""."""
from llm_sdk import Small_LLM_Model
from .models import FunctionDef, FunCall, ValType
from . import decoder


def get_candidates(
        function_defs: list[FunctionDef], model: Small_LLM_Model
    ) -> list[tuple[FunctionDef, list[int]]]:
    """."""
    return [
        (fn, model.encode(fn.name).tolist()[0])
        for fn in function_defs
    ]


def valid_next_tokens(
        candidates: list[tuple[FunctionDef, list[int]]],
        progress: list[int]
    ) -> set[int]:
    """."""
    still_possible = [
        (fn, token_ids) for fn, token_ids in candidates
        if token_ids[:len(progress)] == progress
    ]
    return {token_ids[len(progress)] for fn, token_ids in still_possible}


def selected_function(
        candidates: list[tuple[FunctionDef, list[int]]],
        progress: list[int]
    ) -> FunctionDef | None:
    """."""
    still_possible = [
        (fn, t) for fn, t in candidates
        if t[:len(progress)] == progress
    ]
    if len(still_possible) == 1 and still_possible[0][1] == progress:
        return still_possible[0][0]
    return None


def pick_highest(logits: list[float], allowed: set[int]) -> int:
    """."""
    return max(allowed, key=lambda i: logits[i])


def generate(
        model: Small_LLM_Model, prompt: str,
        function_defs: list[FunctionDef],
        fun_c_ontext: str,
        max_tokens: int = 256
    ) -> FunCall:
    """."""
    full_input = (
        "Choose the correct function for the user.\n\n"
        f"Available functions:\n{fun_c_ontext}\n\n"
        "Example:\n"
        "(User Request: Add 5 and 3\n"
        "Function Name: fn_add_numbers)\n\n"
        f"\n\nUser Request: {prompt}\nFunction call:"
        )
    input_ids: list[int] = model.encode(full_input).tolist()[0]
    candidates = get_candidates(function_defs, model)
    name_progress: list[int] = []

    while selected_function(candidates, name_progress) is None:
        logits = model.get_logits_from_input_ids(input_ids + name_progress)
        allowed = valid_next_tokens(candidates, name_progress)
        print([model.decode([t]) for t in allowed])
        next_token = pick_highest(logits, allowed)
        name_progress.append(next_token)
        
        print(model.decode(name_progress))

    chosen_def = selected_function(candidates, name_progress)
    if chosen_def is None:
        raise RuntimeError("phase 1 exited without selecting a function")

    param_prom :list[str] = [
        "Generate a valid JSON object with the parameters of ",
        "the chosen function.",
        "You must choose the correct function parameter values from the prompt",
        "Rules: ",
        '- If the prompt says "replace all numbers", just use "[0-9]+" (DO NOT ADD ANY THING, THIS IS ENOUGH) for regex, nothing more.',
        '- If the prompt says "Substitute the word cat", just use "cat" for regex, nothing more.',
        "Example output: ",
        '{"prompt":"Compute the sum of 15 and 27",',
        '"name":"fn_add_numbers",',
        '"parameters":{"a":15,"b":27}} ',
        "Example 2: ",
        '{"prompt":"Replace every sequence of digits in \'Order 512 ',
        "costs 49 dollars' with <NUM>",
        ',"name":"fn_substitute_string_with_regex",',
        '"parameters":{',
        '"source_string":"Order 512 costs 49 dollars",',
        '"regex":"[0-9]+",',
        '"replacement":"<NUM>"',
        '}} ',
        "Example 3: ",
        '{"prompt":"Replace every occurrence of \'apple\' with \'orange\' ',
        "in 'apple pie and apple juice'",
        '"name":"fn_substitute_string_with_regex",',
        '"parameters":{',
        '"source_string":"apple pie and apple juice",',
        '"regex":"apple",',
        '"replacement":"orange"',
        '}} ',
        '{"prompt":"',prompt,'",',
        '"name":"',chosen_def.name,'",',
        '"parameters":'
    ]

    input_ids = model.encode(''.join(param_prom)).tolist()[0]

    generate_ids = []

    plan = decoder.build_skeleton_plan(chosen_def, model)
    plan_index = 0
    value_buffer = ""
    regex_param = False

    while not decoder.is_complete(plan, plan_index):
        piece = plan[plan_index]
        

        if isinstance(piece, list):
            if "regex" in model.decode(piece):
                regex_param = True
            else:
                regex_param = False
            generate_ids.extend(piece)
            plan_index += 1
            continue

        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        allowed = decoder.get_valid_token_ids(
            plan, plan_index, value_buffer, model, regex_param
            )
        next_token = pick_highest(logits, allowed)

        if piece not in (ValType.STRING, ValType.BOOLEAN):
            next_piece = plan[plan_index + 1]
            if next_token == next_piece[0]:
                plan_index += 1
                value_buffer = ""
                continue

        generate_ids.append(next_token)
        print(model.decode(generate_ids))
        value_buffer += model.decode([next_token])
        print(value_buffer)

        if decoder.value_is_complete(value_buffer, piece):
            plan_index += 1
            value_buffer = ""

        if len(generate_ids) > max_tokens:
            raise RuntimeError(
                "generation exceeded max_tokens without completing"
                )
    raw_params_text = "".join(
        model.decode([t]) for t in generate_ids
    )

    return decoder.build_final_json(chosen_def, prompt, raw_params_text)
