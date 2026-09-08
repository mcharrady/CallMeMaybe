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

    param_prom = (
        "Extract parameters in JSON format.\n\n"
        "Example:\n"
        "Function: fn_add_numbers\n"
        "User Request: Add 5 and 3\n"
        "Function call: fn_add_numbers{\"a\":5,\"b\":3}\n\n"
        f"Function: {chosen_def.name}\n"
        f"User Request: {prompt}\n"
        "Function call: "
    )

    input_ids = model.encode(param_prom).tolist()[0]

    generate_ids = name_progress[:]

    plan = decoder.build_skeleton_plan(chosen_def, model)
    plan_index = 0
    value_buffer = ""

    while not decoder.is_complete(plan, plan_index):
        piece = plan[plan_index]

        if isinstance(piece, list):
            generate_ids.extend(piece)
            plan_index += 1
            continue

        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        allowed = decoder.get_valid_token_ids(
            plan, plan_index, value_buffer, model
            )
        next_token = pick_highest(logits, allowed)

        if piece not in (ValType.STRING, ValType.BOOLEAN):
            next_piece = plan[plan_index + 1]
            if next_token == next_piece[0]:
                plan_index += 1
                value_buffer = ""
                continue

        generate_ids.append(next_token)
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
        model.decode([t]) for t in generate_ids[len(name_progress):]
    )

    return decoder.build_final_json(chosen_def, prompt, raw_params_text)
