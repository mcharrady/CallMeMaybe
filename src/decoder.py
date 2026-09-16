"""Provide token-level constrained decoding for function parameters."""
import json
from llm_sdk import Small_LLM_Model
from .models import FunctionDef, ValType, FunCall
from typing import Any
from . import validator


NUMBER_CHARS = "0123456789.-e+"
BOOL_LITERALS = ("true", "false")

_char_id_cache: dict[str, int] | None = None
_str_vocab_cache: tuple[set[int], set[int]] | None = None

SkeletonPlan = list[list[int] | ValType]


def build_skeleton(chosen_def: FunctionDef) -> list[str | ValType]:
    """Build a JSON generation skeleton from a function definition.
    
    Args:
        chosen_def: The function definition whose parameters determine
           the structure of the generated JSON object.

    Returns:
        A list of an ordered sequence containing fixed JSON
            fragments and 'ValType' markers representing values.
    """
    pieces: list[str | ValType] = ["{"]
    names = list(chosen_def.parameters)
    for i, name in enumerate(names):
        pieces.append(f'"{name}":')
        pieces.append(chosen_def.parameters[name].type)
        if i < len(names) - 1:
            pieces.append(",")
    pieces.append("}")
    return pieces


def build_skeleton_plan(
        chosen_def: FunctionDef,
        model: Small_LLM_Model
        ) -> SkeletonPlan:
    """Convert a JSON skeleton into a token-generation plan.
    
    Args:
        chosen_def: Function definition used to determine the required
            JSON structure and parameter types.
        model: Language model used to encode fixed JSON fragments into
            token IDs.

    Returns:
        plan: An ordered generation plan containing encoded structural
            token sequences and parameter type markers.
    """
    plan: SkeletonPlan = []
    for piece in build_skeleton(chosen_def):
        if isinstance(piece, ValType):
            plan.append(piece)
        else:
            plan.append(model.encode(piece).tolist()[0])
    return plan


def is_complete(plan: SkeletonPlan, index: int) -> bool:
    """Check whether every element of a generation plan has been
    processed.

    Args:
        plan: The complete skeleton generation plan.
        index: Current position within the generation plan.

    Returns:
        'True' when the index has reached or passed the end of the
            plan; otherwise 'False'.
    """
    return index >= len(plan)


def value_is_complete(buffer: str, val_type: ValType) -> bool:
    """Determine whether the currently generated value is complete.
    
    Args:
        buffer: Text generated so far for the current parameter value.
        val_type: Expected type of the current parameter value.

    Returns:
        'True' when the value satisfies the completion condition for
            its type; otherwise 'False'.
    """
    if val_type == ValType.STRING:
        return (len(buffer) >= 2
                and buffer.startswith('"')
                and buffer.endswith('"'))
    if val_type == ValType.BOOLEAN:
        return buffer in BOOL_LITERALS
    return False


def char_token_ids(model: Small_LLM_Model) -> dict[str, int]:
    """Return token IDs corresponding to characters used in number 
    generation.
    
    Args:
        model: Language model used to encode individual numeric
            characters.

    Returns:
        _char_id_cache: A mapping from supported numeric characters to
            their corresponding model token IDs.
    """
    global _char_id_cache
    if _char_id_cache is None:
        _char_id_cache = {
            c: model.encode(c).tolist()[0][0]
            for c in NUMBER_CHARS
            }
    return _char_id_cache


def str_vocab(model: Small_LLM_Model) -> tuple[set[int], set[int]]:
    """Build and cache token sets suitable for constrained string
    generation.

    Args:
        model: Language model whose vocabulary is used to construct 
            the string-token sets.

    Returns:
        _str_vocab_cache: A pair containing the token IDs used
            for quotation marks and the token IDs considered safe for
            string bodies.
    """
    global _str_vocab_cache
    if _str_vocab_cache is None:
        quote_ids = set(model.encode('"').tolist()[0])
        with open(model.get_path_to_vocab_file()) as f:
            raw: dict[str, str] = json.load(f)
        body_ids = set()
        for tid_str in raw.values():
            tid = int(tid_str)
            decoded = model.decode([tid])
            if (
                decoded and '"' not in decoded
                and "\\" not in decoded
                and "{" not in decoded and "}" not in decoded
                and all(ord(c) >= 0x20 for c in decoded)
            ):
                body_ids.add(tid)
        _str_vocab_cache = (quote_ids, body_ids)
    return _str_vocab_cache


def number_candidates(
        buffer: str, model: Small_LLM_Model,
        allow_dot: bool = False
        ) -> set[int]:
    """Determine valid next tokens for the current numeric value.

    Args:
        buffer: Numeric text generated so far.
        model: Language model used to map valid numeric characters to
            token IDs.
        allow_dot: Whether decimal-point and scientific-notation 
            characters may be considered for the current number.
            Defaults to ``False``.

    Returns:
        A set of token IDs that can validly continue the current numeric
            value.
    """
    ids = char_token_ids(model)
    if buffer.lstrip("-") == "0":
        allowed: set[str] = set()
    else:
        allowed = set("0123456789")
    if buffer == "":
        allowed.add("-")
    if allow_dot and "e" not in buffer:
        if "." not in buffer.lstrip("-") and buffer.lstrip("-") != "":
            allowed.add(".")
        if buffer and buffer[-1].isdigit():
            allowed.add("e")
    if buffer and buffer[-1] == "e":
        allowed.add("-")
        allowed.add("+")
    return {ids[c] for c in allowed}


def bool_candidates(buffer: str, model: Small_LLM_Model) -> set[int]:
    """Determine valid next tokens for a JSON boolean value.

    Args:
        buffer: Boolean text generated so far.
        model: Language model used to encode the next characters.

    Returns:
        A set of token IDs that can continue either 'true' or 'false'
            based on the current buffer.
    """
    return {
        model.encode(lit[len(buffer)]).tolist()[0][0]
        for lit in BOOL_LITERALS if lit.startswith(buffer)
    }


def is_repeating(buffer: str) -> bool:
    """Detect repeated suffix patterns in a generated string.

    Args:
        buffer: String content generated so far.

    Returns:
        'True' if a repeated suffix pattern is detected; otherwise
            'False'.
    """
    n = len(buffer)
    for length in range(5, 20):
        if n >= 2 * length:
            tail = buffer[-length:]
            if tail in buffer[:-length]:
                return True
    return False


def str_candidates(
        buffer: str, model: Small_LLM_Model, regex_param: bool
        ) -> set[int]:
    """Determine valid next tokens for a constrained string value.

    Args:
        buffer: String content generated so far, including quotation
            marks when they have already been generated.
        model: Language model whose vocabulary provides the allowed
            string token IDs.
        regex_param: Whether the current parameter represents a regular
            expression, enabling additional regex-specific termination
            constraints.

    Returns:
        A set of token IDs that are valid continuations of the current
            constrained string.
    """
    quote_ids, body_ids = str_vocab(model)
    if buffer == "":
        return quote_ids
    if len(buffer) > 60 or is_repeating(buffer):
        return quote_ids
    if regex_param and buffer[-1] in '+*])':
        return quote_ids
    return quote_ids | body_ids


def get_valid_token_ids(
        plan: SkeletonPlan, index: int, buffer: str,
        model: Small_LLM_Model, regex_param: bool
        ) -> set[int]:
    """Determine valid next token IDs for the current generation step.

    Args:
        plan: Schema-driven generation plan containing JSON structure and
            parameter value types.
        index: Current position in the generation plan.
        buffer: Text generated so far for the current parameter value.
        model: Language model used for token encoding and vocabulary
            access.
        regex_param: Whether the current string parameter represents a
            regular expression.

    Returns:
        A set of token IDs that are valid for the current generation
            step.
    """
    val_type = plan[index]
    assert isinstance(val_type, ValType)
    if val_type == ValType.STRING:
        return str_candidates(buffer, model, regex_param)
    if val_type == ValType.BOOLEAN:
        return bool_candidates(buffer, model)
    if val_type == ValType.NUMBER:
        allowed = number_candidates(buffer, model, True)
    else:
        allowed = number_candidates(buffer, model)
    next_piece = plan[index + 1]
    if (
        buffer.lstrip("-") != ""
        and buffer[-1] not in ".e+-"
        and isinstance(next_piece, list)
    ):
        allowed.add(next_piece[0])
    return allowed


def build_final_json(
        chosen_def: FunctionDef, prompt: str, raw_params_text: str
        ) -> FunCall:
    """Parse, normalize, and validate generated parameter JSON.

    Args:
        chosen_def: Function definition corresponding to the generated
            parameters.
        prompt: Original natural-language request.
        raw_params_text: JSON text containing the generated parameter
            object.

    Returns:
        A FunCall:
            The parsed and validated function call containing the
            prompt, selected function name, and generated parameters.
    """
    parameters: dict[str, Any] = json.loads(raw_params_text)
    for name, value in chosen_def.parameters.items():
        if value.type == ValType.NUMBER:
            parameters[name] = float(parameters[name])
    print(prompt)
    print(chosen_def.name)
    print(parameters)
    result = FunCall(
        prompt=prompt,
        name=chosen_def.name,
        parameters=parameters
    )
    validator.validate(result, chosen_def)
    return result
