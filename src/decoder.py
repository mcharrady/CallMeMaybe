"""."""

import json
from llm_sdk import Small_LLM_Model
from .models import FunctionDef, ValType


NUMBER_CHARS = "0123456789.-"
BOOL_LITERALS = ("true", "false")

_char_id_cache: dict[str, int] | None = None
_str_vocab_cache: tuple[set[int], set[int]] | None = None

SkeletonPlan = list[list[int] | ValType]


def build_skeleton(chosen_def: FunctionDef) -> list[str | ValType]:
    """."""
    pass


def build_skeleton_plan(
        chosen_def: FunctionDef,
        model: Small_LLM_Model
        ) -> SkeletonPlan:
    """."""
    pass


def is_complete(plan: SkeletonPlan, index: int) -> bool:
    """."""
    pass


def value_is_complete(buffer: str, val_type: ValType) -> bool:
    """."""
    pass


def char_token_ids(model: Small_LLM_Model) -> dict[str, int]:
    """."""
    pass


def str_vocab(model: Small_LLM_Model) -> tuple[set[int], set[int]]:
    """."""
    global _str_vocab_cache
    if _str_vocab_cache is None:
        quote_ids = set(model.encode('"').tolist()[0][0])
        with open(model.get_path_to_vocab_file()) as f:
            raw: dict[str, str] = json.load(f)
        body_ids = {
            int(tid) for tid, text in raw.items()
            if text and '"' not in text and "\\" not in text
        }
        _str_vocab_cache = (quote_ids, body_ids)
    return _str_vocab_cache


def number_candidates(
        buffer: str, model: Small_LLM_Model,
        allow_dot: bool = False
        ) -> set[int]:
    """."""
    ids = char_token_ids(model)#############
    allowed = set("0123456789")
    if buffer == "":
        allowed.add("-")
    if allow_dot and "." not in buffer.lstrip("-"):
        allowed.add(".")
    return {ids[c] for c in allowed}


def bool_candidates(buffer: str, model: Small_LLM_Model) -> set[int]:
    """."""
    return {
        model.encode(lit[len(buffer)]).tolist()[0][0]
        for lit in BOOL_LITERALS if lit.startswith(buffer)
    }


def str_candidates(buffer: str, model: Small_LLM_Model) -> set[int]:
    """."""
    quote_ids, body_ids = str_vocab(model)################
    return quote_ids if buffer == "" else quote_ids | body_ids


def get_valid_token_ids(
        plan: SkeletonPlan, index: int, buffer: str,
        model: Small_LLM_Model
    ) -> set[int]:
    """."""
    val_type = plan[index]
    assert isinstance(val_type, ValType)
    if val_type == ValType.STRING:
        return str_candidates(buffer, model)#############
    if val_type == ValType.BOOLEAN:
        return bool_candidates(buffer, model)
    if val_type == ValType.NUMBER:
        return number_candidates(buffer, model, True)#############
    return number_candidates(buffer, model)#############


def build_final_json(
        chosen_def: FunctionDef, prompt: str, raw_params_text: str
    ) -> str:
    """."""
    pass
