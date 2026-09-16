"""Validate generated function calls against their function
definitions.
"""
from .models import FunCall, FunctionDef, ValType
from typing import Any


def validate(func_call: FunCall, chosen_def: FunctionDef) -> None:
    """Validate a generated function call against its selected
    definition.
    
    Args:
        func_call: The generated function call that must be validated.
        chosen_def: The function definition against which the
            generated call is checked.
    """
    if func_call.name != chosen_def.name:
        raise ValueError(
            "name mismatch: "
            f"{func_call.name} != {chosen_def.name}"
            )
    expected = set(chosen_def.parameters)
    actual = set(func_call.parameters)
    if actual != expected:
        raise ValueError(
            "parameter mismatch: "
            f"{actual} != {expected}"
        )
    for name, schema in chosen_def.parameters.items():
        value = func_call.parameters[name]
        if not matched_type(value, schema.type):
            raise ValueError(
                f"{name}'s type is wrong: "
                f"expected {schema.type}"
                )


def matched_type(value: Any, val_type: ValType) -> bool:
    """Check whether a Python value matches an expected parameter
    type.
    
    Args:
        value: The generated parameter value to inspect.
        val_type: The expected 'ValType' specified by the function
            definition.
    """
    if val_type == ValType.BOOLEAN:
        return isinstance(value, bool)
    if val_type == ValType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if val_type == ValType.NUMBER:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    return isinstance(value, str)
