"""Pydantic models used to represent function-calling data."""
from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Any


class ValType(str, Enum):
    """the supported parameter value types."""
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"


class ParaSchema(BaseModel):
    """the expected type of a function parameter."""
    type: ValType


class Prompt(BaseModel):
    """Represent a natural-language prompt."""
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionDef(BaseModel):
    """Represent a function available for language-model function 
    calling.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParaSchema]
    returns: ParaSchema


class FunCall(BaseModel):
    """Represent a generated structured function call."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
