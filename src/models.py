from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Any


class ValType(str, Enum):
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"


class ParaSchema(BaseModel):
    type: ValType


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParaSchema]
    returns: ParaSchema


class FunCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
