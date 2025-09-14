from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Literal, Union

TextPartType = Literal["text"]

class TextPart(BaseModel):
    type: TextPartType = "text"
    text: str

class Message(BaseModel):
    role: Literal["user", "agent"]
    messageId: str
    parts: List[TextPart]

class A2AParams(BaseModel):
    message: Message

class A2ARequest(BaseModel):
    method: Literal["message/send"]
    params: A2AParams

class A2AResponse(BaseModel):
    message: Message

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    method: Literal["message/send"]
    params: A2AParams

class JSONRPCSuccess(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    result: A2AResponse

class JSONRPCErrorObj(BaseModel):
    code: int
    message: str

class JSONRPCError(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]]
    error: JSONRPCErrorObj
