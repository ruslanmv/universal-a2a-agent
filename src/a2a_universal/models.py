from __future__ import annotations
import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict, Any

# =============================================================================
# Core A2A Object Models
# =============================================================================

TextPartType = Literal["text"]

class TextPart(BaseModel):
    """Defines a simple text part of a message."""
    type: TextPartType = "text"
    text: str

class Message(BaseModel):
    """
    Defines the standard Message object.
    - `messageId` is optional for incoming requests and gets a default value.
    - `contextId` is optional.
    """
    role: Literal["user", "agent"]
    messageId: Optional[str] = Field(default_factory=lambda: f"msg-{uuid.uuid4()}")
    contextId: Optional[str] = None
    parts: List[TextPart]
    metadata: Optional[Dict[str, Any]] = None

class TaskStatus(BaseModel):
    """Defines the status of a task."""
    state: Literal["pending", "running", "completed", "failed", "cancelled"]
    details: Optional[str] = None

class Task(BaseModel):
    """Defines a Task object, representing a unit of work."""
    kind: Literal["task"] = "task"
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4()}")
    contextId: Optional[str] = None
    status: TaskStatus

class StatusUpdate(BaseModel):
    """Defines an incremental status update for a Task."""
    kind: Literal["status-update"] = "status-update"
    id: str
    status: TaskStatus

class Artifact(BaseModel):
    """Defines an Artifact, which can be any data payload."""
    parts: List[Dict[str, Any]]

class ArtifactUpdate(BaseModel):
    """Defines an incremental update for an Artifact."""
    kind: Literal["artifact-update"] = "artifact-update"
    id: str
    artifact: Artifact

# A union of all possible successful response event types from an agent
A2ASuccessEvent = Union[Task, StatusUpdate, ArtifactUpdate, Message]

# =============================================================================
# JSON-RPC Request/Response Models
# =============================================================================

class A2AParams(BaseModel):
    """Parameters for the message/send method."""
    message: Message

class JSONRPCRequest(BaseModel):
    """Defines a JSON-RPC 2.0 request for the message/send method."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    method: Literal["message/send"]
    params: A2AParams

class JSONRPCSuccess(BaseModel):
    """
    Defines a successful JSON-RPC 2.0 response.
    The `result` field directly contains one of the valid A2A event objects.
    """
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    result: A2ASuccessEvent

class JSONRPCErrorObj(BaseModel):
    """The 'error' object within a JSON-RPC error response."""
    code: int
    message: str

class JSONRPCError(BaseModel):
    """Defines a JSON-RPC 2.0 error response."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]]
    error: JSONRPCErrorObj

