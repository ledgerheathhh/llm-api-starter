from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ] = Field(
        description="用户发送的消息",
    )
    conversation_id: str | None = Field(
        default=None,
        description="会话 ID，第一次请求可以不传",
    )


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    provider: str
    model: str
