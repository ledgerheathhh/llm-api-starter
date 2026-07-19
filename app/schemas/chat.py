from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
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
