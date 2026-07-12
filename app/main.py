import json
import os
from typing import AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIConnectionError, APIError, APITimeoutError
from pydantic import BaseModel, Field


load_dotenv()


app = FastAPI(
    title="LLM API Starter",
    version="0.3.0",
)


# 本地 Vue3 调试时需要 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def create_llm_client() -> tuple[AsyncOpenAI, str, str]:
    """
    根据环境变量创建 LLM Client。

    返回：
    - client: AsyncOpenAI 实例
    - provider: deepseek / openrouter
    - model: 当前使用的模型名
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not api_key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        return client, provider, model

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

        if not api_key:
            raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_SITE_URL",
                    "http://localhost:8000",
                ),
                "X-Title": os.getenv(
                    "OPENROUTER_APP_NAME",
                    "LLM API Starter",
                ),
            },
        )

        return client, provider, model

    raise RuntimeError(f"不支持的 LLM_PROVIDER: {provider}")


def build_messages(message: str) -> list[dict[str, str]]:
    """
    构造发送给模型的 messages。
    后续做多轮会话时，可以在这里加入历史消息。
    """
    return [
        {
            "role": "system",
            "content": "你是一个 AI 助手，回答要准确、简洁、结构清晰。",
        },
        {
            "role": "user",
            "content": message,
        },
    ]


def to_sse_data(data: dict) -> str:
    """
    把 Python dict 转成 SSE 数据格式。

    SSE 格式：
    data: {...}

    注意结尾必须是两个换行：\\n\\n
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    普通非流式聊天接口。
    模型完整生成结束后，一次性返回 answer。
    """
    conversation_id = request.conversation_id or str(uuid4())

    try:
        client, provider, model = create_llm_client()

        completion = await client.chat.completions.create(
            model=model,
            messages=build_messages(request.message),
            temperature=0.3,
            stream=False,
        )

        answer = completion.choices[0].message.content

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="模型返回为空",
            )

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            provider=provider,
            model=model,
        )

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="模型请求超时",
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="无法连接到模型服务",
        )

    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"模型服务错误: {str(e)}",
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


async def stream_chat_response(request: ChatRequest) -> AsyncIterator[str]:
    """
    流式调用模型，并按 SSE 格式逐段返回。
    """
    conversation_id = request.conversation_id or str(uuid4())

    try:
        client, provider, model = create_llm_client()

        yield to_sse_data(
            {
                "type": "meta",
                "conversation_id": conversation_id,
                "provider": provider,
                "model": model,
            }
        )

        stream = await client.chat.completions.create(
            model=model,
            messages=build_messages(request.message),
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if not content:
                continue

            yield to_sse_data(
                {
                    "type": "delta",
                    "content": content,
                }
            )

        yield to_sse_data(
            {
                "type": "done",
            }
        )

    except APITimeoutError:
        yield to_sse_data(
            {
                "type": "error",
                "message": "模型请求超时",
            }
        )

    except APIConnectionError:
        yield to_sse_data(
            {
                "type": "error",
                "message": "无法连接到模型服务",
            }
        )

    except APIError as e:
        yield to_sse_data(
            {
                "type": "error",
                "message": f"模型服务错误: {str(e)}",
            }
        )

    except RuntimeError as e:
        yield to_sse_data(
            {
                "type": "error",
                "message": str(e),
            }
        )

    except Exception as e:
        yield to_sse_data(
            {
                "type": "error",
                "message": f"服务异常: {str(e)}",
            }
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    流式聊天接口。
    前端需要用 fetch + ReadableStream 接收。
    """
    return StreamingResponse(
        stream_chat_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )