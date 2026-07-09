import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field

load_dotenv()


app = FastAPI(
    title="LLM API Starter",
    version="0.2.0",
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
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

        if not api_key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        return client, provider, model

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")

        if not api_key:
            raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "LLM API Starter"),
            },
        )

        return client, provider, model

    raise RuntimeError(f"不支持的 LLM_PROVIDER: {provider}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())

    try:
        client, provider, model = create_llm_client()

        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个 AI 助手，回答要准确、简洁、结构清晰。",
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
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
