import json
from collections.abc import AsyncIterator
from uuid import uuid4

from openai import AsyncOpenAI, APIConnectionError, APIError, APITimeoutError

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse


SYSTEM_PROMPT = "你是一个 AI 助手，回答要准确、简洁、结构清晰。"


class EmptyModelResponseError(RuntimeError):
    pass


def create_llm_client() -> tuple[AsyncOpenAI, str, str]:
    """根据配置创建 LLM Client，并返回 client、provider 和 model。"""
    provider = settings.LLM_PROVIDER

    if provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

        client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

        return client, provider, settings.DEEPSEEK_MODEL

    if provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY")

        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": settings.OPENROUTER_SITE_URL,
                "X-Title": settings.OPENROUTER_APP_NAME,
            },
        )

        return client, provider, settings.OPENROUTER_MODEL

    raise RuntimeError(f"不支持的 LLM_PROVIDER: {provider}")


def build_messages(message: str) -> list[dict[str, str]]:
    """构造发送给模型的 messages。"""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": message,
        },
    ]


def to_sse_data(data: dict) -> str:
    """把 Python dict 转成 SSE 数据格式。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def chat_completion(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    client, provider, model = create_llm_client()

    completion = await client.chat.completions.create(
        model=model,
        messages=build_messages(request.message),
        temperature=0.3,
        stream=False,
    )

    answer = completion.choices[0].message.content

    if not answer:
        raise EmptyModelResponseError("模型返回为空")

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        provider=provider,
        model=model,
    )


async def stream_chat_completion(request: ChatRequest) -> AsyncIterator[str]:
    """流式调用模型，并按 SSE 格式逐段返回。"""
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
