import json
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from openai import AsyncOpenAI, APIConnectionError, APIError, APITimeoutError

from app.core.config import settings
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from app.services.conversation_store import conversation_store


SYSTEM_PROMPT = "你是一个 AI 助手，回答要准确、简洁、结构清晰。"
logger = logging.getLogger(__name__)

LLMClientConfig = tuple[AsyncOpenAI, str, str]
_llm_client_config: LLMClientConfig | None = None


class EmptyModelResponseError(RuntimeError):
    pass


def create_llm_client() -> LLMClientConfig:
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


def get_llm_client() -> LLMClientConfig:
    """首次调用时创建客户端，后续请求复用同一个连接池。"""
    global _llm_client_config

    if _llm_client_config is None:
        _llm_client_config = create_llm_client()

    return _llm_client_config


async def close_llm_client() -> None:
    """关闭已创建的客户端。"""
    global _llm_client_config

    if _llm_client_config is None:
        return

    client, _, _ = _llm_client_config

    try:
        await client.close()
    finally:
        _llm_client_config = None


def build_messages(
    history: list[ChatMessage],
    current_message: str,
) -> list[dict[str, str]]:
    """根据系统提示词、历史消息和当前问题构造模型上下文。"""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        {
            "role": message.role,
            "content": message.content,
        }
        for message in history
    )

    messages.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    return messages


def to_sse_data(data: dict) -> str:
    """把 Python dict 转成 SSE 数据格式。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def chat_completion(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())

    history = conversation_store.get_messages(conversation_id)

    client, provider, model = get_llm_client()

    completion = await client.chat.completions.create(
        model=model,
        messages=build_messages(
            history=history,
            current_message=request.message,
        ),
        temperature=0.3,
        stream=False,
    )

    if not completion.choices:
        raise EmptyModelResponseError("模型返回为空")

    answer = completion.choices[0].message.content

    if not answer:
        raise EmptyModelResponseError("模型返回为空")

    conversation_store.append_messages(
        conversation_id,
        [
            ChatMessage(
                role="user",
                content=request.message,
            ),
            ChatMessage(
                role="assistant",
                content=answer,
            ),
        ],
    )

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
        client, provider, model = get_llm_client()

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
            messages=build_messages(
                history=[],
                current_message=request.message,
            ),
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
        logger.exception("模型请求超时")
        yield to_sse_data(
            {
                "type": "error",
                "message": "模型请求超时",
            }
        )

    except APIConnectionError:
        logger.exception("无法连接到模型服务")
        yield to_sse_data(
            {
                "type": "error",
                "message": "无法连接到模型服务",
            }
        )

    except APIError:
        logger.exception("模型服务错误")
        yield to_sse_data(
            {
                "type": "error",
                "message": "模型服务错误",
            }
        )

    except RuntimeError:
        logger.exception("模型服务配置错误")
        yield to_sse_data(
            {
                "type": "error",
                "message": "模型服务配置错误",
            }
        )

    except Exception:
        logger.exception("流式聊天服务异常")
        yield to_sse_data(
            {
                "type": "error",
                "message": "服务异常",
            }
        )
