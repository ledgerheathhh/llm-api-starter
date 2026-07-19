import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIError, APITimeoutError

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import (
    EmptyModelResponseError,
    chat_completion,
    stream_chat_completion,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """普通非流式聊天接口。"""
    try:
        return await chat_completion(request)

    except APITimeoutError:
        logger.exception("模型请求超时")
        raise HTTPException(
            status_code=504,
            detail="模型请求超时",
        )

    except APIConnectionError:
        logger.exception("无法连接到模型服务")
        raise HTTPException(
            status_code=502,
            detail="无法连接到模型服务",
        )

    except APIError:
        logger.exception("模型服务错误")
        raise HTTPException(
            status_code=502,
            detail="模型服务错误",
        )

    except EmptyModelResponseError:
        logger.exception("模型返回为空")
        raise HTTPException(
            status_code=502,
            detail="模型返回为空",
        )

    except RuntimeError:
        logger.exception("模型服务配置错误")
        raise HTTPException(
            status_code=500,
            detail="模型服务配置错误",
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口。"""
    return StreamingResponse(
        stream_chat_completion(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
