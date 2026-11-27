from fastapi import APIRouter, HTTPException, Response, status

from app.schemas.chat import ConversationResponse
from app.services.conversation_store import conversation_store


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    responses={404: {"description": "会话不存在"}},
)
async def get_conversation(conversation_id: str) -> ConversationResponse:
    """查询指定会话当前保留的历史消息。"""
    messages = conversation_store.get_messages(conversation_id)

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    return ConversationResponse(
        conversation_id=conversation_id,
        messages=messages,
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_conversation(conversation_id: str) -> Response:
    """删除指定会话的历史；重复删除同样返回 204。"""
    conversation_store.clear(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
