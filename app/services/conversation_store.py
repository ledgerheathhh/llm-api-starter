from collections import defaultdict

from app.schemas.chat import ChatMessage


class InMemoryConversationStore:
    def __init__(self, max_messages: int = 20) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages 必须大于 0")

        self._conversations: dict[str, list[ChatMessage]] = defaultdict(list)
        self._max_messages = max_messages

    def get_messages(self, conversation_id: str) -> list[ChatMessage]:
        """
        获取指定会话的消息列表。

        返回消息副本，防止外部代码修改内部存储。
        """
        return [
            message.model_copy(deep=True)
            for message in self._conversations.get(conversation_id, [])
        ]

    def append_messages(
        self,
        conversation_id: str,
        messages: list[ChatMessage],
    ) -> None:
        """
        向指定会话追加消息，并限制最大消息数量。
        """
        if not messages:
            return

        conversation = self._conversations[conversation_id]

        conversation.extend(
            message.model_copy(deep=True)
            for message in messages
        )

        if len(conversation) > self._max_messages:
            self._conversations[conversation_id] = conversation[
                -self._max_messages :
            ]

    def clear(self, conversation_id: str) -> None:
        """
        删除指定会话。
        """
        self._conversations.pop(conversation_id, None)


conversation_store = InMemoryConversationStore()
