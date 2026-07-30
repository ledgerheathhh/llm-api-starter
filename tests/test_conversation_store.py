import unittest

from app.schemas.chat import ChatMessage
from app.services.conversation_store import InMemoryConversationStore


class InMemoryConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore(max_messages=4)

    def test_append_and_get_messages(self) -> None:
        self.store.append_messages(
            "conversation-1",
            [
                ChatMessage(
                    role="user",
                    content="什么是 FastAPI？",
                ),
                ChatMessage(
                    role="assistant",
                    content="FastAPI 是一个 Python Web 框架。",
                ),
            ],
        )

        messages = self.store.get_messages("conversation-1")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "什么是 FastAPI？")
        self.assertEqual(messages[1].role, "assistant")

    def test_different_conversations_are_isolated(self) -> None:
        self.store.append_messages(
            "conversation-1",
            [
                ChatMessage(
                    role="user",
                    content="会话一",
                ),
            ],
        )

        self.store.append_messages(
            "conversation-2",
            [
                ChatMessage(
                    role="user",
                    content="会话二",
                ),
            ],
        )

        conversation_1 = self.store.get_messages("conversation-1")
        conversation_2 = self.store.get_messages("conversation-2")

        self.assertEqual(conversation_1[0].content, "会话一")
        self.assertEqual(conversation_2[0].content, "会话二")

    def test_messages_are_trimmed_to_limit(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=f"message-{index}",
            )
            for index in range(6)
        ]

        self.store.append_messages(
            "conversation-1",
            messages,
        )

        stored_messages = self.store.get_messages("conversation-1")

        self.assertEqual(len(stored_messages), 4)
        self.assertEqual(stored_messages[0].content, "message-2")
        self.assertEqual(stored_messages[-1].content, "message-5")

    def test_clear_conversation(self) -> None:
        self.store.append_messages(
            "conversation-1",
            [
                ChatMessage(
                    role="user",
                    content="需要被清除的消息",
                ),
            ],
        )

        self.store.clear("conversation-1")

        self.assertEqual(
            self.store.get_messages("conversation-1"),
            [],
        )

    def test_get_messages_does_not_expose_internal_list(self) -> None:
        self.store.append_messages(
            "conversation-1",
            [
                ChatMessage(
                    role="user",
                    content="原始消息",
                ),
            ],
        )

        messages = self.store.get_messages("conversation-1")
        messages.clear()

        stored_messages = self.store.get_messages("conversation-1")

        self.assertEqual(len(stored_messages), 1)

    def test_invalid_max_messages_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryConversationStore(max_messages=0)

    def test_mutating_input_message_does_not_change_store(self) -> None:
        message = ChatMessage(
            role="user",
            content="原始内容",
        )

        self.store.append_messages(
            "conversation-1",
            [message],
        )

        message.content = "外部修改"

        stored_messages = self.store.get_messages("conversation-1")

        self.assertEqual(
            stored_messages[0].content,
            "原始内容",
        )

    def test_mutating_returned_message_does_not_change_store(self) -> None:
        self.store.append_messages(
            "conversation-1",
            [
                ChatMessage(
                    role="user",
                    content="原始内容",
                ),
            ],
        )

        messages = self.store.get_messages("conversation-1")
        messages[0].content = "外部修改"

        stored_messages = self.store.get_messages("conversation-1")

        self.assertEqual(
            stored_messages[0].content,
            "原始内容",
        )

if __name__ == "__main__":
    unittest.main()
