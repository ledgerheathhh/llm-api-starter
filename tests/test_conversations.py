import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers import conversations as conversations_router
from app.schemas.chat import ChatMessage
from app.services import llm_service
from app.services.conversation_store import InMemoryConversationStore


class ConversationAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore()
        self.enterContext(
            patch.object(conversations_router, "conversation_store", self.store)
        )
        self.enterContext(
            patch.object(llm_service, "conversation_store", self.store)
        )
        self.enterContext(
            patch.object(
                llm_service,
                "get_llm_client",
                side_effect=AssertionError("Unexpected LLM call"),
            )
        )
        self.client = self.enterContext(TestClient(app))

    def seed_conversation(self, conversation_id: str) -> None:
        self.store.append_messages(
            conversation_id,
            [
                ChatMessage(role="user", content="你好"),
                ChatMessage(role="assistant", content="你好，有什么可以帮助你？"),
            ],
        )

    def test_get_conversation_returns_history(self) -> None:
        self.seed_conversation("conversation-1")
        self.seed_conversation("conversation-2")

        response = self.client.get("/conversations/conversation-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "conversation_id": "conversation-1",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，有什么可以帮助你？"},
                ],
            },
        )

    def test_get_unknown_conversation_returns_404(self) -> None:
        response = self.client.get("/conversations/unknown")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "会话不存在"})
        self.assertEqual(self.store.get_messages("unknown"), [])

    def test_delete_conversation_clears_only_target(self) -> None:
        self.seed_conversation("conversation-1")
        self.seed_conversation("conversation-2")
        other_history = self.store.get_messages("conversation-2")

        response = self.client.delete("/conversations/conversation-1")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        self.assertEqual(
            self.client.get("/conversations/conversation-1").status_code, 404
        )
        self.assertEqual(
            self.client.get("/conversations/conversation-2").status_code, 200
        )
        self.assertEqual(self.store.get_messages("conversation-2"), other_history)

    def test_delete_is_idempotent(self) -> None:
        self.seed_conversation("conversation-1")

        for _ in range(2):
            response = self.client.delete("/conversations/conversation-1")
            self.assertEqual(response.status_code, 204)
            self.assertEqual(response.content, b"")

        response = self.client.delete("/conversations/never-created")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")

    def test_chat_history_can_be_queried_deleted_and_recreated(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="模拟回答"))]
        )
        create = AsyncMock(return_value=completion)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.object(
            llm_service, "get_llm_client", return_value=(client, "deepseek", "mock-model")
        ):
            response = self.client.post("/chat", json={"message": "第一轮"})
            self.assertEqual(response.status_code, 200)
            conversation_id = response.json()["conversation_id"]
            path = f"/conversations/{conversation_id}"

            history = self.client.get(path)
            self.assertEqual(history.status_code, 200)
            self.assertEqual(
                history.json()["messages"],
                [
                    {"role": "user", "content": "第一轮"},
                    {"role": "assistant", "content": "模拟回答"},
                ],
            )
            self.assertEqual(self.client.delete(path).status_code, 204)
            self.assertEqual(self.client.get(path).status_code, 404)

            response = self.client.post(
                "/chat",
                json={"message": "新一轮", "conversation_id": conversation_id},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["conversation_id"], conversation_id)
            self.assertEqual(
                create.await_args.kwargs["messages"],
                [
                    {"role": "system", "content": llm_service.SYSTEM_PROMPT},
                    {"role": "user", "content": "新一轮"},
                ],
            )
            self.assertEqual(
                self.client.get(path).json()["messages"],
                [
                    {"role": "user", "content": "新一轮"},
                    {"role": "assistant", "content": "模拟回答"},
                ],
            )

    def test_stream_history_is_available_to_conversation_api(self) -> None:
        async def stream_chunks():
            for content in ("流式", "回答"):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value=stream_chunks())
                )
            )
        )
        with patch.object(
            llm_service, "get_llm_client", return_value=(client, "deepseek", "mock-model")
        ):
            response = self.client.post("/chat/stream", json={"message": "流式问题"})

        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[0]["type"], "meta")
        self.assertEqual(events[-1], {"type": "done"})
        path = f"/conversations/{events[0]['conversation_id']}"
        history = self.client.get(path)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(
            history.json()["messages"],
            [
                {"role": "user", "content": "流式问题"},
                {"role": "assistant", "content": "流式回答"},
            ],
        )
        self.assertEqual(self.client.delete(path).status_code, 204)
        self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
