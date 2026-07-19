import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from app import main
from app.routers import chat as chat_router
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import llm_service


class AppLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_closes_llm_client(self) -> None:
        with patch.object(main, "close_llm_client", AsyncMock()) as close_client:
            async with main.lifespan(main.app):
                pass

        close_client.assert_awaited_once_with()


class LLMClientLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        llm_service._llm_client_config = None

    async def asyncTearDown(self) -> None:
        llm_service._llm_client_config = None

    async def test_client_is_reused_and_closed(self) -> None:
        client = SimpleNamespace(close=AsyncMock())
        client_config = (client, "deepseek", "deepseek-chat")

        with patch.object(
            llm_service,
            "create_llm_client",
            return_value=client_config,
        ) as create_client:
            self.assertIs(llm_service.get_llm_client(), client_config)
            self.assertIs(llm_service.get_llm_client(), client_config)
            create_client.assert_called_once_with()

            await llm_service.close_llm_client()

        client.close.assert_awaited_once_with()
        self.assertIsNone(llm_service._llm_client_config)


class ChatCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_choices_raises_expected_error(self) -> None:
        completions = SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(choices=[]))
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with patch.object(
            llm_service,
            "get_llm_client",
            return_value=(client, "deepseek", "deepseek-chat"),
        ):
            with self.assertRaises(llm_service.EmptyModelResponseError):
                await llm_service.chat_completion(ChatRequest(message="hello"))

    async def test_stream_hides_unexpected_error_details(self) -> None:
        with patch.object(
            llm_service,
            "get_llm_client",
            side_effect=ValueError("sensitive internal detail"),
        ):
            with self.assertLogs(llm_service.logger, level="ERROR"):
                events = [
                    event
                    async for event in llm_service.stream_chat_completion(
                        ChatRequest(message="hello")
                    )
                ]

        payload = json.loads(events[0].removeprefix("data: "))
        self.assertEqual(payload, {"type": "error", "message": "服务异常"})
        self.assertNotIn("sensitive internal detail", events[0])


class ChatRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_error_is_sanitized(self) -> None:
        with patch.object(
            chat_router,
            "chat_completion",
            AsyncMock(side_effect=RuntimeError("sensitive internal detail")),
        ):
            with self.assertLogs(chat_router.logger, level="ERROR"):
                with self.assertRaises(HTTPException) as context:
                    await chat_router.chat(ChatRequest(message="hello"))

        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.detail, "模型服务配置错误")

    async def test_success_response_is_unchanged(self) -> None:
        expected = ChatResponse(
            answer="hello",
            conversation_id="conversation-id",
            provider="deepseek",
            model="deepseek-chat",
        )

        with patch.object(
            chat_router,
            "chat_completion",
            AsyncMock(return_value=expected),
        ):
            response = await chat_router.chat(ChatRequest(message="hello"))

        self.assertEqual(response, expected)


class ChatRequestTests(unittest.TestCase):
    def test_message_is_trimmed(self) -> None:
        self.assertEqual(ChatRequest(message="  hello  ").message, "hello")

    def test_whitespace_only_message_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(message="   ")


if __name__ == "__main__":
    unittest.main()
