# llm-api-starter

A minimal FastAPI backend for calling LLM chat APIs. It supports DeepSeek, OpenRouter, standard responses, and SSE streaming responses.

## Project Structure

```text
.
├── app/
│   ├── core/
│   │   └── config.py       # Environment configuration
│   ├── routers/
│   │   └── chat.py         # Chat API routes and HTTP error mapping
│   ├── schemas/
│   │   └── chat.py         # Request and response models
│   ├── services/
│   │   └── llm_service.py  # Provider selection, client lifecycle, and LLM calls
│   └── main.py             # FastAPI application, lifecycle, and router registration
├── tests/
│   └── test_chat.py        # Chat, validation, error, and lifecycle tests
├── .env.example       # Environment variable example
├── .gitignore
├── README.md
└── requirements.txt   # Python dependencies
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Supported `LLM_PROVIDER` values:

| Value | Description |
| --- | --- |
| `deepseek` | Calls the DeepSeek API directly. This is the default when the variable is not set. |
| `openrouter` | Calls a model through OpenRouter. |

### DeepSeek

Add the following values to `.env`:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
```

- `DEEPSEEK_API_KEY`: Required. Your DeepSeek API key.
- `DEEPSEEK_MODEL`: Optional. Defaults to `deepseek-chat`.

### OpenRouter

Add the following values to `.env`:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=deepseek/deepseek-chat
OPENROUTER_SITE_URL=http://localhost:8000
OPENROUTER_APP_NAME=LLM API Starter
```

- `OPENROUTER_API_KEY`: Required. Your OpenRouter API key.
- `OPENROUTER_MODEL`: Optional. Defaults to `deepseek/deepseek-chat`.
- `OPENROUTER_SITE_URL`: Optional. Sent to OpenRouter as the `HTTP-Referer` header. Defaults to `http://localhost:8000`.
- `OPENROUTER_APP_NAME`: Optional. Sent to OpenRouter as the `X-Title` header. Defaults to `LLM API Starter`.

## Running the Application

```bash
uvicorn app.main:app --reload
```

After startup, the following URLs are available:

- Service: <http://127.0.0.1:8000>
- Swagger API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Standard Chat Endpoint

`POST /chat` returns the complete answer after the model finishes generating it.

Example request:

```json
{
  "message": "Describe FastAPI in one sentence.",
  "conversation_id": "optional-conversation-id"
}
```

- `message`: Required. Must contain between 1 and 2,000 characters.
- Leading and trailing whitespace is removed. A message containing only whitespace is rejected with HTTP `422`.
- `conversation_id`: Optional. The server generates a UUID when this field is omitted. It currently identifies the request only; the server does not automatically store or load conversation history.

Example response:

```json
{
  "answer": "FastAPI is a modern, high-performance Python web framework built around type hints.",
  "conversation_id": "optional-conversation-id",
  "provider": "deepseek",
  "model": "deepseek-chat"
}
```

Test with curl:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Describe FastAPI in one sentence."}'
```

## Streaming Chat Endpoint

`POST /chat/stream` returns `text/event-stream`. Each SSE event consists of a `data:` line containing JSON, followed by a blank line.

Events are returned in this order:

1. `meta`: Provides the conversation, provider, and model information.
2. `delta`: Provides one chunk of generated text. This event may occur zero or more times.
3. `done`: Indicates that the stream completed normally.

Example:

```text
data: {"type":"meta","conversation_id":"7e23d71f-89b4-4be8-a41b-91f17a274d21","provider":"deepseek","model":"deepseek-chat"}

data: {"type":"delta","content":"FastAPI"}

data: {"type":"delta","content":" is a modern Python web framework."}

data: {"type":"done"}

```

If the request fails, the stream returns an `error` event and then closes:

```text
data: {"type":"error","message":"模型请求超时"}

```

Error messages returned to clients are intentionally stable and do not include raw provider or internal exception details. Detailed exceptions are written to the server logs.

Use `-N` to disable curl response buffering and display events as they arrive:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Describe FastAPI in three points."}'
```

## Runtime Behavior

The LLM client is created lazily on the first chat request and reused by later requests so its connection pool can be shared. FastAPI closes the client during application shutdown. This means the service and `/health` can start without an API key, while chat requests report a configuration error until the selected provider is configured.

The API returns sanitized errors:

| Scenario | Standard endpoint | Streaming endpoint |
| --- | --- | --- |
| Provider timeout | HTTP `504` | `error` event with `模型请求超时` |
| Provider connection failure | HTTP `502` | `error` event with `无法连接到模型服务` |
| Provider API failure | HTTP `502` | `error` event with `模型服务错误` |
| Missing or invalid provider configuration | HTTP `500` | `error` event with `模型服务配置错误` |

## Testing

The tests use Python's standard-library `unittest` module and mock all LLM calls, so they do not require an API key or send requests to DeepSeek or OpenRouter.

```bash
python -m unittest discover -v
```
