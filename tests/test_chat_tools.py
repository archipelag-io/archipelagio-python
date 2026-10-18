"""
Tests for chat completions, streaming and the tool-calling agent loop.
"""

import json
from typing import Optional

import pytest

from archipelag import AsyncClient, Client, Tool, tool
from archipelag.client import DEFAULT_BASE_URL
from archipelag.exceptions import ArchipelagError, JobFailedError, ValidationError

URL = f"{DEFAULT_BASE_URL}/api/v1/chat/completions"


def completion(message: dict, finish_reason: str = "stop", usage: Optional[dict] = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "auto",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@tool
def get_weather(city: str, unit: Optional[str] = "celsius") -> dict:
    """Current weather for a city.

    Longer notes that should not end up in the description.
    """
    return {"city": city, "temp": 4, "unit": unit}


def sent_json(httpx_mock, index: int = -1) -> dict:
    return json.loads(httpx_mock.get_requests()[index].content)


def test_default_base_url_is_the_working_api_host():
    assert DEFAULT_BASE_URL == "https://app.archipelag.io"


def test_tool_schema_comes_from_the_signature():
    assert isinstance(get_weather, Tool)
    assert get_weather.definition() == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def test_chat_completion_sends_tools_and_params(httpx_mock):
    httpx_mock.add_response(url=URL, json=completion({"role": "assistant", "content": "Hi"}))
    client = Client(api_key="ak_test")

    response = client.chat_completion(
        [{"role": "user", "content": "Hi"}],
        model="gguf-qwen3.5-9b",
        tools=[get_weather],
        temperature=0,
        max_tokens=None,
    )

    assert response["choices"][0]["message"]["content"] == "Hi"
    body = sent_json(httpx_mock)
    assert body["model"] == "gguf-qwen3.5-9b"
    assert body["temperature"] == 0
    assert "max_tokens" not in body
    assert body["tools"][0]["function"]["name"] == "get_weather"


def test_run_agent_runs_tools_and_returns_the_answer(httpx_mock):
    httpx_mock.add_response(
        url=URL,
        json=completion(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "get_weather", {"city": "Oslo"})],
            },
            "tool_calls",
        ),
    )
    httpx_mock.add_response(
        url=URL, json=completion({"role": "assistant", "content": "4°C in Oslo."})
    )
    client = Client(api_key="ak_test")
    seen = []

    result = client.run_agent(
        [{"role": "user", "content": "Weather in Oslo?"}],
        tools=[get_weather],
        on_step=seen.append,
    )

    assert result.content == "4°C in Oslo."
    assert result.steps == 1
    assert result.usage.total_tokens == 30
    assert len(seen) == 2

    second = sent_json(httpx_mock, 1)["messages"]
    assert second[1]["tool_calls"][0]["id"] == "c1"
    assert second[2] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "get_weather",
        "content": json.dumps({"city": "Oslo", "temp": 4, "unit": "celsius"}),
    }
    # The caller's list is left alone
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant"]


def test_plain_functions_work_as_tools_and_errors_reach_the_model(httpx_mock):
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        return a / b

    httpx_mock.add_response(
        url=URL,
        json=completion(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    tool_call("c1", "divide", {"a": 1, "b": 0}),
                    tool_call("c2", "nope", {}),
                ],
            },
            "tool_calls",
        ),
    )
    httpx_mock.add_response(
        url=URL, json=completion({"role": "assistant", "content": "Can't divide by zero."})
    )
    client = Client(api_key="ak_test")

    result = client.run_agent([{"role": "user", "content": "1/0?"}], tools=[divide])

    assert result.content == "Can't divide by zero."
    tool_results = [m for m in sent_json(httpx_mock, 1)["messages"] if m["role"] == "tool"]
    assert tool_results[0]["content"].startswith("Error: ")
    assert tool_results[1]["content"] == "Error: unknown tool 'nope'"


def test_run_agent_stops_after_max_steps(httpx_mock):
    looping = completion(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("c", "get_weather", {"city": "X"})],
        },
        "tool_calls",
    )
    for _ in range(3):
        httpx_mock.add_response(url=URL, json=looping)
    client = Client(api_key="ak_test")

    with pytest.raises(ArchipelagError, match="after 2 rounds"):
        client.run_agent([{"role": "user", "content": "loop"}], tools=[get_weather], max_steps=2)


def sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def chunk(delta: dict, finish_reason: Optional[str] = None) -> dict:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def test_chat_completion_stream_yields_chunks(httpx_mock):
    httpx_mock.add_response(
        url=URL,
        headers={"content-type": "text/event-stream"},
        content=sse(
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, "stop"),
        ),
    )
    client = Client(api_key="ak_test")

    chunks = list(client.chat_completion_stream([{"role": "user", "content": "Hi"}]))

    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert sent_json(httpx_mock)["stream"] is True


def test_stream_errors_raise(httpx_mock):
    httpx_mock.add_response(
        url=URL,
        headers={"content-type": "text/event-stream"},
        content=sse(
            chunk({"content": "Par"}), {"error": {"message": "Job failed: Island disconnected"}}
        ),
    )
    client = Client(api_key="ak_test")

    with pytest.raises(JobFailedError, match="Island disconnected"):
        list(client.chat_completion_stream([{"role": "user", "content": "Hi"}]))


def test_chat_and_chat_stream_use_the_chat_api(httpx_mock):
    httpx_mock.add_response(
        url=URL,
        json=completion(
            {"role": "assistant", "content": "Hello!"},
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        ),
    )
    httpx_mock.add_response(
        url=URL,
        headers={"content-type": "text/event-stream"},
        content=sse(
            chunk({"content": "Hey"}),
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        ),
    )
    client = Client(api_key="ak_test")

    result = client.chat("Hi", system_prompt="Be brief.")
    assert result.content == "Hello!"
    assert result.usage.total_tokens == 5
    assert result.finish_reason == "stop"
    assert sent_json(httpx_mock, 0)["messages"][0] == {"role": "system", "content": "Be brief."}
    assert sent_json(httpx_mock, 0)["model"] == "auto"

    events = list(client.chat_stream("Hi"))
    assert [e.type.value for e in events] == ["token", "done"]
    assert events[0].content == "Hey"
    assert events[1].usage.total_tokens == 4


def test_bad_requests_raise_validation_errors(httpx_mock):
    httpx_mock.add_response(
        url=URL,
        status_code=400,
        json={"error": {"message": "n > 1 is not supported", "type": "invalid_request_error"}},
    )
    client = Client(api_key="ak_test")

    with pytest.raises(ValidationError, match="n > 1"):
        client.chat_completion([{"role": "user", "content": "Hi"}], n=2)


async def test_async_run_agent_awaits_async_tools(httpx_mock):
    @tool
    async def lookup(term: str) -> str:
        """Look something up."""
        return f"found {term}"

    httpx_mock.add_response(
        url=URL,
        json=completion(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    tool_call("a", "lookup", {"term": "x"}),
                    tool_call("b", "lookup", {"term": "y"}),
                ],
            },
            "tool_calls",
        ),
    )
    httpx_mock.add_response(url=URL, json=completion({"role": "assistant", "content": "Done."}))
    client = AsyncClient(api_key="ak_test")

    result = await client.run_agent(
        [{"role": "user", "content": "look up x and y"}], tools=[lookup]
    )

    assert result.content == "Done."
    contents = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert contents == ["found x", "found y"]
    await client.close()
