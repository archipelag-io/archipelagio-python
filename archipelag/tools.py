"""
Tool calling helpers.

Turn Python functions into tools a model can call, and run the agent loop
(model -> tool calls -> results -> model ...) against the chat API.

Example:
    from archipelag import Client, tool

    @tool
    def get_weather(city: str) -> str:
        "Current weather for a city."
        return "4°C and raining"

    client = Client(api_key="ak_xxx")
    result = client.run_agent(
        [{"role": "user", "content": "Weather in Oslo?"}],
        tools=[get_weather],
    )
    print(result.content)
"""

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from archipelag.models import Usage

DEFAULT_MAX_STEPS = 6

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Tool:
    """A function the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def definition(self) -> dict[str, Any]:
        """The OpenAI-compatible tool definition sent to the API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Tool":
        """
        Build a tool from a function's signature and docstring.

        Parameters with type hints `str`, `int`, `float`, `bool`, `list` or
        `dict` (optionally `Optional[...]`) become JSON schema properties;
        parameters without a default are required.
        """
        hints = typing.get_type_hints(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in inspect.signature(fn).parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema: dict[str, Any] = {}
            json_type = _json_type(hints.get(param.name))
            if json_type:
                schema["type"] = json_type
            properties[param.name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(param.name)

        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        doc = inspect.getdoc(fn) or ""
        return cls(
            name=name or fn.__name__,
            description=description or doc.split("\n\n")[0].strip(),
            parameters=parameters,
            function=fn,
        )


def _json_type(hint: Any) -> Optional[str]:
    if hint in _JSON_TYPES:
        return _JSON_TYPES[hint]
    origin = typing.get_origin(hint)
    if origin is Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else None
    if origin in _JSON_TYPES:
        return _JSON_TYPES[origin]
    return None


def tool(
    fn: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """
    Decorator that turns a function into a `Tool`.

    Use as `@tool` or `@tool(name=..., description=...)`.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_function(f, name=name, description=description)

    return wrap(fn) if fn is not None else wrap


@dataclass
class AgentResult:
    """Outcome of `Client.run_agent`."""

    content: str
    """The model's final answer."""
    messages: list[dict[str, Any]]
    """The whole conversation, including tool calls and their results."""
    steps: int
    """How many tool rounds ran."""
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


def as_tool(item: Union[Tool, Callable[..., Any]]) -> Tool:
    """Accept a `Tool` or a plain function."""
    return item if isinstance(item, Tool) else Tool.from_function(item)


def parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """The arguments of a tool call as a dict."""
    raw = call.get("function", {}).get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    args = json.loads(raw)
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return args


def format_result(value: Any) -> str:
    """Tool results are sent to the model as text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def tool_message(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.get("id"),
        "name": call.get("function", {}).get("name"),
        "content": content,
    }


def run_call(tools: dict[str, Tool], call: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call synchronously; errors become the tool's result."""
    name = call.get("function", {}).get("name")
    if name not in tools:
        return tool_message(call, f"Error: unknown tool '{name}'")
    try:
        value = tools[name].function(**parse_arguments(call))
        if inspect.isawaitable(value):
            raise TypeError("async tools need AsyncClient.run_agent")
        return tool_message(call, format_result(value))
    except Exception as exc:  # noqa: BLE001 - the model should see any failure
        return tool_message(call, f"Error: {exc}")


async def arun_call(tools: dict[str, Tool], call: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call, awaiting async tool functions."""
    name = call.get("function", {}).get("name")
    if name not in tools:
        return tool_message(call, f"Error: unknown tool '{name}'")
    try:
        value = tools[name].function(**parse_arguments(call))
        if inspect.isawaitable(value):
            value = await value
        return tool_message(call, format_result(value))
    except Exception as exc:  # noqa: BLE001 - the model should see any failure
        return tool_message(call, f"Error: {exc}")


def add_usage(total: Usage, response: dict[str, Any]) -> Usage:
    usage = response.get("usage") or {}
    return Usage(
        prompt_tokens=(total.prompt_tokens or 0) + (usage.get("prompt_tokens") or 0),
        completion_tokens=(total.completion_tokens or 0) + (usage.get("completion_tokens") or 0),
        total_tokens=(total.total_tokens or 0) + (usage.get("total_tokens") or 0),
    )
