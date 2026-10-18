"""
Archipelag.io Python SDK

Official Python client for the Archipelag.io distributed compute platform.

Example usage:
    from archipelag import Client

    client = Client(api_key="ak_xxx")

    # Synchronous
    result = client.chat("Hello, how are you?")
    print(result.content)

    # Streaming
    for event in client.chat_stream("Tell me a story"):
        if event.type == "token":
            print(event.content, end="", flush=True)

    # Tool calling: the model calls your Python functions
    from archipelag import tool

    @tool
    def get_weather(city: str) -> str:
        "Current weather for a city."
        return "4°C and raining"

    result = client.run_agent(
        [{"role": "user", "content": "Weather in Oslo?"}], tools=[get_weather]
    )
    print(result.content)
"""

from archipelag.client import AsyncClient, Client
from archipelag.exceptions import (
    ArchipelagError,
    AuthenticationError,
    InsufficientCreditsError,
    JobFailedError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from archipelag.models import (
    ApiKey,
    BatchChild,
    BatchConfig,
    BatchJob,
    BatchProgress,
    ChatResult,
    ImageResult,
    Job,
    JobOutput,
    JobStatus,
    StreamEvent,
    Usage,
    Workload,
)
from archipelag.tools import AgentResult, Tool, tool

__version__ = "0.1.0"
__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    # Tool calling
    "AgentResult",
    "Tool",
    "tool",
    # Models
    "BatchChild",
    "BatchConfig",
    "BatchJob",
    "BatchProgress",
    "Job",
    "JobStatus",
    "JobOutput",
    "ChatResult",
    "ImageResult",
    "StreamEvent",
    "Usage",
    "Workload",
    "ApiKey",
    # Exceptions
    "ArchipelagError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "JobFailedError",
    "NotFoundError",
    "ValidationError",
]
