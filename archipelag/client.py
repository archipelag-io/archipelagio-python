"""
Archipelag SDK Client

Main client classes for interacting with the Archipelag.io API.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Sequence, Union

import httpx
from httpx_sse import EventSource, aconnect_sse, connect_sse

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
    Account,
    ApiKey,
    BatchConfig,
    BatchJob,
    BatchProgress,
    ChatResult,
    ImageResult,
    Job,
    JobStatus,
    StreamEvent,
    StreamEventType,
    Usage,
    Workload,
)
from archipelag.tools import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    Tool,
    add_usage,
    arun_call,
    as_tool,
    run_call,
)

DEFAULT_BASE_URL = "https://app.archipelag.io"
# Servers that only accept JSON reject a bare text/event-stream Accept
STREAM_ACCEPT = "text/event-stream, application/json"
DEFAULT_TIMEOUT = 60.0
# Chat completions wait for an Island to generate the whole answer
CHAT_TIMEOUT = 180.0

ToolLike = Union[Tool, Callable[..., Any], dict[str, Any]]


def _chat_payload(
    messages: list[dict[str, Any]],
    model: str,
    tools: Optional[Sequence[ToolLike]],
    params: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    payload.update({k: v for k, v in params.items() if v is not None})
    if tools:
        payload["tools"] = [t if isinstance(t, dict) else as_tool(t).definition() for t in tools]
    return payload


def _stream_chunk(data: str) -> Optional[dict[str, Any]]:
    """Parse one SSE data line; None at the end of the stream."""
    if data == "[DONE]":
        return None
    chunk: dict[str, Any] = json.loads(data)
    if "error" in chunk:
        raise JobFailedError(chunk["error"].get("message", "Chat completion failed"))
    return chunk


def _chat_result(response: dict[str, Any]) -> ChatResult:
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    return ChatResult(
        content=choice["message"].get("content") or "",
        job_id=response.get("id", ""),
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        ),
        model=response.get("model"),
        finish_reason=choice.get("finish_reason"),
    )


def _simple_messages(prompt: str, system_prompt: Optional[str]) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return messages + [{"role": "user", "content": prompt}]


def _tools_by_name(tools: Sequence[Union[Tool, Callable[..., Any]]]) -> dict[str, Tool]:
    resolved = [as_tool(t) for t in tools]
    return {t.name: t for t in resolved}


class Client:
    """
    Synchronous client for the Archipelag.io API.

    Example:
        client = Client(api_key="ak_xxx")
        result = client.chat("Hello!")
        print(result.content)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the client.

        Args:
            api_key: Your Archipelag API key (starts with 'ak_')
            base_url: API base URL (default: https://app.archipelag.io)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "archipelag-python/0.2.0",
            },
            timeout=timeout,
        )

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        if response.status_code == 200 or response.status_code == 201:
            data: dict[str, Any] = response.json()
            return data

        try:
            error_data = response.json()
            message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, status_code=401)
        elif response.status_code == 402:
            raise InsufficientCreditsError(message, status_code=402)
        elif response.status_code == 404:
            raise NotFoundError(message, status_code=404)
        elif response.status_code in (400, 422):
            errors = error_data.get("errors") if "error_data" in dir() else None
            raise ValidationError(message, errors=errors, status_code=422)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                status_code=429,
            )
        else:
            raise ArchipelagError(message, status_code=response.status_code)

    # =========================================================================
    # Account
    # =========================================================================

    def get_account(self) -> Account:
        """Get current account information."""
        response = self._client.get("/api/v1/account")
        data = self._handle_response(response)
        return Account(**data["data"])

    # =========================================================================
    # Jobs
    # =========================================================================

    def create_job(self, workload: str, input: dict[str, Any]) -> Job:
        """
        Create a new job.

        Args:
            workload: Workload slug (e.g., "llm-chat", "sdxl")
            input: Job input parameters

        Returns:
            The created job
        """
        response = self._client.post(
            "/api/v1/jobs",
            json={"workload": workload, "input": input},
        )
        data = self._handle_response(response)
        return Job(**data["data"])

    def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        response = self._client.get(f"/api/v1/jobs/{job_id}")
        data = self._handle_response(response)
        return Job(**data["data"])

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[Job]:
        """List recent jobs."""
        response = self._client.get(
            "/api/v1/jobs",
            params={"limit": limit, "offset": offset},
        )
        data = self._handle_response(response)
        return [Job(**job) for job in data["data"]]

    def cancel_job(self, job_id: str) -> Job:
        """Cancel a running job."""
        response = self._client.delete(f"/api/v1/jobs/{job_id}")
        data = self._handle_response(response)
        return Job(**data["data"])

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Job:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = no timeout)

        Returns:
            The completed job

        Raises:
            JobFailedError: If the job fails
            TimeoutError: If timeout is exceeded
        """
        start = time.time()
        while True:
            job = self.get_job(job_id)
            if job.is_complete:
                if job.status == JobStatus.FAILED:
                    raise JobFailedError(
                        job.error or "Job failed",
                        job_id=job_id,
                    )
                return job

            if timeout and (time.time() - start) > timeout:
                from archipelag.exceptions import TimeoutError

                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            time.sleep(poll_interval)

    def stream_job(self, job_id: str) -> Iterator[StreamEvent]:
        """
        Stream output from a job.

        Args:
            job_id: Job ID to stream

        Yields:
            StreamEvent objects
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}/stream"
        with connect_sse(
            self._client,
            "GET",
            url,
        ) as event_source:
            for sse in event_source.iter_sse():
                yield self._parse_stream_event(sse.data)

    def _parse_stream_event(self, data: str) -> StreamEvent:
        """Parse SSE data into StreamEvent."""
        import json

        parsed = json.loads(data)
        event_type = parsed.get("type", "token")

        return StreamEvent(
            type=StreamEventType(event_type),
            content=parsed.get("content") or parsed.get("chunk"),
            step=parsed.get("step"),
            total=parsed.get("total"),
            image_data=parsed.get("image_data"),
            image_format=parsed.get("format"),
            error=parsed.get("error"),
            usage=Usage(**parsed["usage"]) if "usage" in parsed else None,
        )

    # =========================================================================
    # Workloads
    # =========================================================================

    def list_workloads(self) -> list[Workload]:
        """List available workloads."""
        response = self._client.get("/api/v1/workloads")
        data = self._handle_response(response)
        return [Workload(**w) for w in data["data"]]

    def get_workload(self, slug: str) -> Workload:
        """Get workload by slug."""
        response = self._client.get(f"/api/v1/workloads/{slug}")
        data = self._handle_response(response)
        return Workload(**data["data"])

    # =========================================================================
    # API Keys
    # =========================================================================

    def list_api_keys(self) -> list[ApiKey]:
        """List API keys."""
        response = self._client.get("/api/v1/api-keys")
        data = self._handle_response(response)
        return [ApiKey(**k) for k in data["data"]]

    def create_api_key(self, name: str) -> tuple[ApiKey, str]:
        """
        Create a new API key.

        Returns:
            Tuple of (ApiKey info, full key string)
            Note: The full key is only returned once!
        """
        response = self._client.post("/api/v1/api-keys", json={"name": name})
        data = self._handle_response(response)
        return ApiKey(**data["data"]), data["key"]

    def delete_api_key(self, key_id: str) -> None:
        """Delete an API key."""
        response = self._client.delete(f"/api/v1/api-keys/{key_id}")
        self._handle_response(response)

    # =========================================================================
    # High-level helpers (convenience methods)
    # =========================================================================

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "auto",
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        **params: Any,
    ) -> dict[str, Any]:
        """
        Create a chat completion (OpenAI-compatible).

        Args:
            messages: OpenAI-style messages, including `tool` results
            model: A Cargo slug, an OpenAI alias, or "auto"
            tools: `Tool`s, plain functions, or OpenAI tool definitions
            **params: Any other OpenAI parameter (temperature, max_tokens,
                tool_choice, response_format, seed, stop, enable_thinking...)

        Returns:
            The OpenAI-shaped response (`choices[0].message` may hold
            `tool_calls`)
        """
        payload = _chat_payload(messages, model, tools, params)
        response = self._client.post(
            "/api/v1/chat/completions", json=payload, timeout=max(self.timeout, CHAT_TIMEOUT)
        )
        return self._handle_response(response)

    def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "auto",
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream a chat completion as OpenAI `chat.completion.chunk` dicts.

        Raises:
            JobFailedError: if the Island fails mid-stream
        """
        payload = _chat_payload(messages, model, tools, params)
        payload["stream"] = True
        with self._client.stream(
            "POST",
            "/api/v1/chat/completions",
            json=payload,
            headers={"Accept": STREAM_ACCEPT, "Cache-Control": "no-store"},
            timeout=max(self.timeout, CHAT_TIMEOUT),
        ) as response:
            if response.status_code != 200:
                response.read()
                self._handle_response(response)
            for sse in EventSource(response).iter_sse():
                chunk = _stream_chunk(sse.data)
                if chunk is None:
                    break
                yield chunk

    def run_agent(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Union[Tool, Callable[..., Any]]],
        model: str = "auto",
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Optional[Callable[[dict[str, Any]], None]] = None,
        **params: Any,
    ) -> AgentResult:
        """
        Run a tool-using conversation until the model answers.

        Each round sends the conversation with the tools; when the model
        asks for tools they run locally and their results go back to it.

        Args:
            messages: The conversation so far (not modified)
            tools: `Tool`s or plain functions (see `archipelag.tool`)
            model: A Cargo slug or "auto"
            max_steps: Most tool rounds before giving up
            on_step: Called with each assistant message as it arrives
            **params: Passed to `chat_completion`

        Returns:
            AgentResult with the final answer and the full conversation
        """
        by_name = _tools_by_name(tools)
        conversation = list(messages)
        usage = Usage()

        for step in range(max_steps + 1):
            response = self.chat_completion(
                conversation, model, tools=list(by_name.values()), **params
            )
            usage = add_usage(usage, response)
            choice = response["choices"][0]
            message = choice["message"]
            conversation.append(message)
            if on_step:
                on_step(message)

            calls = message.get("tool_calls") or []
            if not calls:
                return AgentResult(
                    content=message.get("content") or "",
                    messages=conversation,
                    steps=step,
                    finish_reason=choice.get("finish_reason"),
                    usage=usage,
                )
            if step == max_steps:
                break
            conversation.extend(run_call(by_name, call) for call in calls)

        raise ArchipelagError(f"Model still calling tools after {max_steps} rounds")

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        workload: str = "auto",
    ) -> ChatResult:
        """
        Send a chat message and get a response.

        Args:
            prompt: User message
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            workload: Cargo slug to use, or "auto" (default)

        Returns:
            ChatResult with the response
        """
        response = self.chat_completion(
            _simple_messages(prompt, system_prompt),
            workload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _chat_result(response)

    def chat_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        workload: str = "auto",
    ) -> Iterator[StreamEvent]:
        """
        Send a chat message and stream the response.

        Args:
            prompt: User message
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            workload: Cargo slug to use, or "auto" (default)

        Yields:
            StreamEvent objects: TOKEN for each piece of text, then DONE
        """
        usage: Optional[Usage] = None
        for chunk in self.chat_completion_stream(
            _simple_messages(prompt, system_prompt),
            workload,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_options={"include_usage": True},
        ):
            if chunk.get("usage"):
                usage = Usage(**chunk["usage"])
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    yield StreamEvent(type=StreamEventType.TOKEN, content=content)
        yield StreamEvent(type=StreamEventType.DONE, usage=usage)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        workload: str = "sdxl",
    ) -> ImageResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of diffusion steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            workload: Workload to use (default: sdxl)

        Returns:
            ImageResult with base64-encoded image data
        """
        input_data: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        if seed is not None:
            input_data["seed"] = seed

        job = self.create_job(workload, input_data)
        job = self.wait_for_job(job.id)

        # Parse image output
        import json

        output = json.loads(job.output or "{}")

        return ImageResult(
            image_data=output.get("image_data", ""),
            image_format=output.get("format", "png"),
            width=output.get("width", width),
            height=output.get("height", height),
            seed=output.get("seed"),
            job_id=job.id,
            usage=job.usage or Usage(),
        )

    def batch(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[Job]:
        """
        Create multiple jobs at once.

        Args:
            jobs: List of job specs, each with 'workload' and 'input' keys

        Returns:
            List of created jobs
        """
        created = []
        for job_spec in jobs:
            job = self.create_job(
                workload=job_spec["workload"],
                input=job_spec["input"],
            )
            created.append(job)
        return created

    def wait_all(
        self,
        jobs: list[Job],
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> list[Job]:
        """
        Wait for multiple jobs to complete.

        Args:
            jobs: List of jobs to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = no timeout)

        Returns:
            List of completed jobs
        """
        results = []
        for job in jobs:
            completed = self.wait_for_job(
                job.id,
                poll_interval=poll_interval,
                timeout=timeout,
            )
            results.append(completed)
        return results

    # =========================================================================
    # Server-side batch jobs
    # =========================================================================

    def submit_batch(
        self,
        workload: str,
        inputs: list[dict[str, Any]],
        *,
        merge_strategy: str = "concat",
        fail_mode: str = "best_effort",
        max_parallelism: Optional[int] = None,
        region: Optional[str] = None,
        bid_price: Optional[float] = None,
    ) -> BatchJob:
        """
        Submit a batch job to be distributed across multiple Islands.

        Args:
            workload: Cargo slug to run for each input
            inputs: List of input dicts (1-100 items)
            merge_strategy: How to merge results - "concat" or "flatten"
            fail_mode: "best_effort" (partial results OK) or "fail_fast"
            max_parallelism: Max concurrent children (default: unlimited)
            region: Preferred region for placement
            bid_price: Per-child bid price

        Returns:
            BatchJob with parent ID and child job list
        """
        body: dict[str, Any] = {
            "workload": workload,
            "inputs": inputs,
            "merge_strategy": merge_strategy,
            "fail_mode": fail_mode,
        }
        if max_parallelism is not None:
            body["max_parallelism"] = max_parallelism
        if region is not None:
            body["region"] = region
        if bid_price is not None:
            body["bid_price"] = str(bid_price)

        response = self._client.post("/api/v1/jobs/batch", json=body)
        data = self._handle_response(response)
        return BatchJob(**data)

    def get_batch_status(self, batch_id: str) -> BatchProgress:
        """Get detailed progress of a batch job."""
        response = self._client.get(f"/api/v1/jobs/{batch_id}/batch-status")
        data = self._handle_response(response)
        return BatchProgress(**data)

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> BatchJob:
        """
        Wait for a batch job to complete, with optional progress callback.

        Args:
            batch_id: Parent batch job ID
            poll_interval: Seconds between status checks (default: 2)
            timeout: Maximum seconds to wait
            on_progress: Called with (completed, failed, total) on each poll

        Returns:
            Completed BatchJob (poll get_job for output)
        """
        start = time.time()
        while True:
            progress = self.get_batch_status(batch_id)

            if on_progress:
                completed = progress.child_states.get("succeeded", 0)
                failed = progress.child_states.get("failed", 0)
                on_progress(completed, failed, progress.chunk_count)

            if progress.parent_state in ("succeeded", "failed", "cancelled"):
                job = self.get_job(batch_id)
                if job.status == JobStatus.FAILED:
                    raise JobFailedError(job.error or "Batch failed", job_id=batch_id)
                return BatchJob(
                    id=batch_id,
                    state=progress.parent_state,
                    workload="",
                    batch=BatchConfig(
                        chunk_count=progress.chunk_count,
                        merge_strategy=progress.merge_strategy,
                        fail_mode=progress.fail_mode,
                    ),
                    children=progress.children,
                    created_at=datetime.min,
                )

            if timeout and (time.time() - start) > timeout:
                from archipelag.exceptions import TimeoutError

                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

            time.sleep(poll_interval)


class AsyncClient:
    """
    Asynchronous client for the Archipelag.io API.

    Example:
        async with AsyncClient(api_key="ak_xxx") as client:
            result = await client.chat("Hello!")
            print(result.content)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the async client.

        Args:
            api_key: Your Archipelag API key (starts with 'ak_')
            base_url: API base URL (default: https://app.archipelag.io)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "archipelag-python/0.2.0",
            },
            timeout=timeout,
        )

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        if response.status_code == 200 or response.status_code == 201:
            data: dict[str, Any] = response.json()
            return data

        try:
            error_data = response.json()
            message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, status_code=401)
        elif response.status_code == 402:
            raise InsufficientCreditsError(message, status_code=402)
        elif response.status_code == 404:
            raise NotFoundError(message, status_code=404)
        elif response.status_code in (400, 422):
            errors = error_data.get("errors") if "error_data" in dir() else None
            raise ValidationError(message, errors=errors, status_code=422)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                retry_after=int(retry_after) if retry_after else None,
                status_code=429,
            )
        else:
            raise ArchipelagError(message, status_code=response.status_code)

    # =========================================================================
    # Jobs
    # =========================================================================

    async def create_job(self, workload: str, input: dict[str, Any]) -> Job:
        """Create a new job."""
        response = await self._client.post(
            "/api/v1/jobs",
            json={"workload": workload, "input": input},
        )
        data = await self._handle_response(response)
        return Job(**data["data"])

    async def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        response = await self._client.get(f"/api/v1/jobs/{job_id}")
        data = await self._handle_response(response)
        return Job(**data["data"])

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Job:
        """Wait for a job to complete."""
        import asyncio

        start = time.time()
        while True:
            job = await self.get_job(job_id)
            if job.is_complete:
                if job.status == JobStatus.FAILED:
                    raise JobFailedError(
                        job.error or "Job failed",
                        job_id=job_id,
                    )
                return job

            if timeout and (time.time() - start) > timeout:
                from archipelag.exceptions import TimeoutError

                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            await asyncio.sleep(poll_interval)

    async def stream_job(self, job_id: str) -> AsyncIterator[StreamEvent]:
        """Stream output from a job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/stream"
        async with aconnect_sse(
            self._client,
            "GET",
            url,
        ) as event_source:
            async for sse in event_source.aiter_sse():
                yield self._parse_stream_event(sse.data)

    def _parse_stream_event(self, data: str) -> StreamEvent:
        """Parse SSE data into StreamEvent."""
        import json

        parsed = json.loads(data)
        event_type = parsed.get("type", "token")

        return StreamEvent(
            type=StreamEventType(event_type),
            content=parsed.get("content") or parsed.get("chunk"),
            step=parsed.get("step"),
            total=parsed.get("total"),
            image_data=parsed.get("image_data"),
            image_format=parsed.get("format"),
            error=parsed.get("error"),
            usage=Usage(**parsed["usage"]) if "usage" in parsed else None,
        )

    # =========================================================================
    # High-level helpers
    # =========================================================================

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "auto",
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Create a chat completion (OpenAI-compatible). See `Client.chat_completion`."""
        payload = _chat_payload(messages, model, tools, params)
        response = await self._client.post(
            "/api/v1/chat/completions", json=payload, timeout=max(self.timeout, CHAT_TIMEOUT)
        )
        return await self._handle_response(response)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "auto",
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        **params: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion as OpenAI chunk dicts. See `Client.chat_completion_stream`."""
        payload = _chat_payload(messages, model, tools, params)
        payload["stream"] = True
        async with self._client.stream(
            "POST",
            "/api/v1/chat/completions",
            json=payload,
            headers={"Accept": STREAM_ACCEPT, "Cache-Control": "no-store"},
            timeout=max(self.timeout, CHAT_TIMEOUT),
        ) as response:
            if response.status_code != 200:
                await response.aread()
                await self._handle_response(response)
            async for sse in EventSource(response).aiter_sse():
                chunk = _stream_chunk(sse.data)
                if chunk is None:
                    break
                yield chunk

    async def run_agent(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Union[Tool, Callable[..., Any]]],
        model: str = "auto",
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Optional[Callable[[dict[str, Any]], None]] = None,
        **params: Any,
    ) -> AgentResult:
        """
        Run a tool-using conversation until the model answers. Tool
        functions may be async; a round's tool calls run concurrently.
        See `Client.run_agent`.
        """
        by_name = _tools_by_name(tools)
        conversation = list(messages)
        usage = Usage()

        for step in range(max_steps + 1):
            response = await self.chat_completion(
                conversation, model, tools=list(by_name.values()), **params
            )
            usage = add_usage(usage, response)
            choice = response["choices"][0]
            message = choice["message"]
            conversation.append(message)
            if on_step:
                on_step(message)

            calls = message.get("tool_calls") or []
            if not calls:
                return AgentResult(
                    content=message.get("content") or "",
                    messages=conversation,
                    steps=step,
                    finish_reason=choice.get("finish_reason"),
                    usage=usage,
                )
            if step == max_steps:
                break
            results = await asyncio.gather(*(arun_call(by_name, call) for call in calls))
            conversation.extend(results)

        raise ArchipelagError(f"Model still calling tools after {max_steps} rounds")

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        workload: str = "auto",
    ) -> ChatResult:
        """Send a chat message and get a response."""
        response = await self.chat_completion(
            _simple_messages(prompt, system_prompt),
            workload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _chat_result(response)

    async def chat_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        workload: str = "auto",
    ) -> AsyncIterator[StreamEvent]:
        """Send a chat message and stream the response (TOKEN events, then DONE)."""
        usage: Optional[Usage] = None
        async for chunk in self.chat_completion_stream(
            _simple_messages(prompt, system_prompt),
            workload,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_options={"include_usage": True},
        ):
            if chunk.get("usage"):
                usage = Usage(**chunk["usage"])
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    yield StreamEvent(type=StreamEventType.TOKEN, content=content)
        yield StreamEvent(type=StreamEventType.DONE, usage=usage)

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        workload: str = "sdxl",
    ) -> ImageResult:
        """Generate an image from a text prompt."""
        input_data: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        if seed is not None:
            input_data["seed"] = seed

        job = await self.create_job(workload, input_data)
        job = await self.wait_for_job(job.id)

        # Parse image output
        import json

        output = json.loads(job.output or "{}")

        return ImageResult(
            image_data=output.get("image_data", ""),
            image_format=output.get("format", "png"),
            width=output.get("width", width),
            height=output.get("height", height),
            seed=output.get("seed"),
            job_id=job.id,
            usage=job.usage or Usage(),
        )

    # =========================================================================
    # Server-side batch jobs
    # =========================================================================

    async def submit_batch(
        self,
        workload: str,
        inputs: list[dict[str, Any]],
        *,
        merge_strategy: str = "concat",
        fail_mode: str = "best_effort",
        max_parallelism: Optional[int] = None,
        region: Optional[str] = None,
        bid_price: Optional[float] = None,
    ) -> BatchJob:
        """
        Submit a batch job to be distributed across multiple Islands.

        Args:
            workload: Cargo slug to run for each input
            inputs: List of input dicts (1-100 items)
            merge_strategy: How to merge results - "concat" or "flatten"
            fail_mode: "best_effort" (partial results OK) or "fail_fast"
            max_parallelism: Max concurrent children (default: unlimited)
            region: Preferred region for placement
            bid_price: Per-child bid price

        Returns:
            BatchJob with parent ID and child job list
        """
        body: dict[str, Any] = {
            "workload": workload,
            "inputs": inputs,
            "merge_strategy": merge_strategy,
            "fail_mode": fail_mode,
        }
        if max_parallelism is not None:
            body["max_parallelism"] = max_parallelism
        if region is not None:
            body["region"] = region
        if bid_price is not None:
            body["bid_price"] = str(bid_price)

        response = await self._client.post("/api/v1/jobs/batch", json=body)
        data = await self._handle_response(response)
        return BatchJob(**data)

    async def get_batch_status(self, batch_id: str) -> BatchProgress:
        """Get detailed progress of a batch job."""
        response = await self._client.get(f"/api/v1/jobs/{batch_id}/batch-status")
        data = await self._handle_response(response)
        return BatchProgress(**data)

    async def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> BatchJob:
        """
        Wait for a batch job to complete, with optional progress callback.

        Args:
            batch_id: Parent batch job ID
            poll_interval: Seconds between status checks (default: 2)
            timeout: Maximum seconds to wait
            on_progress: Called with (completed, failed, total) on each poll

        Returns:
            Completed BatchJob (poll get_job for output)
        """
        import asyncio

        start = time.time()
        while True:
            progress = await self.get_batch_status(batch_id)

            if on_progress:
                completed = progress.child_states.get("succeeded", 0)
                failed = progress.child_states.get("failed", 0)
                on_progress(completed, failed, progress.chunk_count)

            if progress.parent_state in ("succeeded", "failed", "cancelled"):
                job = await self.get_job(batch_id)
                if job.status == JobStatus.FAILED:
                    raise JobFailedError(job.error or "Batch failed", job_id=batch_id)
                return BatchJob(
                    id=batch_id,
                    state=progress.parent_state,
                    workload="",
                    batch=BatchConfig(
                        chunk_count=progress.chunk_count,
                        merge_strategy=progress.merge_strategy,
                        fail_mode=progress.fail_mode,
                    ),
                    children=progress.children,
                    created_at=datetime.min,
                )

            if timeout and (time.time() - start) > timeout:
                from archipelag.exceptions import TimeoutError

                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

            await asyncio.sleep(poll_interval)
