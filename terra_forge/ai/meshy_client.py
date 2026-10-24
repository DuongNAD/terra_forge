"""
Meshy AI v2 Text-to-3D Client for TerraForge.
Provides an asynchronous, rate-limited, and resilient HTTP client for Meshy v2 text-to-3d API
(https://api.meshy.ai/openapi/v2/text-to-3d).

Features:
- Safe API key resolution (environment variable MESHY_API_KEY with default fallback).
- Plaintext secret masking in logs, repr, and exceptions.
- Task creation for preview (geometry) and refine (PBR texturing).
- Polling loop with exponential backoff and jitter.
- Transient error retry on HTTP 429 (Rate Limit) and 5xx responses.
- Asynchronous model and thumbnail downloading.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

logger = logging.getLogger("terra_forge.ai.meshy_client")

DEFAULT_MESHY_API_KEY = "msy_yFOKAOFOk9yjKUcwuk9rlkb0lTyuWMI0sT1T"
DEFAULT_BASE_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"


def mask_api_key(key: Optional[str]) -> str:
    """Return a masked representation of an API key for safe logging."""
    if not key:
        return "<not-set>"
    clean = key.strip()
    if len(clean) <= 8:
        return "***"
    return f"{clean[:4]}***...{clean[-4:]}"


def resolve_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve the Meshy API key securely.
    Priority:
    1. Explicitly passed parameter (if non-empty).
    2. MESHY_API_KEY environment variable.
    3. Default embedded project key.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = os.environ.get("MESHY_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return DEFAULT_MESHY_API_KEY


class MeshyTaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"

    @classmethod
    def from_str(cls, value: str) -> MeshyTaskStatus:
        try:
            return cls(value.upper())
        except ValueError:
            return cls.PENDING


@dataclass
class MeshyTaskResult:
    """Structured representation of a Meshy Text-to-3D task result."""
    task_id: str
    mode: str
    status: MeshyTaskStatus
    progress: int = 0
    model_urls: Dict[str, str] = field(default_factory=dict)
    thumbnail_url: Optional[str] = None
    texture_urls: List[Dict[str, str]] = field(default_factory=list)
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    task_error: Optional[str] = None
    consumed_credits: int = 0
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def glb_url(self) -> Optional[str]:
        return self.model_urls.get("glb")

    @property
    def is_complete(self) -> bool:
        return self.status == MeshyTaskStatus.SUCCEEDED

    @property
    def is_failed(self) -> bool:
        return self.status in (MeshyTaskStatus.FAILED, MeshyTaskStatus.EXPIRED)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeshyTaskResult:
        task_id = data.get("id") or data.get("result") or ""
        mode = data.get("mode") or "preview"
        status_str = data.get("status") or "PENDING"
        status = MeshyTaskStatus.from_str(status_str)
        progress = int(data.get("progress", 0) or 0)
        model_urls = data.get("model_urls") or {}
        thumbnail_url = data.get("thumbnail_url")
        texture_urls = data.get("texture_urls") or []
        started_at = data.get("started_at")
        finished_at = data.get("finished_at")
        task_error = None
        if data.get("task_error"):
            err = data["task_error"]
            task_error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        consumed_credits = int(data.get("consumed_credits", 0) or 0)

        return cls(
            task_id=task_id,
            mode=mode,
            status=status,
            progress=progress,
            model_urls=model_urls,
            thumbnail_url=thumbnail_url,
            texture_urls=texture_urls,
            started_at=started_at,
            finished_at=finished_at,
            task_error=task_error,
            consumed_credits=consumed_credits,
            raw_response=data,
        )


class MeshyClient:
    """
    Asynchronous client for Meshy v2 text-to-3d API.
    Handles rate-limiting, exponential backoff, safe API key resolution, and lifecycle polling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        timeout: float = 900.0,
    ):
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self._external_client = client is not None
        self._client = client

    def __repr__(self) -> str:
        return f"MeshyClient(base_url={self.base_url!r}, api_key={mask_api_key(self.api_key)!r})"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=30.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "TerraForge-MeshyClient/0.1.0",
                },
            )
        return self._client

    async def close(self) -> None:
        if not self._external_client and self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> MeshyClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        client = await self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
                if response.status_code == 429:
                    wait_time = (self.retry_backoff * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Meshy API rate limited (429). Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                elif response.status_code in (500, 502, 503, 504):
                    wait_time = (self.retry_backoff * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Meshy API transient error ({response.status_code}). Retrying in {wait_time:.2f}s."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                if attempt < self.max_retries:
                    wait_time = (self.retry_backoff * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    logger.warning(f"Network exception: {e}. Retrying in {wait_time:.2f}s.")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        raise RuntimeError(f"Exceeded maximum retries ({self.max_retries}) for request {method} {url}")

    async def create_preview_task(
        self,
        prompt: str,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        art_style: str = "realistic",
        ai_model: str = "meshy-6",
        topology: str = "triangle",
        target_polycount: int = 12000,
        should_remesh: bool = True,
        **kwargs,
    ) -> str:
        """
        Submit a new text-to-3d preview task to Meshy v2 API.
        Returns the generated task_id string.
        """
        payload: Dict[str, Any] = {
            "mode": "preview",
            "prompt": prompt.strip(),
            "ai_model": ai_model,
            "topology": topology,
            "target_polycount": target_polycount,
            "should_remesh": should_remesh,
        }
        if ai_model != "meshy-6":
            payload["art_style"] = art_style
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt.strip()
            else:
                payload["negative_prompt"] = (
                    "human artifacts, saw cuts, modern tools, chopped timber, lumber planks, "
                    "geometric architecture, low quality, cartoon, flat shading"
                )
        if seed is not None:
            payload["seed"] = int(seed)
        payload.update(kwargs)

        logger.info(f"Creating Meshy preview task for prompt: {prompt[:60]!r}...")
        response = await self._request("POST", self.base_url, json=payload)
        data = response.json()
        task_id = data.get("result") or data.get("id")
        if not task_id:
            raise ValueError(f"Unexpected Meshy response, missing task_id: {data}")
        logger.info(f"Created preview task: {task_id}")
        return str(task_id)

    async def create_refine_task(
        self,
        preview_task_id: str,
        texture_richness: str = "high",
        **kwargs,
    ) -> str:
        """
        Submit a refine task for PBR texturing based on a completed preview task.
        Returns the new refine task_id string.
        """
        payload: Dict[str, Any] = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "texture_richness": texture_richness,
        }
        payload.update(kwargs)

        logger.info(f"Creating Meshy refine task for preview_task_id: {preview_task_id}")
        response = await self._request("POST", self.base_url, json=payload)
        data = response.json()
        task_id = data.get("result") or data.get("id")
        if not task_id:
            raise ValueError(f"Unexpected Meshy response, missing refine task_id: {data}")
        logger.info(f"Created refine task: {task_id}")
        return str(task_id)

    async def get_task_status(self, task_id: str) -> MeshyTaskResult:
        """Query the current status of a task."""
        url = f"{self.base_url}/{task_id}"
        response = await self._request("GET", url)
        return MeshyTaskResult.from_dict(response.json())

    async def poll_task(
        self,
        task_id: str,
        poll_interval: float = 3.0,
        max_interval: float = 10.0,
        backoff_factor: float = 1.5,
        timeout: float = 900.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> MeshyTaskResult:
        """
        Poll a task until it completes, fails, or times out.
        Applies exponential backoff between poll attempts up to max_interval.
        """
        start_time = time.monotonic()
        current_interval = poll_interval

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Meshy task {task_id} timed out after {elapsed:.1f}s")

            result = await self.get_task_status(task_id)

            if progress_callback:
                progress_callback(result.progress, result.status.value)

            if result.status == MeshyTaskStatus.SUCCEEDED:
                logger.info(f"Meshy task {task_id} SUCCEEDED in {elapsed:.1f}s")
                return result
            elif result.status in (MeshyTaskStatus.FAILED, MeshyTaskStatus.EXPIRED):
                err_msg = result.task_error or f"Task {task_id} status: {result.status.value}"
                logger.error(f"Meshy task failed: {err_msg}")
                raise RuntimeError(f"Meshy task failed: {err_msg}")

            sleep_duration = current_interval + random.uniform(0.0, 0.5)
            await asyncio.sleep(sleep_duration)
            current_interval = min(max_interval, current_interval * backoff_factor)

    async def generate_text_to_3d(
        self,
        prompt: str,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        art_style: str = "realistic",
        refine: bool = False,
        texture_richness: str = "high",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs,
    ) -> MeshyTaskResult:
        """
        High-level pipeline: submit preview task, poll to completion, and optionally refine.
        """
        preview_id = await self.create_preview_task(
            prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            art_style=art_style,
            **kwargs,
        )
        preview_result = await self.poll_task(
            task_id=preview_id,
            progress_callback=progress_callback,
        )

        if not refine:
            return preview_result

        logger.info(f"Proceeding to refine phase for preview task {preview_id}...")
        refine_id = await self.create_refine_task(
            preview_task_id=preview_id,
            texture_richness=texture_richness,
        )
        refine_result = await self.poll_task(
            task_id=refine_id,
            progress_callback=progress_callback,
        )
        return refine_result

    async def download_model(
        self,
        url: str,
        output_path: Union[str, Path],
    ) -> Path:
        """Download a model asset (e.g. GLB) from signed URL to local filesystem."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        client = await self._get_client()

        headers = {"User-Agent": "TerraForge-Downloader/0.1.0"}
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

        logger.info(f"Successfully downloaded model ({path.stat().st_size} bytes) to {path}")
        return path

    async def download_thumbnail(
        self,
        url: str,
        output_path: Union[str, Path],
    ) -> Path:
        """Download preview thumbnail image."""
        return await self.download_model(url, output_path)
