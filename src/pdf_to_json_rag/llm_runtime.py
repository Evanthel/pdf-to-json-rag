"""Optional local command runtime for prompt-based LLM hooks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import subprocess
from typing import Protocol


DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
LLM_TIMEOUT_ENV = "PDF_TO_JSON_RAG_LLM_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class PromptCommandResult:
    configured: bool
    invoked: bool
    status: str
    stdout: str = ""
    stderr_preview: str = ""
    returncode: int | None = None
    command_preview: str | None = None
    timeout_seconds: float | None = None
    provider_id: str | None = None
    provider_kind: str | None = None


class PromptRuntimeProvider(Protocol):
    provider_id: str
    provider_kind: str

    def run(self, prompt: str) -> PromptCommandResult:
        """Run one prompt and return public-safe runtime details."""


@dataclass(frozen=True)
class LocalCommandPromptProvider:
    env_var: str
    provider_id: str = "local_command"
    provider_kind: str = "subprocess"

    def run(self, prompt: str) -> PromptCommandResult:
        return _run_local_command_prompt(
            prompt=prompt,
            command_env_var=self.env_var,
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
        )


def _preview_text(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _runtime_timeout_seconds() -> float:
    configured = os.environ.get(LLM_TIMEOUT_ENV, "").strip()
    if not configured:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        value = float(configured)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return max(1.0, value)


def _run_local_command_prompt(
    *,
    prompt: str,
    command_env_var: str,
    provider_id: str,
    provider_kind: str,
) -> PromptCommandResult:
    """Send a prompt to an opt-in local command over stdin.

    The command is intentionally configured by environment variable so the package
    stays provider-agnostic and offline by default. It is split with shlex and run
    without a shell; callers should not put secrets in argv because only argv[0] is
    surfaced in public metadata.
    """
    command = os.environ.get(command_env_var, "").strip()
    if not command:
        return PromptCommandResult(
            configured=False,
            invoked=False,
            status="not_configured",
            timeout_seconds=_runtime_timeout_seconds(),
            provider_id=provider_id,
            provider_kind=provider_kind,
        )

    try:
        args = shlex.split(command)
    except ValueError as exc:
        return PromptCommandResult(
            configured=True,
            invoked=False,
            status=f"invalid_command: {exc}",
            timeout_seconds=_runtime_timeout_seconds(),
            provider_id=provider_id,
            provider_kind=provider_kind,
        )
    if not args:
        return PromptCommandResult(
            configured=True,
            invoked=False,
            status="empty_command",
            timeout_seconds=_runtime_timeout_seconds(),
            provider_id=provider_id,
            provider_kind=provider_kind,
        )

    timeout_seconds = _runtime_timeout_seconds()
    try:
        completed = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return PromptCommandResult(
            configured=True,
            invoked=True,
            status="command_not_found",
            command_preview=args[0],
            timeout_seconds=timeout_seconds,
            provider_id=provider_id,
            provider_kind=provider_kind,
        )
    except subprocess.TimeoutExpired as exc:
        return PromptCommandResult(
            configured=True,
            invoked=True,
            status="timeout",
            stdout=exc.stdout or "",
            stderr_preview=_preview_text(exc.stderr or ""),
            command_preview=args[0],
            timeout_seconds=timeout_seconds,
            provider_id=provider_id,
            provider_kind=provider_kind,
        )

    return PromptCommandResult(
        configured=True,
        invoked=True,
        status="ok" if completed.returncode == 0 else "nonzero_exit",
        stdout=completed.stdout.strip(),
        stderr_preview=_preview_text(completed.stderr),
        returncode=completed.returncode,
        command_preview=args[0],
        timeout_seconds=timeout_seconds,
        provider_id=provider_id,
        provider_kind=provider_kind,
    )


def provider_for_env_command(command_env_var: str) -> PromptRuntimeProvider:
    return LocalCommandPromptProvider(env_var=command_env_var)


def run_prompt_command(prompt: str, command_env_var: str) -> PromptCommandResult:
    """Compatibility helper for the default local-command provider."""
    return provider_for_env_command(command_env_var).run(prompt)


def prompt_command_payload(result: PromptCommandResult) -> dict[str, object]:
    """Return public-safe runtime metadata."""
    return {
        "configured": result.configured,
        "invoked": result.invoked,
        "status": result.status,
        "returncode": result.returncode,
        "command_preview": result.command_preview,
        "timeout_seconds": result.timeout_seconds,
        "provider_id": result.provider_id,
        "provider_kind": result.provider_kind,
        "stdout_char_count": len(result.stdout),
        "stderr_preview": result.stderr_preview,
    }
