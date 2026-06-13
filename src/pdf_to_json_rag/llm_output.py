"""Strict output parsing helpers for optional LLM runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+-]+)?\s*\n(?P<body>[\s\S]*?)```",
    re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedJsonOutput:
    ok: bool
    status: str
    value: Any = None
    output_format: str | None = None
    error_preview: str = ""
    raw_char_count: int = 0


def _preview(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _without_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_strict_json_output(text: str, *, require_object: bool = True) -> ParsedJsonOutput:
    """Parse raw or fenced JSON output without accepting surrounding prose."""
    raw = text or ""
    cleaned = _without_think_blocks(raw)
    if not cleaned:
        return ParsedJsonOutput(
            ok=False,
            status="empty_output",
            raw_char_count=len(raw),
        )

    fences = list(_FENCE_RE.finditer(cleaned))
    output_format = "raw_json"
    payload = cleaned
    if fences:
        if len(fences) != 1:
            return ParsedJsonOutput(
                ok=False,
                status="multiple_fenced_blocks",
                error_preview=_preview(cleaned),
                raw_char_count=len(raw),
            )
        fence = fences[0]
        before = cleaned[: fence.start()].strip()
        after = cleaned[fence.end() :].strip()
        if before or after:
            return ParsedJsonOutput(
                ok=False,
                status="text_outside_fence",
                error_preview=_preview(cleaned),
                raw_char_count=len(raw),
            )
        lang = (fence.group("lang") or "").lower()
        if lang and lang not in {"json", "jsonc"}:
            return ParsedJsonOutput(
                ok=False,
                status="non_json_fence",
                error_preview=lang,
                raw_char_count=len(raw),
            )
        payload = fence.group("body").strip()
        output_format = "fenced_json"

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return ParsedJsonOutput(
            ok=False,
            status="invalid_json",
            error_preview=_preview(str(exc)),
            output_format=output_format,
            raw_char_count=len(raw),
        )

    if require_object and not isinstance(parsed, dict):
        return ParsedJsonOutput(
            ok=False,
            status="json_not_object",
            output_format=output_format,
            raw_char_count=len(raw),
        )

    return ParsedJsonOutput(
        ok=True,
        status="ok",
        value=parsed,
        output_format=output_format,
        raw_char_count=len(raw),
    )


def parsed_json_payload(parsed: ParsedJsonOutput) -> dict[str, object]:
    return {
        "ok": parsed.ok,
        "status": parsed.status,
        "output_format": parsed.output_format,
        "raw_char_count": parsed.raw_char_count,
        "error_preview": parsed.error_preview,
    }
