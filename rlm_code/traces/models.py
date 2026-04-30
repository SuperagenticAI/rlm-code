"""Small, dependency-light models for one-span-per-line trace JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _status_code(raw: dict[str, Any]) -> str:
    status = _as_dict(raw.get("status"))
    code = status.get("code") or raw.get("status_code") or "STATUS_CODE_UNSET"
    return str(code)


@dataclass(slots=True)
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: str
    end_time: str
    status_code: str
    attributes: dict[str, Any] = field(default_factory=dict)
    resource_attributes: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: str | bytes) -> "SpanRecord":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("trace line must be a JSON object")
        context = _as_dict(raw.get("context"))
        resource = _as_dict(raw.get("resource"))
        return cls(
            trace_id=str(raw.get("trace_id") or context.get("trace_id") or ""),
            span_id=str(raw.get("span_id") or context.get("span_id") or ""),
            parent_span_id=(
                str(raw.get("parent_span_id"))
                if raw.get("parent_span_id") not in {None, ""}
                else None
            ),
            name=str(raw.get("name") or ""),
            kind=str(raw.get("kind") or raw.get("span_kind") or ""),
            start_time=str(raw.get("start_time") or raw.get("startTimeUnixNano") or ""),
            end_time=str(raw.get("end_time") or raw.get("endTimeUnixNano") or ""),
            status_code=_status_code(raw),
            attributes=dict(_as_dict(raw.get("attributes"))),
            resource_attributes=dict(_as_dict(resource.get("attributes"))),
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceIndexRow:
    trace_id: str
    byte_offsets: list[int] = field(default_factory=list)
    byte_lengths: list[int] = field(default_factory=list)
    span_count: int = 0
    start_time: str = ""
    end_time: str = ""
    has_errors: bool = False
    service_names: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    project_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TraceIndexRow":
        data = json.loads(payload)
        return cls(**data)


@dataclass(slots=True)
class TraceIndexMeta:
    schema_version: int
    trace_count: int
    source_size: int
    source_mtime_ns: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TraceIndexMeta":
        data = json.loads(payload)
        return cls(**data)
