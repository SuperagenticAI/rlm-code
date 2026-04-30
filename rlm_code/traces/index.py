"""Sidecar trace index builder for HALO-style JSONL traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import SpanRecord, TraceIndexMeta, TraceIndexRow


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_int(*values: object) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return 0


@dataclass
class _Accumulator:
    trace_id: str
    byte_offsets: list[int] = field(default_factory=list)
    byte_lengths: list[int] = field(default_factory=list)
    span_count: int = 0
    start_time: str = ""
    end_time: str = ""
    has_errors: bool = False
    service_names: set[str] = field(default_factory=set)
    model_names: set[str] = field(default_factory=set)
    agent_names: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    project_id: str | None = None

    def absorb(self, span: SpanRecord, *, offset: int, length: int) -> None:
        self.byte_offsets.append(offset)
        self.byte_lengths.append(length)
        self.span_count += 1
        if not self.start_time or (span.start_time and span.start_time < self.start_time):
            self.start_time = span.start_time
        if not self.end_time or span.end_time > self.end_time:
            self.end_time = span.end_time
        if span.status_code == "STATUS_CODE_ERROR":
            self.has_errors = True

        attrs = span.attributes
        resource_attrs = span.resource_attributes
        service = _first_str(resource_attrs.get("service.name"), attrs.get("service.name"))
        model = _first_str(attrs.get("inference.llm.model_name"), attrs.get("llm.model_name"))
        agent = _first_str(attrs.get("inference.agent_name"), attrs.get("agent.name"))
        project = _first_str(attrs.get("inference.project_id"), attrs.get("project_id"))
        if service:
            self.service_names.add(service)
        if model:
            self.model_names.add(model)
        if agent:
            self.agent_names.add(agent)
        if project and self.project_id is None:
            self.project_id = project
        self.total_input_tokens += _first_int(
            attrs.get("inference.llm.input_tokens"),
            attrs.get("llm.token_count.prompt"),
        )
        self.total_output_tokens += _first_int(
            attrs.get("inference.llm.output_tokens"),
            attrs.get("llm.token_count.completion"),
        )

    def finalize(self) -> TraceIndexRow:
        return TraceIndexRow(
            trace_id=self.trace_id,
            byte_offsets=self.byte_offsets,
            byte_lengths=self.byte_lengths,
            span_count=self.span_count,
            start_time=self.start_time,
            end_time=self.end_time,
            has_errors=self.has_errors,
            service_names=sorted(self.service_names),
            model_names=sorted(self.model_names),
            agent_names=sorted(self.agent_names),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            project_id=self.project_id,
        )


class TraceIndexBuilder:
    """Build and reuse a sidecar index for one-span-per-line JSONL traces."""

    SCHEMA_VERSION = 1

    @classmethod
    def ensure_index_exists(cls, trace_path: str | Path, index_path: str | Path | None = None) -> Path:
        trace = Path(trace_path).resolve()
        index = Path(index_path).resolve() if index_path is not None else cls.default_index_path(trace)
        meta = cls.meta_path_for(index)
        size, mtime_ns = cls._fingerprint(trace)
        if index.exists() and meta.exists():
            existing = TraceIndexMeta.from_json(meta.read_text(encoding="utf-8"))
            if (
                existing.schema_version == cls.SCHEMA_VERSION
                and existing.source_size == size
                and existing.source_mtime_ns == mtime_ns
            ):
                return index
        cls.build_index(trace, index)
        return index

    @classmethod
    def build_index(cls, trace_path: str | Path, index_path: str | Path | None = None) -> Path:
        trace = Path(trace_path).resolve()
        index = Path(index_path).resolve() if index_path is not None else cls.default_index_path(trace)
        accumulators: dict[str, _Accumulator] = {}

        with trace.open("rb") as handle:
            offset = 0
            for raw_line in handle:
                line_len = len(raw_line)
                stripped = raw_line.rstrip(b"\n")
                if stripped:
                    span = SpanRecord.from_json(stripped)
                    if span.trace_id:
                        acc = accumulators.setdefault(span.trace_id, _Accumulator(span.trace_id))
                        acc.absorb(span, offset=offset, length=len(stripped))
                offset += line_len

        rows = [acc.finalize() for acc in accumulators.values()]
        index.parent.mkdir(parents=True, exist_ok=True)
        tmp = index.with_suffix(index.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.to_json())
                handle.write("\n")
        tmp.replace(index)

        size, mtime_ns = cls._fingerprint(trace)
        meta = TraceIndexMeta(
            schema_version=cls.SCHEMA_VERSION,
            trace_count=len(rows),
            source_size=size,
            source_mtime_ns=mtime_ns,
        )
        cls.meta_path_for(index).write_text(meta.to_json(), encoding="utf-8")
        return index

    @staticmethod
    def default_index_path(trace_path: Path) -> Path:
        return Path(str(trace_path) + ".rlm-trace-index.jsonl")

    @staticmethod
    def meta_path_for(index_path: Path) -> Path:
        return Path(str(index_path) + ".meta.json")

    @staticmethod
    def _fingerprint(trace_path: Path) -> tuple[int, int]:
        stat = trace_path.stat()
        return stat.st_size, stat.st_mtime_ns
