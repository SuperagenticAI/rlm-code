"""Bounded trace query/view/search API used by the trace-analysis environment."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .index import TraceIndexBuilder
from .models import SpanRecord, TraceIndexRow

DISCOVERY_ATTR_CAP = 4096
SURGICAL_ATTR_CAP = 16384
VIEW_TRACE_CHAR_BUDGET = 150_000
OVERVIEW_SAMPLE_TRACE_IDS = 20
NOISY_FLAT_PROJECTION_RE = re.compile(r"^(?:llm\.(?:input|output)_messages|mcp\.tools)\.\d+\.")
EVIDENCE_ATTR_CAP = 2048
TASK_ID_ATTRS = (
    "inference.task_id",
    "task_id",
    "task.id",
    "benchmark.task_id",
    "appworld.task_id",
)
ISSUE_ATTRS = (
    "error.message",
    "exception.message",
    "exception.type",
    "tool.name",
    "input.value",
    "output.value",
)


def _truncate_value(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        if len(value) <= cap:
            return value
        return f"{value[:cap]}... [rlm-code trace truncated: original {len(value)} chars]"
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return value
    if len(serialized) <= cap:
        return value
    return (
        f"{serialized[:cap]}... "
        f"[rlm-code trace truncated: original {len(serialized)} chars; serialized]"
    )


def _render_span(span: SpanRecord, cap: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    dropped = 0
    for key, value in span.attributes.items():
        if NOISY_FLAT_PROJECTION_RE.match(key):
            dropped += 1
            continue
        attrs[key] = _truncate_value(value, cap)
    if dropped:
        attrs["__rlm_code_dropped_flat_projections"] = (
            f"{dropped} flat projection keys dropped; inspect JSON blob attributes instead."
        )
    payload = span.to_dict()
    payload["attributes"] = attrs
    payload.pop("raw", None)
    return payload


class TraceStore:
    """Read-only query API over a trace JSONL file and sidecar index."""

    def __init__(self, trace_path: Path, index_path: Path, rows: list[TraceIndexRow]) -> None:
        self.trace_path = trace_path
        self.index_path = index_path
        self.rows = rows
        self.rows_by_id = {row.trace_id: row for row in rows}

    @classmethod
    def load(cls, trace_path: str | Path, index_path: str | Path | None = None) -> "TraceStore":
        trace = Path(trace_path).resolve()
        index = TraceIndexBuilder.ensure_index_exists(trace, index_path)
        rows = [
            TraceIndexRow.from_json(line)
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(trace_path=trace, index_path=index, rows=rows)

    def get_overview(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        rows = self._filtered_rows(filters)
        services: set[str] = set()
        models: set[str] = set()
        agents: set[str] = set()
        for row in rows:
            services.update(row.service_names)
            models.update(row.model_names)
            agents.update(row.agent_names)
        return {
            "total_traces": len(rows),
            "total_spans": sum(row.span_count for row in rows),
            "earliest_start_time": min((row.start_time for row in rows), default=""),
            "latest_end_time": max((row.end_time for row in rows), default=""),
            "service_names": sorted(services),
            "model_names": sorted(models),
            "agent_names": sorted(agents),
            "error_trace_count": sum(1 for row in rows if row.has_errors),
            "total_input_tokens": sum(row.total_input_tokens for row in rows),
            "total_output_tokens": sum(row.total_output_tokens for row in rows),
            "sample_trace_ids": [row.trace_id for row in rows[:OVERVIEW_SAMPLE_TRACE_IDS]],
        }

    def query_traces(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        rows = self._filtered_rows(filters)
        return {
            "total": len(rows),
            "traces": [self._summary(row) for row in rows[max(0, offset) : max(0, offset) + limit]],
        }

    def count_traces(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"total": len(self._filtered_rows(filters))}

    def view_trace(self, trace_id: str) -> dict[str, Any]:
        spans = [_render_span(span, DISCOVERY_ATTR_CAP) for span in self._read_spans(trace_id)]
        sizes = [len(json.dumps(span, ensure_ascii=False)) for span in spans]
        total_chars = sum(sizes)
        if total_chars > VIEW_TRACE_CHAR_BUDGET:
            names = Counter(str(span.get("name", "")) for span in spans)
            return {
                "trace_id": trace_id,
                "spans": [],
                "oversized": {
                    "trace_id": trace_id,
                    "span_count": len(spans),
                    "total_serialized_chars": total_chars,
                    "char_budget": VIEW_TRACE_CHAR_BUDGET,
                    "span_size_min": min(sizes, default=0),
                    "span_size_median": sorted(sizes)[len(sizes) // 2] if sizes else 0,
                    "span_size_max": max(sizes, default=0),
                    "top_span_names": names.most_common(10),
                    "error_span_count": sum(
                        1 for span in spans if span.get("status_code") == "STATUS_CODE_ERROR"
                    ),
                    "recommendation": "Use search_trace with a pattern, then view_spans for specific span ids.",
                },
            }
        return {"trace_id": trace_id, "spans": spans, "oversized": None}

    def view_spans(self, trace_id: str, span_ids: list[str]) -> dict[str, Any]:
        wanted = {str(item) for item in span_ids}
        spans = [
            _render_span(span, SURGICAL_ATTR_CAP)
            for span in self._read_spans(trace_id)
            if span.span_id in wanted
        ]
        return {"trace_id": trace_id, "spans": spans, "oversized": None}

    def search_trace(self, trace_id: str, pattern: str, *, limit: int = 100) -> dict[str, Any]:
        if trace_id not in self.rows_by_id:
            raise KeyError(trace_id)
        row = self.rows_by_id[trace_id]
        matches: list[dict[str, Any]] = []
        with self.trace_path.open("rb") as handle:
            for offset, length in zip(row.byte_offsets, row.byte_lengths):
                handle.seek(offset)
                raw = handle.read(length).decode("utf-8", errors="replace")
                if pattern in raw:
                    matches.append(_render_span(SpanRecord.from_json(raw), DISCOVERY_ATTR_CAP))
                    if len(matches) >= limit:
                        break
        return {
            "trace_id": trace_id,
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches,
            "truncated": len(matches) >= limit,
        }

    def export_evidence_corpus(
        self,
        output_dir: str | Path,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        include_raw: bool = True,
    ) -> dict[str, Any]:
        """Export a layered evidence corpus for harness-optimization agents.

        The corpus mirrors the AHE progressive-disclosure pattern:
        a compact overview, one detail file per selected trace, an index, and
        optional lightly processed raw JSONL spans for drill-down.
        """
        out = Path(output_dir).resolve()
        detail_dir = out / "detail"
        raw_dir = out / "raw"
        detail_dir.mkdir(parents=True, exist_ok=True)
        if include_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        rows = self._filtered_rows(filters)[: max(0, limit)]
        overview = self.get_overview(filters)
        detail_entries: list[dict[str, Any]] = []
        detail_lines = self._render_overview_markdown(overview, rows, include_raw=include_raw)

        for row in rows:
            spans = self._read_spans(row.trace_id)
            safe_id = self._safe_filename(row.trace_id)
            detail_path = detail_dir / f"{safe_id}.md"
            raw_path = raw_dir / f"{safe_id}.jsonl" if include_raw else None
            detail_path.write_text(
                self._render_detail_markdown(row, spans, raw_path=raw_path),
                encoding="utf-8",
            )
            if raw_path is not None:
                self._write_raw_trace(raw_path, spans)
            detail_entries.append(
                {
                    "trace_id": row.trace_id,
                    "detail_path": str(detail_path),
                    "raw_path": str(raw_path) if raw_path is not None else None,
                    "has_errors": row.has_errors,
                    "span_count": row.span_count,
                    "task_ids": self._task_ids(spans),
                    "error_span_count": sum(1 for span in spans if span.status_code == "STATUS_CODE_ERROR"),
                }
            )
            detail_lines.append(
                f"- `{row.trace_id}`: {row.span_count} spans, "
                f"errors={'yes' if row.has_errors else 'no'}, detail=`detail/{safe_id}.md`"
            )

        overview_path = out / "overview.md"
        index_path = out / "index.json"
        overview_path.write_text("\n".join(detail_lines) + "\n", encoding="utf-8")
        index_payload = {
            "schema_version": "rlm-code.trace_evidence_corpus.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "source_trace_path": str(self.trace_path),
            "source_index_path": str(self.index_path),
            "filters": filters or {},
            "limit": limit,
            "include_raw": include_raw,
            "overview_path": str(overview_path),
            "detail_dir": str(detail_dir),
            "raw_dir": str(raw_dir) if include_raw else None,
            "overview": overview,
            "traces": detail_entries,
        }
        index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "output_dir": str(out),
            "overview_path": str(overview_path),
            "index_path": str(index_path),
            "detail_dir": str(detail_dir),
            "raw_dir": str(raw_dir) if include_raw else None,
            "trace_count": len(detail_entries),
            "detail_paths": [entry["detail_path"] for entry in detail_entries],
        }

    def _read_spans(self, trace_id: str) -> list[SpanRecord]:
        if trace_id not in self.rows_by_id:
            raise KeyError(trace_id)
        row = self.rows_by_id[trace_id]
        spans: list[SpanRecord] = []
        with self.trace_path.open("rb") as handle:
            for offset, length in zip(row.byte_offsets, row.byte_lengths):
                handle.seek(offset)
                spans.append(SpanRecord.from_json(handle.read(length)))
        return spans

    def _filtered_rows(self, filters: dict[str, Any] | None) -> list[TraceIndexRow]:
        filters = filters or {}
        rows = list(self.rows)
        if "has_errors" in filters and filters["has_errors"] is not None:
            expected = bool(filters["has_errors"])
            rows = [row for row in rows if row.has_errors == expected]
        rows = self._filter_any(rows, filters.get("model_names"), "model_names")
        rows = self._filter_any(rows, filters.get("service_names"), "service_names")
        rows = self._filter_any(rows, filters.get("agent_names"), "agent_names")
        project_id = filters.get("project_id")
        if project_id:
            rows = [row for row in rows if row.project_id == project_id]
        return rows

    @staticmethod
    def _filter_any(
        rows: list[TraceIndexRow],
        values: Any,
        field_name: str,
    ) -> list[TraceIndexRow]:
        if not values:
            return rows
        wanted = {str(value) for value in (values if isinstance(values, list) else [values])}
        return [row for row in rows if wanted.intersection(getattr(row, field_name))]

    @staticmethod
    def _summary(row: TraceIndexRow) -> dict[str, Any]:
        return {
            "trace_id": row.trace_id,
            "span_count": row.span_count,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "has_errors": row.has_errors,
            "service_names": row.service_names,
            "model_names": row.model_names,
            "agent_names": row.agent_names,
            "total_input_tokens": row.total_input_tokens,
            "total_output_tokens": row.total_output_tokens,
            "project_id": row.project_id,
        }

    @staticmethod
    def _render_overview_markdown(
        overview: dict[str, Any],
        rows: list[TraceIndexRow],
        *,
        include_raw: bool,
    ) -> list[str]:
        lines = [
            "# Trace Evidence Overview",
            "",
            "Generated by `rlm-code` trace analysis.",
            "",
            "## Dataset",
            "",
            f"- Traces selected: {len(rows)}",
            f"- Total matching traces: {overview['total_traces']}",
            f"- Total matching spans: {overview['total_spans']}",
            f"- Error traces: {overview['error_trace_count']}",
            f"- Services: {', '.join(overview['service_names']) or '-'}",
            f"- Models: {', '.join(overview['model_names']) or '-'}",
            f"- Agents: {', '.join(overview['agent_names']) or '-'}",
            f"- Input tokens: {overview['total_input_tokens']}",
            f"- Output tokens: {overview['total_output_tokens']}",
            f"- Raw span files included: {'yes' if include_raw else 'no'}",
            "",
            "## Trace Details",
            "",
        ]
        return lines

    def _render_detail_markdown(
        self,
        row: TraceIndexRow,
        spans: list[SpanRecord],
        *,
        raw_path: Path | None,
    ) -> str:
        task_ids = self._task_ids(spans)
        error_spans = [span for span in spans if span.status_code == "STATUS_CODE_ERROR"]
        tool_spans = [span for span in spans if self._looks_like_tool_span(span)]
        top_names = Counter(span.name for span in spans).most_common(10)
        lines = [
            f"# Trace Detail: {row.trace_id}",
            "",
            "## Summary",
            "",
            f"- Trace id: `{row.trace_id}`",
            f"- Spans: {row.span_count}",
            f"- Has errors: {'yes' if row.has_errors else 'no'}",
            f"- Error spans: {len(error_spans)}",
            f"- Task ids: {', '.join(task_ids) or '-'}",
            f"- Services: {', '.join(row.service_names) or '-'}",
            f"- Models: {', '.join(row.model_names) or '-'}",
            f"- Agents: {', '.join(row.agent_names) or '-'}",
            f"- Start: {row.start_time or '-'}",
            f"- End: {row.end_time or '-'}",
        ]
        if raw_path is not None:
            lines.append(f"- Raw spans: `{raw_path.name}`")
        lines.extend(["", "## Span Name Counts", ""])
        lines.extend(f"- `{name}`: {count}" for name, count in top_names)
        lines.extend(["", "## Error Spans", ""])
        if error_spans:
            for span in error_spans:
                lines.extend(self._render_span_evidence(span))
        else:
            lines.append("- None")
        lines.extend(["", "## Tool-Like Spans", ""])
        if tool_spans:
            for span in tool_spans[:20]:
                lines.extend(self._render_span_evidence(span))
        else:
            lines.append("- None")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_span_evidence(span: SpanRecord) -> list[str]:
        lines = [
            f"### `{span.name or span.span_id}`",
            "",
            f"- Span id: `{span.span_id}`",
            f"- Parent span id: `{span.parent_span_id or '-'}`",
            f"- Status: {span.status_code}",
        ]
        attrs = {
            key: _truncate_value(span.attributes[key], EVIDENCE_ATTR_CAP)
            for key in ISSUE_ATTRS
            if key in span.attributes
        }
        if attrs:
            lines.append("- Evidence attributes:")
            for key, value in attrs.items():
                lines.append(f"  - `{key}`: `{value}`")
        return lines + [""]

    @staticmethod
    def _write_raw_trace(path: Path, spans: list[SpanRecord]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for span in spans:
                handle.write(json.dumps(_render_span(span, SURGICAL_ATTR_CAP), sort_keys=True))
                handle.write("\n")

    @staticmethod
    def _task_ids(spans: list[SpanRecord]) -> list[str]:
        task_ids: set[str] = set()
        for span in spans:
            for key in TASK_ID_ATTRS:
                value = span.attributes.get(key)
                if isinstance(value, str) and value.strip():
                    task_ids.add(value.strip())
        return sorted(task_ids)

    @staticmethod
    def _looks_like_tool_span(span: SpanRecord) -> bool:
        name = span.name.lower()
        return (
            "tool" in name
            or "function" in name
            or "tool.name" in span.attributes
            or "input.value" in span.attributes
            or "output.value" in span.attributes
        )

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return safe or "trace"
