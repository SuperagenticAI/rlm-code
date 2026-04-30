"""Bounded trace query/view/search API used by the trace-analysis environment."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .index import TraceIndexBuilder
from .models import SpanRecord, TraceIndexRow

DISCOVERY_ATTR_CAP = 4096
SURGICAL_ATTR_CAP = 16384
VIEW_TRACE_CHAR_BUDGET = 150_000
OVERVIEW_SAMPLE_TRACE_IDS = 20
NOISY_FLAT_PROJECTION_RE = re.compile(r"^(?:llm\.(?:input|output)_messages|mcp\.tools)\.\d+\.")


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
