"""Deterministic repository context profiles for Pure RLM runs.

The profiles in this module deliberately keep repository content outside the
root model's prompt.  They only decide what is placed in the REPL ``context``
variable; the Pure RLM environment controls what observations flow back to the
root model.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class RepositoryContextProfile(str, Enum):
    """Supported repository-to-REPL context selection policies."""

    AUTO = "auto"
    MINI = "mini"
    EVIDENCE = "evidence"
    FULL = "full"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class RepositoryContextResult:
    """A selected context plus enough metadata to reproduce the selection."""

    context: dict[str, str]
    profile: str
    files: tuple[str, ...]
    total_chars: int
    search_terms: tuple[str, ...] = ()
    source: str = "repository"


class RepositoryContextBuilder:
    """Build bounded, deterministic repository contexts for RLM tasks."""

    DEFAULT_INCLUDE = (
        "*.py",
        "*.pyi",
        "*.md",
        "*.rst",
        "*.toml",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.js",
        "*.jsx",
        "*.ts",
        "*.tsx",
        "*.java",
        "*.go",
        "*.rs",
        "*.sh",
    )
    DEFAULT_EXCLUDED_PARTS = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".rlm_code",
            ".dspy_code",
            "dist",
            "build",
        }
    )
    _STOP_WORDS = frozenset(
        {
            "about",
            "after",
            "again",
            "against",
            "also",
            "and",
            "any",
            "are",
            "base",
            "code",
            "could",
            "current",
            "does",
            "file",
            "files",
            "find",
            "from",
            "have",
            "into",
            "make",
            "need",
            "please",
            "repo",
            "repository",
            "should",
            "task",
            "that",
            "the",
            "their",
            "then",
            "this",
            "using",
            "want",
            "what",
            "when",
            "where",
            "which",
            "with",
            "would",
        }
    )

    def __init__(self, workdir: Path | None = None):
        self.workdir = (workdir or Path.cwd()).resolve()

    def build(
        self,
        task: str,
        *,
        profile: str | RepositoryContextProfile = RepositoryContextProfile.AUTO,
        paths: Iterable[str] | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        max_files: int | None = None,
        max_total_chars: int | None = None,
    ) -> RepositoryContextResult:
        """Select repository material and return it as a mapping of file to text."""
        requested = self.normalize_profile(profile)
        explicit_paths = self._normalize_paths(paths or ())
        effective = requested
        if requested == RepositoryContextProfile.AUTO:
            effective = (
                RepositoryContextProfile.FULL
                if explicit_paths
                else RepositoryContextProfile.EVIDENCE
            )
        if requested == RepositoryContextProfile.EXPLICIT and not explicit_paths:
            raise ValueError("The explicit context profile requires at least one path.")

        limits = {
            RepositoryContextProfile.MINI: (8, 12_000, 1_800, 2),
            RepositoryContextProfile.EVIDENCE: (20, 48_000, 4_000, 4),
            RepositoryContextProfile.FULL: (24, 160_000, 16_000, 0),
            RepositoryContextProfile.EXPLICIT: (24, 160_000, 24_000, 0),
        }
        file_limit, total_limit, per_file_limit, radius = limits[effective]
        file_limit = max(1, int(max_files or file_limit))
        total_limit = max(1, int(max_total_chars or total_limit))

        candidates = (
            self._resolve_explicit(explicit_paths)
            if explicit_paths
            else self._discover(include=include, exclude=exclude)
        )
        terms = self.extract_search_terms(task)
        text_cache: dict[Path, str] = {}
        if effective in {
            RepositoryContextProfile.MINI,
            RepositoryContextProfile.EVIDENCE,
        }:
            candidates, text_cache = self._rank_evidence_candidates(candidates, terms)
        candidates = candidates[:file_limit]

        context: dict[str, str] = {}
        remaining = total_limit
        for path in candidates:
            if remaining <= 0:
                break
            text = text_cache.get(path) or self._read_text(path)
            if not text:
                continue
            if effective in {
                RepositoryContextProfile.MINI,
                RepositoryContextProfile.EVIDENCE,
            }:
                snippet = self._evidence_snippet(
                    text,
                    terms,
                    radius=radius,
                    max_chars=min(per_file_limit, remaining),
                )
            else:
                snippet = text[: min(per_file_limit, remaining)]
            if not snippet:
                continue
            relative = path.relative_to(self.workdir).as_posix()
            context[relative] = snippet
            remaining -= len(snippet)

        if not context:
            available = [p.relative_to(self.workdir).as_posix() for p in candidates[:80]]
            fallback = (
                f"Workspace: {self.workdir}\n"
                "No readable snippets matched. Candidate files:\n" + "\n".join(available)
            ).strip()
            context["_workspace"] = fallback[:total_limit]

        source = "explicit_paths" if explicit_paths else "repository"
        return RepositoryContextResult(
            context=context,
            profile=effective.value,
            files=tuple(context),
            total_chars=sum(len(value) for value in context.values()),
            search_terms=terms,
            source=source,
        )

    @classmethod
    def normalize_profile(cls, profile: str | RepositoryContextProfile) -> RepositoryContextProfile:
        if isinstance(profile, RepositoryContextProfile):
            return profile
        normalized = str(profile or "auto").strip().lower().replace("-", "_")
        aliases = {"repo_evidence": "evidence", "small": "mini", "all": "full"}
        normalized = aliases.get(normalized, normalized)
        try:
            return RepositoryContextProfile(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in RepositoryContextProfile)
            raise ValueError(
                f"Unknown repository context profile '{profile}'. Supported: {supported}"
            ) from exc

    @classmethod
    def extract_search_terms(cls, task: str, *, limit: int = 16) -> tuple[str, ...]:
        """Extract stable identifiers and task terms used by evidence profiles."""
        text = str(task or "")
        raw: list[str] = []
        raw.extend(re.findall(r"`([^`]{2,80})`", text))
        raw.extend(
            re.findall(
                r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*\b|"
                r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b",
                text,
            )
        )
        terms: list[str] = []
        seen: set[str] = set()
        for value in raw:
            candidate = value.strip().strip(".,:;()[]{}'\"")
            lowered = candidate.lower()
            if (
                len(candidate) < 3
                or lowered in cls._STOP_WORDS
                or lowered in seen
                or candidate.endswith((".py", ".md", ".toml", ".yaml", ".yml", ".json"))
            ):
                continue
            seen.add(lowered)
            terms.append(candidate)
            if len(terms) >= max(1, int(limit)):
                break
        return tuple(terms)

    def _discover(
        self,
        *,
        include: Iterable[str] | None,
        exclude: Iterable[str] | None,
    ) -> list[Path]:
        include_globs = tuple(include or self.DEFAULT_INCLUDE)
        exclude_globs = tuple(exclude or ())
        found: list[Path] = []
        for path in self.workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.workdir)
            rel_text = relative.as_posix()
            if any(part in self.DEFAULT_EXCLUDED_PARTS for part in relative.parts):
                continue
            if exclude_globs and any(fnmatch.fnmatch(rel_text, item) for item in exclude_globs):
                continue
            if not any(
                fnmatch.fnmatch(path.name, item) or fnmatch.fnmatch(rel_text, item)
                for item in include_globs
            ):
                continue
            found.append(path)
        return sorted(found, key=lambda item: item.relative_to(self.workdir).as_posix())

    def _normalize_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in paths:
            item = str(raw or "").strip().strip("`'\"")
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return tuple(normalized)

    def _resolve_explicit(self, paths: Iterable[str]) -> list[Path]:
        resolved: list[Path] = []
        for raw in paths:
            try:
                candidate = (self.workdir / raw).resolve()
            except Exception:
                continue
            if self.workdir not in candidate.parents or not candidate.is_file():
                continue
            resolved.append(candidate)
        return resolved

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return ""

    def _rank_evidence_candidates(
        self,
        candidates: list[Path],
        terms: tuple[str, ...],
        *,
        scan_limit: int = 500,
        scan_chars_per_file: int = 100_000,
    ) -> tuple[list[Path], dict[Path, str]]:
        if not terms:
            return candidates, {}
        lowered_terms = tuple(term.lower() for term in terms)
        cache: dict[Path, str] = {}
        scored: list[tuple[int, str, Path]] = []
        for index, path in enumerate(candidates):
            relative = path.relative_to(self.workdir).as_posix()
            path_text = relative.lower()
            path_score = sum(20 for term in lowered_terms if term in path_text)
            content_score = 0
            if index < max(1, int(scan_limit)):
                text = self._read_text(path)
                cache[path] = text
                sample = text[:scan_chars_per_file].lower()
                content_score = sum(min(10, sample.count(term)) for term in lowered_terms)
            scored.append((path_score + content_score, relative, path))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scored], cache

    @staticmethod
    def _evidence_snippet(
        text: str,
        terms: tuple[str, ...],
        *,
        radius: int,
        max_chars: int,
    ) -> str:
        lines = text.splitlines()
        lowered_terms = tuple(item.lower() for item in terms)
        occurrences = [
            [index for index, line in enumerate(lines) if term in line.lower()]
            for term in lowered_terms
        ]
        ordered_matches: list[int] = []
        seen_matches: set[int] = set()
        occurrence_index = 0
        while any(occurrence_index < len(items) for items in occurrences):
            for items in occurrences:
                if occurrence_index >= len(items):
                    continue
                match = items[occurrence_index]
                if match not in seen_matches:
                    seen_matches.add(match)
                    ordered_matches.append(match)
            occurrence_index += 1
        if not ordered_matches and lines:
            ordered_matches = [0]

        covered: set[int] = set()
        rendered: list[str] = []
        rendered_chars = 0
        for match in ordered_matches:
            window = [
                index
                for index in range(
                    max(0, match - radius),
                    min(len(lines), match + radius + 1),
                )
                if index not in covered
            ]
            if not window:
                continue
            if rendered:
                rendered.append("...")
                rendered_chars += 4
            for index in window:
                row = f"{index + 1:>5} | {lines[index]}"
                rendered.append(row)
                covered.add(index)
                rendered_chars += len(row) + 1
                if rendered_chars >= max_chars:
                    break
            if rendered_chars >= max_chars:
                break
        return "\n".join(rendered)[:max_chars]


def describe_explicit_context(context: Any) -> tuple[list[str], int]:
    """Return stable file/key labels and a character count for caller context."""
    if isinstance(context, dict):
        labels = [str(key) for key in context]
        return labels, sum(len(str(value)) for value in context.values())
    if isinstance(context, (list, tuple)):
        return [f"context_{index}" for index in range(len(context))], sum(
            len(str(value)) for value in context
        )
    return ["context"], len(str(context))
