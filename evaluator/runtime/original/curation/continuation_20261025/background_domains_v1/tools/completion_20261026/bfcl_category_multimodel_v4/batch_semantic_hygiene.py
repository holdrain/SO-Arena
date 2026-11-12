"""Batch-local cleanup for public candidate provenance and numeric menus."""
from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import json
import math
import re
from typing import Any


_NON_TASK_KEYS = {
    "provenance",
    "candidateid",
    "eventindex",
    "turnindex",
    "sequenceindex",
    "requestid",
    "traceid",
    "toolcallid",
    "sourcepath",
    "calculationid",
}


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _is_non_task_key(key: Any) -> bool:
    return _normalized_key(key) in _NON_TASK_KEYS


def _walk_public_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        out = [value]
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return out
        if isinstance(parsed, (list, tuple, dict)):
            out.extend(_walk_public_strings(parsed))
        return out
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            if not _is_non_task_key(key)
            for item in _walk_public_strings(child)
        ]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _walk_public_strings(child)]
    return []


def _walk_public_scalars(value: Any, source_path: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        out = [{"value": value, "source_path": source_path, "type": type(value).__name__}]
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return out
            if isinstance(parsed, (list, tuple, dict)):
                out.extend(_walk_public_scalars(parsed, f"{source_path}.<literal_repr>"))
        return out
    if isinstance(value, dict):
        out: list[dict[str, Any]] = []
        for key, child in value.items():
            if _is_non_task_key(key):
                continue
            path = f"{source_path}.{key}" if source_path else str(key)
            out.extend(_walk_public_scalars(child, path))
        return out
    if isinstance(value, (list, tuple)):
        return [
            item
            for index, child in enumerate(value)
            for item in _walk_public_scalars(child, f"{source_path}[{index}]")
        ]
    return []


def _numeric_identity(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return None


def _candidate_priority(item: dict[str, Any]) -> tuple[int, int]:
    source = str(item.get("source", "")).casefold()
    if source.startswith("documented native range endpoint"):
        rank = 0
    elif "native schema enum" in source or "closed choices documented" in source:
        rank = 1
    elif "public_calculator.calculate result" in source:
        rank = 2
    elif source.startswith("public result"):
        rank = 3
    elif source.startswith("visible user") or source.startswith("user goal"):
        rank = 4
    else:
        rank = 5
    return rank, len(source)


def install(runner: Any) -> dict[str, Any]:
    """Drop helper bookkeeping and collapse only number/float value duplicates."""
    runner._walk_public_strings = _walk_public_strings
    runner._walk_public_scalars = _walk_public_scalars
    original = runner.argument_value_candidates

    def candidates(
        parameter_name: str,
        parameter_schema: dict[str, Any],
        goal: str,
        history: list[dict[str, Any]],
        *,
        required: bool,
        path_goal: str | None = None,
    ) -> list[dict[str, Any]]:
        items = original(
            parameter_name,
            parameter_schema,
            goal,
            history,
            required=required,
            path_goal=path_goal,
        )
        declared = parameter_schema.get("type", "string")
        declared_types = set(declared if isinstance(declared, list) else [declared])
        if not declared_types.intersection({"number", "float"}):
            return items

        groups: dict[Decimal, list[tuple[int, dict[str, Any]]]] = {}
        passthrough: list[tuple[int, dict[str, Any]]] = []
        for index, item in enumerate(items):
            identity = _numeric_identity(item.get("value"))
            if identity is None:
                passthrough.append((index, item))
            else:
                groups.setdefault(identity, []).append((index, item))

        selected: list[tuple[int, dict[str, Any]]] = list(passthrough)
        for duplicates in groups.values():
            selected.append(min(
                duplicates,
                key=lambda entry: (_candidate_priority(entry[1]), entry[0]),
            ))
        selected.sort(key=lambda entry: entry[0])
        return [item for _index, item in selected]

    runner.argument_value_candidates = candidates
    return {
        "metadata_keys_excluded_from_public_scalar_and_string_walks": sorted(_NON_TASK_KEYS),
        "numeric_deduplication": "schema number/float values grouped by numeric value; Booleans excluded; documented range endpoints preferred; integer fields retain exact native types",
        "full_valid_range_preserved": True,
    }
