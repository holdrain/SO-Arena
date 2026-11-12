"""Collect type-valid argument values from visible request, results, and schema."""
from __future__ import annotations

import datetime
import json
import math
import re
from typing import Any, Iterable


_MAX_STRING = 8192
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")
_MONTH = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_NUMBER_WORDS = {
    "age", "amount", "count", "cost", "distance", "duration", "height", "hour",
    "hours", "index", "limit", "minute", "minutes", "number", "percent",
    "percentage", "price", "quantity", "rate", "rating", "score", "size", "total",
    "weight", "year", "years",
}
_STOP = {
    "a", "an", "and", "as", "at", "before", "between", "by", "call", "can",
    "contact", "could", "current", "directory", "do", "during", "email", "find",
    "folder", "from", "get", "gather", "help", "here", "i", "in", "into", "it",
    "me", "message", "my", "of", "on", "or", "over", "please", "reach", "send",
    "set", "tell", "the", "this", "through", "to", "under", "use", "we", "with",
}
_QUOTED = re.compile(r'''"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|`([^`\n]+)`|(?<!\w)'([^'\n]+)'(?!\w)''')
_LABELED_VALUE = r'''(?P<value>"[^"\n]{1,8192}"|'[^'\n]{1,8192}'|“[^”\n]{1,8192}”|‘[^’\n]{1,8192}’|[$€£¥]?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[A-Za-z0-9][A-Za-z0-9_@./+-]*)'''


def _words(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.casefold())


def _aliases(name: str) -> list[list[str]]:
    words = _words(name)
    aliases = [words] if words else []
    if words and words[-1] in {"id", "identifier", "name", "date", "time"}:
        aliases.append([words[-1]])
    return aliases


def _label(words: list[str]) -> str:
    return r"(?<!\w)" + r"[\s_-]*".join(map(re.escape, words)) + r"(?!\w)"


def _is_id(name: str) -> bool:
    return bool(_words(name) and _words(name)[-1] in {"id", "identifier"})


def _type_ok(value: Any, schema: dict[str, Any]) -> bool:
    kinds = schema.get("type")
    if isinstance(kinds, list):
        if not any(_type_ok(value, {**schema, "type": kind}) for kind in kinds):
            return False
        kind = next((part for part in kinds if _type_ok(value, {"type": part})), None)
    else:
        kind = kinds
    checks = {
        "null": lambda: value is None,
        "string": lambda: type(value) is str,
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float) and (type(value) is not float or math.isfinite(value)),
        "float": lambda: type(value) is float and math.isfinite(value),
        "boolean": lambda: type(value) is bool,
        "object": lambda: type(value) is dict,
        "dict": lambda: type(value) is dict,
        "array": lambda: type(value) is list,
    }
    if kind in checks and not checks[kind]():
        return False
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        return False
    if "const" in schema and not (type(value) is type(schema["const"]) and value == schema["const"]):
        return False
    if kind in {"object", "dict"} and type(value) is dict:
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        for key, child in value.items():
            if key in properties and not _type_ok(child, properties[key]):
                return False
            if key not in properties and schema.get("additionalProperties") is False:
                return False
    elif kind == "array" and type(value) is list:
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return False
        item_schema = schema.get("items", {})
        if isinstance(item_schema, dict) and any(not _type_ok(item, item_schema) for item in value):
            return False
    elif kind == "string" and type(value) is str:
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    return False
            except re.error:
                pass
        if schema.get("format") == "date":
            try:
                datetime.date.fromisoformat(value)
            except ValueError:
                return False
    elif kind in {"integer", "number", "float"} and type(value) in (int, float):
        for key, ok in (("minimum", lambda x, y: x >= y), ("maximum", lambda x, y: x <= y),
                        ("exclusiveMinimum", lambda x, y: x > y), ("exclusiveMaximum", lambda x, y: x < y)):
            if key in schema and not ok(value, schema[key]):
                return False
    return True


def _walk(value: Any, path: str) -> Iterable[tuple[Any, str]]:
    if type(value) is dict:
        yield value, path
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif type(value) is list:
        yield value, path
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
    elif value is None or type(value) in (str, int, float, bool):
        yield value, path


def _request_structures(text: str) -> Iterable[tuple[Any, str]]:
    """Extract JSON structures from visible text without evaluating code."""
    for match in re.finditer(r"\{[^{}\n]{1,8192}\}|\[[^\[\]\n]{1,8192}\]", text):
        try:
            value = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if type(value) in (dict, list):
            yield value, f"JSON value in request at character {match.start()}"


def _number(raw: str, schema: dict[str, Any], name: str) -> Any | None:
    raw = re.sub(r"^\s*[$€£¥]\s*", "", raw.strip()).replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
        return None
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item in {"integer", "number", "float"}), None)
    if kind == "integer":
        return int(raw) if "." not in raw else None
    if kind == "float" or kind == "number" and "." in raw:
        if _is_id(name):
            return None
        result = float(raw)
        return result if math.isfinite(result) else None
    if kind == "number":
        return int(raw)
    return None


def _date_values(text: str, name: str, schema: dict[str, Any]) -> Iterable[tuple[str, str]]:
    if not ({"date", "time", "birth", "departure", "arrival"} & set(_words(name)) or schema.get("format") == "date"):
        return
    patterns = (
        re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b"),
        re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b"),
        re.compile(r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})\b", re.I),
        re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[,]?\s+(?P<year>\d{4})\b", re.I),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            month = groups["month"]
            month = int(month) if month.isdigit() else _MONTH.get(month.casefold())
            if month is None:
                continue
            try:
                iso = datetime.date(int(groups["year"]), month, int(groups["day"])).isoformat()
            except ValueError:
                continue
            raw = match.group(0)
            value = iso if schema.get("format") == "date" else raw
            yield value, f"visible date {raw!r}" + (" normalized to schema date format" if value != raw else "")


def collect_public_values(
    request: str,
    results: list[Any],
    parameter_schema: dict[str, Any],
    parameter_name: str,
) -> list[dict[str, Any]]:
    """Return deduplicated ``{value, provenance}`` additions only.

    All values must match the public schema. Request IDs remain exact strings
    for string parameters; integer IDs use ``int``; ID values are never parsed
    through floating point. Ambiguous values remain separate for the chooser.
    """
    if not isinstance(request, str) or not isinstance(results, list) or not isinstance(parameter_schema, dict):
        return []
    rows: list[tuple[Any, str]] = []
    kind = parameter_schema.get("type")
    kinds = set(kind) if isinstance(kind, list) else {kind}
    aliases = _aliases(parameter_name)

    def add(value: Any, source: str) -> None:
        if type(value) is str and len(value) > _MAX_STRING:
            return
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return
        if _type_ok(value, parameter_schema):
            rows.append((value, source))

    # Values in the public schema remain selectable; the caller adds omission.
    if isinstance(parameter_schema.get("enum"), list):
        for value in parameter_schema["enum"]:
            add(value, "public schema enum")
    if "default" in parameter_schema and parameter_schema["default"] != "None":
        add(parameter_schema["default"], "public schema default (explicit value)")
    if "const" in parameter_schema:
        add(parameter_schema["const"], "public schema const")

    for index, result in enumerate(results):
        for value, path in _walk(result, f"public result[{index}]$"):
            add(value, path)
    for structure, source in _request_structures(request):
        for value, path in _walk(structure, source):
            add(value, path)

    # A quoted string is useful when nearby request words identify this field.
    for match in _QUOTED.finditer(request):
        value = next((part for part in match.groups() if part is not None), "").strip()
        if not value or "string" not in kinds:
            continue
        window = request[max(0, match.start() - 72):min(len(request), match.end() + 72)]
        if any(re.search(_label(alias), window, re.I) for alias in aliases):
            add(value, f"quoted request span at character {match.start()}")

    # Pull a literal directly following the matching public parameter label.
    for alias in aliases:
        pattern = re.compile(
            _label(alias) + r"\s*(?:(?:is|equals?|named|called|set\s+to|as|with|to|of)\s+|[=:]\s*|#\s*)?" + _LABELED_VALUE,
            re.I,
        )
        for match in pattern.finditer(request):
            raw = match.group("value").strip()
            quoted = raw[:1] in {'"', "'", "“", "‘"} and raw[-1:] in {'"', "'", "”", "’"}
            value = raw[1:-1] if quoted else raw.rstrip(".,;:!?)]}")
            if value.casefold() in _STOP:
                continue
            if "string" in kinds and not (not quoted and value.casefold() in {"true", "false", "yes", "no", "null"} and set(kinds) - {"string"}):
                if not (not quoted and re.fullmatch(r"[$€£¥]?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value) and set(kinds) - {"string"}):
                    add(value, f"field-labeled request value at character {match.start('value')}")
            if not quoted and value.casefold() == "null" and "null" in kinds:
                add(None, f"explicit null request value at character {match.start('value')}")
            if not quoted:
                number = _number(value, parameter_schema, parameter_name)
                if number is not None:
                    add(number, f"field-labeled numeric value at character {match.start('value')}")
                boolean = value.casefold() in {"true", "yes", "false", "no"} and "boolean" in kinds
                if boolean:
                    add(value.casefold() in {"true", "yes"}, f"field-labeled boolean value at character {match.start('value')}")

    # A bare amount can be grounded by an adjacent currency or numeric cue.
    if kinds.intersection({"integer", "number", "float"}):
        name_words = set(_words(parameter_name))
        for match in _NUMBER.finditer(request):
            before = request[max(0, match.start() - 48):match.start()].casefold()
            label_before = any(re.search(_label(alias), before) and re.fullmatch(
                r".*" + _label(alias) + r"\s*(?:(?:is|equals?|set\s+to|to|of|as|with)\s*)?[:=]?\s*[$€£¥#]?\s*",
                before,
                re.I,
            ) for alias in aliases)
            currency = bool(re.search(r"[$€£¥]\s*$", before))
            cue = bool(re.search(r"\b(?:amount|cost|count|distance|duration|height|hour|hours|index|limit|minute|minutes|number|percent|percentage|price|quantity|rate|rating|score|size|total|weight|year|years|id|identifier)\s*(?:is|equals?|of|to|=|:|#)?\s*$", before))
            if label_before or currency and name_words & _NUMBER_WORDS or cue and name_words & _NUMBER_WORDS | {"id", "identifier"}:
                number = _number(match.group(0), parameter_schema, parameter_name)
                if number is not None:
                    add(number, f"numeric request span {match.group(0)!r} at character {match.start()}")

    # Parse dates only for date-like parameters; preserve the source and format.
    if "string" in kinds:
        for value, source in _date_values(request, parameter_name, parameter_schema):
            add(value, source)

    # Unquoted proper names and identifiers retain their exact visible text.
    if "string" in kinds and set(_words(parameter_name)) & {"name", "person", "contact", "friend", "recipient", "user", "owner"}:
        for match in re.finditer(r"\b[A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){0,3}\b", request):
            parts = match.group(0).split()
            while parts and _words(parts[0])[:1] and _words(parts[0])[0] in _STOP:
                parts.pop(0)
            if parts:
                add(" ".join(parts), f"capitalized name span at character {match.start()}")
    if "string" in kinds and _is_id(parameter_name):
        for match in re.finditer(r"\b(?:id|identifier|reference|token|code)\b\s*(?:is|equals?|of|to|=|:|#)?\s*([A-Za-z0-9][A-Za-z0-9_.@/-]*)", request, re.I):
            add(match.group(1).rstrip(".,;:!?)]}"), f"explicit identifier at character {match.start(1)}")

    merged: dict[str, dict[str, Any]] = {}
    for value, source in rows:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        item = merged.setdefault(key, {"value": value, "provenance": []})
        if source not in item["provenance"]:
            item["provenance"].append(source)
    return list(merged.values())
