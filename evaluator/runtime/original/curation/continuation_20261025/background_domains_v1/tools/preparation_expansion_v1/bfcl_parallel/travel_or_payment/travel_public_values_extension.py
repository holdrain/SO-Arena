"""Field-specific public value extensions for travel dates and access tokens.

This module wraps the frozen BFCL public-value collector. It uses only the
visible request, public tool results, and public parameter schema. It neither
loads evaluator data nor uses source-reference routes.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import re
from typing import Any, Callable


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3,
    "march": 3, "apr": 4, "april": 4, "may": 5, "jun": 6,
    "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}
_MONTH_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
_DAY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_TOKEN_RE = re.compile(
    r"\b(?:access\s+(?:token|code)|account\s+token|token)\b"
    r"\s*(?:(?:is|of|equals?|=|:|,)\s*)?"
    r"(?:\(\s*|['\"“‘])?([A-Za-z0-9][A-Za-z0-9._~+/_-]*)",
    re.IGNORECASE,
)
_TOKEN_RESULT_PATH_RE = re.compile(
    r"(?:^|[.$\[])\s*(?:access_token|auth_token|token)(?=$|[.\]])",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CARD_NUMBER_RE = re.compile(
    r"\b(?:credit\s+card|travel\s+card|card)\b.{0,180}?\bnumber\b"
    r"\s*(?:(?:is|of|=|:)\s*)?['\"“‘]?"
    r"(?P<number>\d(?:[ -]?\d){11,18})", re.IGNORECASE | re.DOTALL,
)
_DATE_MONTH = (
    r"January|February|March|April|May|June|July|August|September|Sept|Sep|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Oct|Nov|Dec"
)
_DATE_MONTH_NUMBERS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}


def _schema_accepts_string(value: Any, schema: dict[str, Any]) -> bool:
    kinds = schema.get("type")
    if kinds is not None:
        allowed = set(kinds) if isinstance(kinds, list) else {kinds}
        if "string" not in allowed or type(value) is not str:
            return False
    if type(value) is not str:
        return False
    if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
        return False
    if "enum" in schema and not any(type(item) is str and item == value for item in schema["enum"]):
        return False
    if "const" in schema and not (type(schema["const"]) is str and schema["const"] == value):
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
            dt.date.fromisoformat(value)
        except ValueError:
            return False
    return True


def _visible_travel_dates(request: str) -> list[tuple[str, str]]:
    """Normalize explicit dates expressed in common public request wording."""
    output: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(year: str | int, month: str | int, day: str | int,
            start: int, end: int, label: str) -> None:
        try:
            month_number = (_DATE_MONTH_NUMBERS[str(month).casefold()]
                            if not str(month).isdigit() else int(month))
            value = dt.date(int(year), month_number, int(day)).isoformat()
        except (KeyError, TypeError, ValueError):
            return
        if value not in seen:
            seen.add(value)
            output.append((value, f"visible {label} at characters {start}:{end}"))

    # ISO dates are explicit and take priority over weaker date fragments such
    # as a card expiry like 12/25.
    for match in re.finditer(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b", request):
        add(match.group("year"), match.group("month"), match.group("day"),
            match.start(), match.end(), "ISO date")

    # Resolve wording that states both the relative date and its explicit
    # calendar month/year, e.g. "15th of next month, which is November of 2026".
    next_month_explicit = re.search(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?:the\s+)?next\s+month\b"
        rf".{{0,100}}?\b(?P<month>{_DATE_MONTH})\s+of\s+(?:the\s+)?(?P<year>\d{{4}})\b",
        request,
        re.IGNORECASE,
    )
    if next_month_explicit:
        add(next_month_explicit.group("year"), next_month_explicit.group("month"),
            next_month_explicit.group("day"), next_month_explicit.start(),
            next_month_explicit.end(), "relative day with explicit month-year date")

    ordinal_month = re.search(
        r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+of\s+(?:the\s+)?year\s+"
        r"(?P<year>\d{4})['’]s\s+(?P<month>\d{1,2}|[a-z]+)\s+month\b",
        request,
        re.IGNORECASE,
    )
    if ordinal_month:
        raw_month = ordinal_month.group("month").casefold()
        month = int(raw_month) if raw_month.isdigit() else _MONTH_ORDINALS.get(raw_month)
        try:
            value = dt.date(
                int(ordinal_month.group("year")), month or 0,
                int(ordinal_month.group("day")),
            ).isoformat()
        except ValueError:
            value = None
        if value:
            add(ordinal_month.group("year"), month, ordinal_month.group("day"),
                ordinal_month.start(), ordinal_month.end(), "ordinal-month date")

    relative = re.search(
        r"\b(?P<count>\d+|one|two|three|four|five)\s+days?\s+after\b.*?"
        r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+day\b.*?"
        r"\byear\s+(?P<year>\d{4})\b",
        request,
        re.IGNORECASE,
    )
    if relative:
        count_raw = relative.group("count").casefold()
        count = int(count_raw) if count_raw.isdigit() else _DAY_WORDS.get(count_raw, 0)
        # A month explicitly named as the next month is the best public cue;
        # otherwise use the first month name in the relative-date clause.
        next_month = re.search(
            r"\bnext\s+month\s+(?P<month>January|February|March|April|May|June|"
            r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|"
            r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b",
            request,
            re.IGNORECASE,
        )
        month_match = next_month or re.search(
            r"\b(January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
            r"Oct|Nov|Dec)\b",
            request,
            re.IGNORECASE,
        )
        month = _MONTHS.get(month_match.group(1).casefold()) if month_match else None
        if month is not None:
            try:
                value = (
                    dt.date(int(relative.group("year")), month, int(relative.group("day")))
                    + dt.timedelta(days=count)
                ).isoformat()
            except ValueError:
                value = None
            if value:
                if value not in seen:
                    seen.add(value)
                    output.append((value, f"visible relative-day date at characters {relative.start()}:{relative.end()}"))

    # Day-first forms such as "1st of December, 2026" and "19th Sept 2026".
    day_first = re.compile(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
        rf"(?P<month>{_DATE_MONTH})\b[,]?\s+(?:(?:in|of)\s+)?(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    )
    for match in day_first.finditer(request):
        add(match.group("year"), match.group("month"), match.group("day"),
            match.start(), match.end(), "day-month-year date")

    # Month-first forms such as "December 15, 2026" or "October 10th 2026".
    month_first = re.compile(
        rf"\b(?P<month>{_DATE_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
        rf"[,]?\s+(?:(?:in|of)\s+)?(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    )
    for match in month_first.finditer(request):
        add(match.group("year"), match.group("month"), match.group("day"),
            match.start(), match.end(), "month-day-year date")

    # Phrasing found in selected travel requests: "the 6th day of October in
    # 2026" and "last day of October 2026".
    day_of_month = re.compile(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+day\s+of\s+"
        rf"(?P<month>{_DATE_MONTH})\s+(?:in|of)\s+(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    )
    for match in day_of_month.finditer(request):
        add(match.group("year"), match.group("month"), match.group("day"),
            match.start(), match.end(), "day-of-month date")
    month_end = re.compile(
        rf"\b(?:last|final)\s+day\s+of\s+(?P<month>{_DATE_MONTH})[,]?\s+(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    )
    for match in month_end.finditer(request):
        month = _DATE_MONTH_NUMBERS[match.group("month").casefold()]
        year = int(match.group("year"))
        day = calendar.monthrange(year, month)[1]
        add(year, month, day, match.start(), match.end(), "month-end date")
    return output


def _special_visible_travel_date(request: str) -> tuple[str, str] | None:
    """Compatibility wrapper for callers that need the first parsed date."""
    dates = _visible_travel_dates(request)
    return dates[0] if dates else None


def _visible_access_token(request: str) -> tuple[str, str] | None:
    match = _TOKEN_RE.search(request)
    if not match:
        return None
    value = match.group(1).rstrip(".,;:!?)]}’”")
    if value.casefold() in {"is", "of", "the", "type", "expires", "scope", "and", "my"}:
        return None
    if not value:
        return None
    return value, f"visible request access/account token at character {match.start(1)}"


def _visible_card_number(request: str) -> tuple[str, str] | None:
    match = _CARD_NUMBER_RE.search(request)
    if not match:
        return None
    value = match.group("number").strip().rstrip(".,;:!?)]}’”")
    digits = re.sub(r"\D", "", value)
    if not 12 <= len(digits) <= 19:
        return None
    return value, f"visible request card-number span at characters {match.start('number')}:{match.end('number')}"


def _visible_expiration_date(request: str, schema: dict[str, Any]) -> tuple[str, str] | None:
    """Normalize an explicit MM/YY value only when public schema says MM/YYYY."""
    description = str(schema.get("description", ""))
    if "MM/YYYY" not in description.upper():
        return None
    match = re.search(
        r"\b(?P<month>\d{1,2})/(?P<year>\d{2}|\d{4})\b",
        request,
        re.IGNORECASE,
    )
    if not match:
        return None
    month = int(match.group("month"))
    raw_year = match.group("year")
    if not 1 <= month <= 12:
        return None
    year = raw_year if len(raw_year) == 4 else "20" + raw_year
    value = f"{month:02d}/{year}"
    if not _schema_accepts_string(value, schema):
        return None
    return value, f"visible request expiry normalized to documented MM/YYYY schema"


def _source_rows(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in base_rows:
        if not isinstance(row, dict) or "value" not in row:
            continue
        provenance = row.get("provenance", [])
        if isinstance(provenance, str):
            provenance = [provenance]
        out.append({"value": row["value"], "provenance": list(provenance)})
    return out


def _add(rows: list[dict[str, Any]], value: str, provenance: str,
         schema: dict[str, Any]) -> None:
    if not _schema_accepts_string(value, schema):
        return
    for row in rows:
        if type(row.get("value")) is str and row["value"] == value:
            if provenance not in row["provenance"]:
                row["provenance"].append(provenance)
            return
    rows.append({"value": value, "provenance": [provenance]})


def extend_public_values(
    base_collector: Callable[..., list[dict[str, Any]]],
    request: str,
    results: list[Any],
    parameter_schema: dict[str, Any],
    parameter_name: str,
) -> list[dict[str, Any]]:
    """Add visible travel dates/tokens and reject cross-field string leakage.

    For ``travel_date``, only valid ISO dates, public schema values, and a
    request-derived date are retained. This prevents airport codes and error
    messages from becoming date candidates.

    For ``access_token``, only public schema values, values from explicitly
    named public token-result fields, and the visible request credential are
    retained. This prevents card IDs, airport codes, and tool error strings
    from being offered as credentials.
    """
    if not isinstance(request, str) or not isinstance(results, list) or not isinstance(parameter_schema, dict):
        return []
    base_rows = _source_rows(base_collector(
        request=request,
        results=results,
        parameter_schema=parameter_schema,
        parameter_name=parameter_name,
    ))
    field = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", parameter_name).casefold()
    if field == "travel_date":
        rows = []
        for row in base_rows:
            value = row["value"]
            provenance = row["provenance"]
            schema_value = any(item.startswith("public schema ") for item in provenance)
            if schema_value:
                _add(rows, value, provenance[0], parameter_schema)
            elif type(value) is str and _ISO_DATE_RE.fullmatch(value):
                try:
                    dt.date.fromisoformat(value)
                except ValueError:
                    continue
                for source in provenance or ["public ISO travel date"]:
                    _add(rows, value, source, parameter_schema)
        for value, provenance in _visible_travel_dates(request):
            _add(rows, value, provenance, parameter_schema)
        return rows
    if field == "access_token":
        rows = []
        for row in base_rows:
            value = row["value"]
            provenance = row["provenance"]
            if any(item.startswith("public schema ") for item in provenance):
                for source in provenance:
                    _add(rows, value, source, parameter_schema)
                continue
            for source in provenance:
                if source.startswith("public result[") and _TOKEN_RESULT_PATH_RE.search(source):
                    _add(rows, value, source, parameter_schema)
        token = _visible_access_token(request)
        if token:
            _add(rows, token[0], token[1], parameter_schema)
        return rows
    if field == "card_number":
        rows = []
        for row in base_rows:
            value = row["value"]
            provenance = row["provenance"]
            if any(item.startswith("public schema ") for item in provenance):
                for source in provenance:
                    _add(rows, value, source, parameter_schema)
                continue
            if any(item.startswith("public result[") and re.search(r"card[_-]?number", item, re.I)
                   for item in provenance):
                for source in provenance:
                    _add(rows, value, source, parameter_schema)
        card_number = _visible_card_number(request)
        if card_number:
            _add(rows, card_number[0], card_number[1], parameter_schema)
        return rows
    return base_rows

