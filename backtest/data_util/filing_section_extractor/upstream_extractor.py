#!/usr/bin/env python3
"""
Standalone filing section extractor.

This script is intentionally self-contained. It reads clean-text parquet filings,
extracts a fixed item set for 10-K / 10-Q, runs lightweight item-based audits,
and writes compact run outputs.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - runtime dependency guard
    pq = None


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = ROOT_DIR / "datasets" / "sp500_2000_2025_parquet"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "scripts" / "python" / "filing_section_extractor" / "outputs" / "runs"
DEFAULT_FORMS = ("10-K", "10-Q")
FORM_TO_BUCKET = {
    "10-K": "filingk",
    "10-Q": "filingq",
}
PARQUET_ROW_COLUMNS = (
    "date",
    "symbol",
    "cik",
    "filing_idx",
    "filing_text",
    "year",
    "accession",
)
BODY_PROFILE_NARRATIVE = "narrative"
BODY_PROFILE_TABLE_HEAVY = "table_heavy"
BODY_PROFILE_SHORT_DISCLOSURE = "short_disclosure"
BODY_PROFILES = {
    BODY_PROFILE_NARRATIVE,
    BODY_PROFILE_TABLE_HEAVY,
    BODY_PROFILE_SHORT_DISCLOSURE,
}
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
ROMAN_PARTS = {"I", "II", "III", "IV"}
ITEM_PREFIX_RE = re.compile(
    r"(?i)\bi\s*t\s*e\s*m(?:s)?\s*\.?\s*([0-9]{1,2}(?:\s*[A-Za-z])?)"
)
ITEM_PREFIX_START_RE = re.compile(
    r"(?i)^i\s*t\s*e\s*m(?:s)?\s*\.?\s*([0-9]{1,2}(?:\s*[A-Za-z])?)\b"
)
PART_RE = re.compile(r"(?i)\bpart\b\s*([ivx]+)\b")
PART_START_RE = re.compile(r"(?i)^part\s*([ivx]+)\b")
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True)
class ItemSpec:
    item_key: str
    form: str
    part_name: str | None
    item_code: str
    canonical_heading: str
    heading_fragments: tuple[str, ...]
    boundary_candidates: tuple[str, ...]
    body_profile: str
    is_primary_target: bool
    min_chars: int
    allow_short_stub: bool
    allow_reference_only: bool
    require_late_document_position: bool
    prefer_prose_start: bool
    allow_table_start: bool


@dataclass(frozen=True)
class FilingRow:
    date: str
    symbol: str
    cik: str
    accession: str
    form: str
    year: int
    filing_idx: int
    filing_text: str
    source_file: str
    source_row_idx: int


@dataclass(frozen=True)
class HeadingCandidate:
    line_index: int
    score: float
    heading_preview: str
    fragment_hits: int
    detected_part_name: str | None
    prose_lines_after_heading: int
    body_char_count: int
    body_nonempty_line_count: int
    is_toc_candidate: bool
    body_preview: str
    boundary_line_number: int | None
    boundary_preview: str | None
    score_notes: tuple[str, ...]
    body_line_start: int
    body_line_end: int

    @property
    def line_number(self) -> int:
        return self.line_index + 1

    def sort_key(self) -> tuple[float, int, int]:
        return (self.score, self.prose_lines_after_heading, self.line_index)

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "line_index": self.line_index,
            "line_number": self.line_number,
            "score": self.score,
            "heading_preview": self.heading_preview,
            "fragment_hits": self.fragment_hits,
            "detected_part_name": self.detected_part_name,
            "prose_lines_after_heading": self.prose_lines_after_heading,
            "body_char_count": self.body_char_count,
            "body_nonempty_line_count": self.body_nonempty_line_count,
            "is_toc_candidate": self.is_toc_candidate,
            "body_preview": self.body_preview,
            "boundary_line_number": self.boundary_line_number,
            "boundary_preview": self.boundary_preview,
            "score_notes": list(self.score_notes),
            "body_line_start": self.body_line_start,
            "body_line_end": self.body_line_end,
        }


@dataclass(frozen=True)
class SectionOverlap:
    left_item_key: str
    right_item_key: str
    left_line_start: int
    left_line_end: int
    right_line_start: int
    right_line_end: int
    overlap_line_count: int
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ItemAuditRecord:
    item_key: str
    body_profile: str
    status: str
    flags: tuple[str, ...]
    char_count: int
    body_char_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "body_profile": self.body_profile,
            "status": self.status,
            "flags": list(self.flags),
            "char_count": self.char_count,
            "body_char_count": self.body_char_count,
        }


# Supported item registry.
# We keep it local to this file so the script remains standalone and easy to ship.
ITEM_SPECS: dict[str, list[ItemSpec]] = {
    "10-K": [
        ItemSpec("item_1", "10-K", "Part I", "1", "Business", ("business",), ("item_1a", "item_1b", "item_1c", "item_2"), BODY_PROFILE_NARRATIVE, False, 500, False, False, False, True, False),
        ItemSpec("item_1a", "10-K", "Part I", "1A", "Risk Factors", ("risk", "factors"), ("item_1b", "item_1c", "item_2"), BODY_PROFILE_NARRATIVE, False, 300, False, False, False, True, False),
        ItemSpec(
            "item_5",
            "10-K",
            "Part II",
            "5",
            "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
            ("market", "registrant", "common", "equity"),
            ("item_6", "item_7"),
            BODY_PROFILE_NARRATIVE,
            False,
            200,
            False,
            False,
            True,
            True,
            False,
        ),
        ItemSpec(
            "item_7",
            "10-K",
            "Part II",
            "7",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
            ("management", "discussion", "analysis"),
            ("item_7a", "item_8", "item_9"),
            BODY_PROFILE_NARRATIVE,
            True,
            1000,
            False,
            False,
            True,
            True,
            False,
        ),
        ItemSpec(
            "item_7a",
            "10-K",
            "Part II",
            "7A",
            "Quantitative and Qualitative Disclosures About Market Risk",
            ("quantitative", "qualitative", "market", "risk"),
            ("item_8", "item_9", "item_9a"),
            BODY_PROFILE_NARRATIVE,
            False,
            120,
            False,
            False,
            True,
            True,
            False,
        ),
        ItemSpec(
            "item_8",
            "10-K",
            "Part II",
            "8",
            "Financial Statements and Supplementary Data",
            ("financial", "statements"),
            ("item_9", "item_9a", "item_9b"),
            BODY_PROFILE_TABLE_HEAVY,
            False,
            500,
            False,
            True,
            True,
            False,
            True,
        ),
        ItemSpec(
            "item_9a",
            "10-K",
            "Part II",
            "9A",
            "Controls and Procedures",
            ("controls", "procedures"),
            ("item_9b", "item_9c", "item_10"),
            BODY_PROFILE_SHORT_DISCLOSURE,
            False,
            120,
            True,
            True,
            True,
            False,
            False,
        ),
    ],
    "10-Q": [
        ItemSpec("part_i_item_1", "10-Q", "Part I", "1", "Financial Statements", ("financial", "statements"), ("part_i_item_2", "part_i_item_3"), BODY_PROFILE_TABLE_HEAVY, False, 500, False, False, False, False, True),
        ItemSpec(
            "part_i_item_2",
            "10-Q",
            "Part I",
            "2",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
            ("management", "discussion", "analysis"),
            ("part_i_item_3", "part_i_item_4", "part_ii_item_1"),
            BODY_PROFILE_NARRATIVE,
            True,
            800,
            False,
            False,
            True,
            True,
            False,
        ),
        ItemSpec(
            "part_i_item_3",
            "10-Q",
            "Part I",
            "3",
            "Quantitative and Qualitative Disclosures About Market Risk",
            ("quantitative", "qualitative", "market", "risk"),
            ("part_i_item_4", "part_ii_item_1"),
            BODY_PROFILE_NARRATIVE,
            False,
            100,
            False,
            False,
            True,
            True,
            False,
        ),
        ItemSpec("part_i_item_4", "10-Q", "Part I", "4", "Controls and Procedures", ("controls", "procedures"), ("part_ii_item_1", "part_ii_item_1a"), BODY_PROFILE_NARRATIVE, False, 100, False, False, True, True, False),
        ItemSpec("part_ii_item_1", "10-Q", "Part II", "1", "Legal Proceedings", ("legal", "proceedings"), ("part_ii_item_1a", "part_ii_item_2", "part_ii_item_3"), BODY_PROFILE_SHORT_DISCLOSURE, False, 120, True, True, True, False, False),
        ItemSpec("part_ii_item_1a", "10-Q", "Part II", "1A", "Risk Factors", ("risk", "factors"), ("part_ii_item_2", "part_ii_item_3", "part_ii_item_4"), BODY_PROFILE_NARRATIVE, False, 150, False, False, True, True, False),
        ItemSpec(
            "part_ii_item_2",
            "10-Q",
            "Part II",
            "2",
            "Unregistered Sales of Equity Securities and Use of Proceeds",
            ("unregistered", "sales", "equity", "proceeds"),
            ("part_ii_item_3", "part_ii_item_4", "part_ii_item_5"),
            BODY_PROFILE_SHORT_DISCLOSURE,
            False,
            120,
            True,
            True,
            True,
            False,
            False,
        ),
    ],
}


def validate_item_specs() -> None:
    for form, specs in ITEM_SPECS.items():
        seen: set[str] = set()
        for spec in specs:
            if spec.form != form:
                raise ValueError(f"Item spec form mismatch: registry={form} spec={spec.form} item={spec.item_key}")
            if spec.item_key in seen:
                raise ValueError(f"Duplicate item spec: {spec.item_key}")
            if spec.body_profile not in BODY_PROFILES:
                raise ValueError(f"Unsupported body profile: {spec.body_profile} for {spec.item_key}")
            seen.add(spec.item_key)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# CLI and request normalization.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone clean-text filing section extractor")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help=f"Parquet root, default: {DEFAULT_INPUT_ROOT}")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=f"Run output root, default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    parser.add_argument("--tickers", nargs="+", help="Target tickers, e.g. TSLA AMZN")
    parser.add_argument("--year-from", type=int, required=True, help="Start year, inclusive")
    parser.add_argument("--year-to", type=int, required=True, help="End year, inclusive")
    parser.add_argument("--forms", nargs="+", default=list(DEFAULT_FORMS), help="Forms to include, default: 10-K 10-Q")
    parser.add_argument("--items", nargs="+", help="Requested item keys; default is all supported items for the selected forms")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit across all selected input files")
    parser.add_argument(
        "--save-full-filing-text",
        action="store_true",
        help="Include the raw filing_text in per-row extraction artifacts. Default is off to reduce output size and write I/O.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only collect candidates and write manifest")
    return parser.parse_args()


def normalize_form(value: str) -> str:
    form = str(value).strip().upper()
    if form not in FORM_TO_BUCKET:
        raise ValueError(f"Unsupported form: {value}")
    return form


def normalize_ticker(value: str) -> str:
    return str(value).strip().upper()


def normalize_item_code(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", str(value)).upper()
    return cleaned or None


def normalize_part_name(value: str | None) -> str | None:
    if value is None:
        return None
    match = PART_RE.search(str(value))
    if not match:
        return None
    return f"Part {match.group(1).upper()}"


def normalized_text(text: str) -> str:
    normalized = html.unescape(text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.translate(DASH_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def collapsed_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalized_text(text).lower())


def preview_text(text: str, max_chars: int = 200) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def promote_status(current: str, candidate: str) -> str:
    if STATUS_RANK[candidate] > STATUS_RANK[current]:
        return candidate
    return current


def parse_item_code_from_leading_text(text: str | None) -> str | None:
    if not text:
        return None
    match = ITEM_PREFIX_START_RE.search(normalized_text(text))
    if not match:
        return None
    return normalize_item_code(match.group(1))


def parse_part_name_from_leading_text(text: str | None) -> str | None:
    if not text:
        return None
    match = PART_START_RE.search(normalized_text(text))
    if not match:
        return None
    return f"Part {match.group(1).upper()}"


def looks_like_explicit_heading_lead(text: str | None) -> bool:
    return (
        parse_item_code_from_leading_text(text) is not None
        or parse_part_name_from_leading_text(text) is not None
    )


def looks_like_prose_line(text: str) -> bool:
    normalized = normalized_text(text)
    if len(normalized) < 45:
        return False
    if normalized.isupper():
        return False
    if not re.search(r"[a-z]{3,}", normalized, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"[.;:?!]", normalized)) or len(normalized.split()) >= 10


def looks_like_toc_line(text: str) -> bool:
    normalized = normalized_text(text).lower()
    if not normalized:
        return False
    if "table of contents" in normalized:
        return True
    if PAGE_NUMBER_RE.fullmatch(normalized):
        return True
    if normalized.endswith(" page"):
        return True
    return False


def ends_with_page_number(text: str) -> bool:
    return bool(re.search(r"\b\d{1,4}\s*$", normalized_text(text)))


def is_decorative_line(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    return not any(char.isalnum() for char in stripped)


def available_item_keys(forms: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for form in forms:
        for spec in ITEM_SPECS[form]:
            keys.append(spec.item_key)
    return keys


def resolve_requested_items(forms: list[str], requested_items: list[str] | None) -> list[str]:
    allowed = set(available_item_keys(forms))
    if not requested_items:
        return available_item_keys(forms)
    out: list[str] = []
    seen: set[str] = set()
    for item in requested_items:
        key = str(item).strip().lower()
        if key not in allowed:
            raise ValueError(f"Unsupported item for selected forms: {item}")
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def item_specs_for_request(forms: list[str], requested_items: list[str]) -> list[ItemSpec]:
    selected = set(requested_items)
    specs: list[ItemSpec] = []
    for form in forms:
        for spec in ITEM_SPECS[form]:
            if spec.item_key in selected:
                specs.append(spec)
    return specs


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def input_files_for_forms(input_root: Path, year_from: int, year_to: int, forms: list[str]) -> list[tuple[str, int, Path]]:
    files: list[tuple[str, int, Path]] = []
    for year in range(year_from, year_to + 1):
        for form in forms:
            bucket = FORM_TO_BUCKET[form]
            path = input_root / bucket / f"year={year}" / "part-000.parquet"
            if path.exists():
                files.append((form, year, path))
    return files


def iter_filing_rows(
    *,
    input_root: Path,
    year_from: int,
    year_to: int,
    forms: list[str],
    tickers: set[str] | None,
    limit: int,
) -> Iterable[FilingRow]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet input")

    emitted = 0
    for form, year, path in input_files_for_forms(input_root, year_from, year_to, forms):
        parquet_file = pq.ParquetFile(path)
        source_row_idx = 0
        for row_group_index in range(parquet_file.num_row_groups):
            symbol_table = parquet_file.read_row_group(row_group_index, columns=["symbol"])
            symbol_values = symbol_table.column("symbol").to_pylist()
            row_group_size = len(symbol_values)
            row_offset = source_row_idx
            source_row_idx += row_group_size

            selected_indexes: list[int] = []
            for index, raw_symbol in enumerate(symbol_values):
                symbol = normalize_ticker(raw_symbol or "")
                if tickers and symbol not in tickers:
                    continue
                selected_indexes.append(index)

            if not selected_indexes:
                continue

            row_group_table = parquet_file.read_row_group(row_group_index, columns=list(PARQUET_ROW_COLUMNS))
            table = row_group_table.to_pydict()
            n = len(table.get("symbol", []))
            for index in selected_indexes:
                symbol = normalize_ticker(table["symbol"][index] or "")
                row = FilingRow(
                    date=str(table["date"][index]),
                    symbol=symbol,
                    cik=str(table["cik"][index] or "").strip(),
                    accession=str(table.get("accession", [""] * n)[index] or "").strip(),
                    form=form,
                    year=int(table.get("year", [year] * n)[index] or year),
                    filing_idx=int(table["filing_idx"][index]),
                    filing_text=str(table["filing_text"][index] or ""),
                    source_file=str(path),
                    source_row_idx=row_offset + index,
                )
                yield row
                emitted += 1
                if limit > 0 and emitted >= limit:
                    return


def next_nonempty_lines(lines: list[str], start_index: int, max_lines: int = 6) -> list[str]:
    results: list[str] = []
    for index in range(start_index, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        results.append(stripped)
        if len(results) >= max_lines:
            break
    return results


def previous_nonempty_lines(lines: list[str], start_index: int, max_lines: int = 4) -> list[str]:
    results: list[str] = []
    for index in range(start_index - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        results.append(stripped)
        if len(results) >= max_lines:
            break
    return list(reversed(results))


def parse_item_code_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = ITEM_PREFIX_RE.search(normalized_text(text))
    if not match:
        return None
    return normalize_item_code(match.group(1))


def parse_part_name_from_text(text: str | None) -> str | None:
    if not text:
        return None
    return normalize_part_name(text)


def looks_like_heading_continuation_candidate(text: str, spec: ItemSpec) -> bool:
    normalized = normalized_text(text)
    if not normalized:
        return False
    if looks_like_toc_line(normalized):
        return False
    if heading_fragment_score(normalized, spec) >= 1:
        return True
    if len(normalized) <= 72 and normalized.isupper():
        return True
    if normalized in {"(unaudited)", "(audited)"}:
        return True
    return False


def heading_window_lines(
    lines: list[str],
    index: int,
    max_nonempty: int = 5,
    spec: ItemSpec | None = None,
) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    for current_index in range(index, len(lines)):
        stripped = lines[current_index].strip()
        if not stripped:
            continue
        if results:
            current_item = parse_item_code_from_leading_text(stripped)
            existing_item = parse_item_code_from_text(" ".join(item for _, item in results))
            current_part = parse_part_name_from_leading_text(stripped)
            existing_part = parse_part_name_from_text(" ".join(item for _, item in results))
            if current_item and existing_item and current_item != existing_item:
                break
            if current_part and existing_part and current_part != existing_part:
                break
            if looks_like_prose_line(stripped):
                break
            if spec is not None and not looks_like_heading_continuation_candidate(stripped, spec):
                break
        results.append((current_index, stripped))
        if len(results) >= max_nonempty:
            break
        if len(stripped) >= 140:
            break
    return results


def heading_window_text(lines: list[str], index: int, max_nonempty: int = 5, spec: ItemSpec | None = None) -> str:
    return " ".join(text for _, text in heading_window_lines(lines, index, max_nonempty=max_nonempty, spec=spec))


def parse_part_name_near_index(lines: list[str], index: int) -> str | None:
    current = parse_part_name_from_leading_text(lines[index])
    if current:
        return current
    for line in reversed(previous_nonempty_lines(lines, index, max_lines=3)):
        part_name = parse_part_name_from_leading_text(line)
        if part_name:
            return part_name
    return None


def heading_fragment_score(window_text: str, spec: ItemSpec) -> int:
    collapsed = collapsed_text(window_text)
    return sum(1 for fragment in spec.heading_fragments if collapsed_text(fragment) in collapsed)


def item_key_to_part_and_code(item_key: str) -> tuple[str | None, str | None]:
    if item_key.startswith("part_i_item_"):
        return "Part I", normalize_item_code(item_key.removeprefix("part_i_item_"))
    if item_key.startswith("part_ii_item_"):
        return "Part II", normalize_item_code(item_key.removeprefix("part_ii_item_"))
    if item_key.startswith("item_"):
        return None, normalize_item_code(item_key.removeprefix("item_"))
    return None, None


def standardized_heading(spec: ItemSpec) -> str:
    if spec.part_name:
        return f"{spec.part_name} Item {spec.item_code}. {spec.canonical_heading}"
    return f"Item {spec.item_code}. {spec.canonical_heading}"


def locate_heading_end(lines: list[str], start_index: int, spec: ItemSpec) -> int:
    window = heading_window_lines(lines, start_index, max_nonempty=5, spec=spec)
    if not window:
        return start_index
    return window[-1][0]


def body_score_after_heading(lines: list[str], start_index: int, spec: ItemSpec) -> tuple[float, int]:
    heading_end = locate_heading_end(lines, start_index, spec)
    probe = next_nonempty_lines(lines, heading_end + 1, max_lines=8)
    prose_lines = sum(1 for line in probe if looks_like_prose_line(line))
    toc_lines = sum(1 for line in probe[:4] if looks_like_toc_line(line))
    score = prose_lines * 12.0 - toc_lines * 20.0
    if probe and looks_like_prose_line(probe[0]):
        score += 18.0
    return score, prose_lines


def looks_like_body_toc_cluster(body_text: str) -> bool:
    probe_lines = body_open_lines(body_text, max_lines=8)
    if not probe_lines:
        return False
    toc_like = 0
    shortish = 0
    for line in probe_lines:
        normalized = normalized_text(line)
        if looks_like_toc_line(normalized):
            toc_like += 1
        if len(normalized) <= 90 and not looks_like_prose_line(normalized):
            shortish += 1
        if re.search(r"\b\d{1,4}\s*$", normalized):
            toc_like += 1
    return toc_like >= 2 or (shortish >= 4 and toc_like >= 1)


def leading_cross_item_markers(body_text: str, spec: ItemSpec, max_lines: int = 10) -> list[str]:
    markers: list[str] = []
    target_item = normalize_item_code(spec.item_code)
    target_part = normalize_part_name(spec.part_name)
    for line in body_open_lines(body_text, max_lines=max_lines):
        part_name = parse_part_name_from_leading_text(line)
        item_code = parse_item_code_from_leading_text(line)
        if part_name and target_part is not None and part_name != target_part:
            markers.append(f"part:{part_name}")
            continue
        if item_code and item_code != target_item:
            markers.append(f"item:{item_code}")
    return markers


def profile_candidate_score(
    *,
    spec: ItemSpec,
    line_index: int,
    total_line_count: int,
    fragment_hits: int,
    part_name: str | None,
    heading_text: str,
    body_score: float,
    prose_lines: int,
    body_text: str,
    body_char_count: int,
    body_nonempty_line_count: int,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 50.0 + fragment_hits * 8.0 + body_score

    target_part = normalize_part_name(spec.part_name)
    if part_name == target_part and target_part is not None:
        score += 20.0
        notes.append("part_match")
    if looks_like_explicit_heading_lead(heading_text):
        score += 8.0
        notes.append("explicit_heading_prefix")

    early_absolute = line_index <= 220
    early_relative = total_line_count > 0 and line_index <= max(80, int(total_line_count * 0.12))
    is_early = early_absolute or early_relative
    body_toc_cluster = looks_like_body_toc_cluster(body_text)
    cross_item_markers = leading_cross_item_markers(body_text, spec)
    is_short_stub = looks_like_short_stub_value(body_text)
    is_reference_stub = looks_like_reference_stub(body_text)

    if spec.body_profile == BODY_PROFILE_NARRATIVE:
        score += min(body_char_count / 500.0, 25.0)
        if prose_lines > 0:
            score += 12.0
            notes.append("prose_open")
        if body_char_count == 0:
            score -= 120.0
            notes.append("heading_only_penalty")
        if is_early and body_char_count < max(spec.min_chars // 2, 160):
            score -= 80.0
            notes.append("early_small_body_penalty")
        if body_toc_cluster:
            score -= 90.0
            notes.append("toc_cluster_penalty")
        if spec.require_late_document_position and is_early:
            score -= 30.0
            notes.append("late_position_penalty")

    elif spec.body_profile == BODY_PROFILE_TABLE_HEAVY:
        score += min(body_char_count / 350.0, 30.0)
        score += min(body_nonempty_line_count * 2.0, 16.0)
        if body_char_count == 0:
            score -= 140.0
            notes.append("heading_only_penalty")
        if body_toc_cluster:
            score -= 110.0
            notes.append("toc_cluster_penalty")
        if len(cross_item_markers) >= 2:
            score -= 240.0
            notes.append("cross_item_head_penalty")
        elif cross_item_markers:
            score -= 80.0
            notes.append("single_cross_item_head_penalty")
        if spec.require_late_document_position and is_early:
            score -= 45.0
            notes.append("late_position_penalty")
        if body_nonempty_line_count >= 6:
            score += 10.0
            notes.append("substantial_table_body")

    elif spec.body_profile == BODY_PROFILE_SHORT_DISCLOSURE:
        if body_char_count == 0:
            score -= 140.0
            notes.append("heading_only_penalty")
        else:
            score += min(body_char_count / 120.0, 18.0)
        if spec.allow_short_stub and is_short_stub:
            score += 28.0
            notes.append("allowed_short_stub")
        if spec.allow_reference_only and is_reference_stub:
            score += 18.0
            notes.append("allowed_reference_only")
        if body_toc_cluster:
            score -= 110.0
            notes.append("toc_cluster_penalty")
        if spec.require_late_document_position and is_early:
            score -= 55.0
            notes.append("late_position_penalty")

    return round(score, 3), notes


def apply_toc_heading_penalty(*, score: float, score_notes: list[str], spec: ItemSpec, is_toc_candidate: bool) -> tuple[float, list[str]]:
    if not is_toc_candidate:
        return score, score_notes
    adjusted_notes = list(score_notes)
    if spec.body_profile == BODY_PROFILE_TABLE_HEAVY:
        adjusted_notes.append("toc_heading_penalty")
        return score - 25.0, adjusted_notes
    adjusted_notes.append("toc_heading_penalty")
    return score - 120.0, adjusted_notes


# Extraction pipeline: locate the best heading candidate, find the next item
# boundary, then materialize a clean section payload for each requested item.
def build_heading_candidate(
    *,
    lines: list[str],
    index: int,
    spec: ItemSpec,
    total_line_count: int,
    heading_text: str,
    heading_window: str,
    fragment_hits: int,
    part_name: str | None,
) -> HeadingCandidate:
    # Candidate selection is body-aware on purpose. Title matches alone are not
    # reliable because TOC rows and cross-references often mention the same item.
    body_score, prose_lines = body_score_after_heading(lines, index, spec)
    provisional_boundary_index, provisional_boundary_meta = find_next_boundary(lines, index, spec)
    _, provisional_body_text, provisional_body_line_start, provisional_body_line_end, provisional_body_nonempty_line_count = (
        build_section_text(lines, index, provisional_boundary_index, spec)
    )
    provisional_body_char_count = len(provisional_body_text)

    score, score_notes = profile_candidate_score(
        spec=spec,
        line_index=index,
        total_line_count=total_line_count,
        fragment_hits=fragment_hits,
        part_name=part_name,
        heading_text=heading_text,
        body_score=body_score,
        prose_lines=prose_lines,
        body_text=provisional_body_text,
        body_char_count=provisional_body_char_count,
        body_nonempty_line_count=provisional_body_nonempty_line_count,
    )
    is_toc_candidate = looks_like_toc_heading_candidate(lines, index)
    score, score_notes = apply_toc_heading_penalty(
        score=score,
        score_notes=score_notes,
        spec=spec,
        is_toc_candidate=is_toc_candidate,
    )

    return HeadingCandidate(
        line_index=index,
        score=round(score, 3),
        heading_preview=preview_text(heading_window, 240),
        fragment_hits=fragment_hits,
        detected_part_name=part_name,
        prose_lines_after_heading=prose_lines,
        body_char_count=provisional_body_char_count,
        body_nonempty_line_count=provisional_body_nonempty_line_count,
        is_toc_candidate=is_toc_candidate,
        body_preview=preview_text(provisional_body_text, 160),
        boundary_line_number=None if provisional_boundary_index is None else provisional_boundary_index + 1,
        boundary_preview=provisional_boundary_meta.get("boundary_preview"),
        score_notes=tuple(score_notes),
        body_line_start=provisional_body_line_start,
        body_line_end=provisional_body_line_end,
    )


def select_preferred_heading_candidates(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    preferred = [candidate for candidate in candidates if not candidate.is_toc_candidate]
    return preferred or candidates


def find_best_item_start(lines: list[str], spec: ItemSpec) -> tuple[int | None, dict[str, Any]]:
    candidates: list[HeadingCandidate] = []
    target_item = normalize_item_code(spec.item_code)
    target_part = normalize_part_name(spec.part_name)
    total_line_count = len(lines)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if not looks_like_explicit_heading_lead(line):
            continue
        window_text = heading_window_text(lines, index, max_nonempty=5, spec=spec)
        if not window_text:
            continue

        item_code = parse_item_code_from_text(window_text)
        if item_code != target_item:
            continue

        part_name = parse_part_name_near_index(lines, index)
        if target_part is not None and part_name not in {target_part, None}:
            continue

        fragment_hits = heading_fragment_score(window_text, spec)
        if fragment_hits == 0:
            continue

        candidates.append(
            build_heading_candidate(
                lines=lines,
                index=index,
                spec=spec,
                total_line_count=total_line_count,
                heading_text=line,
                heading_window=window_text,
                fragment_hits=fragment_hits,
                part_name=part_name,
            )
        )

    if not candidates:
        return None, {"matched": False, "reason": "no_heading_candidate"}

    # If at least one non-TOC heading exists, we discard TOC candidates entirely.
    candidates = select_preferred_heading_candidates(candidates)

    best = max(candidates, key=lambda item: item.sort_key())
    return int(best.line_index), {"matched": True, "best_candidate": best.to_debug_payload(), "candidate_count": len(candidates)}


def matches_boundary(lines: list[str], index: int, boundary_part: str | None, boundary_item_code: str | None) -> bool:
    current_text = normalized_text(lines[index])
    if not looks_like_explicit_heading_lead(current_text):
        return False
    current_item_code = parse_item_code_from_leading_text(current_text)
    current_part_name = parse_part_name_from_leading_text(current_text)
    if current_item_code is None and current_part_name is None:
        return False

    window_text = heading_window_text(lines, index, max_nonempty=4)
    if not window_text:
        return False
    item_code = current_item_code or parse_item_code_from_text(window_text)
    part_name = current_part_name or parse_part_name_near_index(lines, index)
    if boundary_item_code is not None and item_code != boundary_item_code:
        return False
    if boundary_part is not None and part_name not in {boundary_part, None}:
        return False
    return True


def looks_like_toc_heading_candidate(lines: list[str], index: int, lookback: int = 30) -> bool:
    line = lines[index]
    if not looks_like_explicit_heading_lead(line):
        return False
    if not ends_with_page_number(line):
        return False
    start = max(0, index - lookback)
    recent_window = lines[start:index]
    return any("table of contents" in normalized_text(candidate).lower() for candidate in recent_window)


def find_next_boundary(lines: list[str], start_index: int, spec: ItemSpec) -> tuple[int | None, dict[str, Any]]:
    boundary_keys = spec.boundary_candidates
    boundary_specs = [item_key_to_part_and_code(item_key) for item_key in boundary_keys]

    for index in range(start_index + 1, len(lines)):
        if not lines[index].strip():
            continue
        for boundary_key, (boundary_part, boundary_item_code) in zip(boundary_keys, boundary_specs):
            if matches_boundary(lines, index, boundary_part, boundary_item_code):
                if looks_like_toc_heading_candidate(lines, index):
                    continue
                return index, {
                    "matched": True,
                    "boundary_item_key": boundary_key,
                    "boundary_line_number": index + 1,
                    "boundary_preview": preview_text(heading_window_text(lines, index, max_nonempty=4), 200),
                }
    return None, {"matched": False, "reason": "no_boundary_found"}


def normalize_body_lines(body_lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in body_lines:
        normalized = line.rstrip()
        if not normalized.strip():
            cleaned.append("")
            continue
        # Decorative separator rows appear frequently around headings and short
        # disclosures, but they are not meaningful body content.
        if is_decorative_line(normalized):
            continue
        if PAGE_NUMBER_RE.fullmatch(normalized.strip()):
            continue
        if normalized_text(normalized).lower() == "table of contents":
            continue
        cleaned.append(normalized)

    while cleaned and not cleaned[0].strip():
        cleaned = cleaned[1:]
    while cleaned and not cleaned[-1].strip():
        cleaned = cleaned[:-1]
    return cleaned


def trim_leading_toc_body_lines(body_lines: list[str], spec: ItemSpec) -> list[str]:
    toc_index: int | None = None
    for index, line in enumerate(body_lines[:60]):
        if normalized_text(line).lower() == "table of contents":
            toc_index = index
            break
    if toc_index is None:
        return body_lines

    prefix_lines = [line for line in body_lines[:toc_index] if line.strip()]
    if not prefix_lines:
        return body_lines

    target_item = normalize_item_code(spec.item_code)
    target_part = normalize_part_name(spec.part_name)
    cross_item_markers = 0
    page_number_like_lines = 0
    for line in prefix_lines[:20]:
        part_name = parse_part_name_from_leading_text(line)
        item_code = parse_item_code_from_leading_text(line)
        if part_name and target_part is not None and part_name != target_part:
            cross_item_markers += 1
        elif item_code and item_code != target_item:
            cross_item_markers += 1
        if ends_with_page_number(line):
            page_number_like_lines += 1

    # Some clean-text filings keep the Item 1 heading from the TOC, then place
    # the actual financial statements only after a TOC block. In that shape, we
    # keep the heading but trim the TOC body prefix before building section text.
    if cross_item_markers < 2 and page_number_like_lines < 3:
        return body_lines

    trimmed_index = toc_index + 1
    while trimmed_index < len(body_lines) and not body_lines[trimmed_index].strip():
        trimmed_index += 1
    return body_lines[trimmed_index:]


def build_section_text(lines: list[str], start_index: int, boundary_index: int | None, spec: ItemSpec) -> tuple[str, str, int, int, int]:
    heading_end = locate_heading_end(lines, start_index, spec)
    body_start = heading_end + 1
    body_end = len(lines) if boundary_index is None else boundary_index
    raw_body_lines = trim_leading_toc_body_lines(lines[body_start:body_end], spec)
    body_lines = normalize_body_lines(raw_body_lines)
    body_text = "\n".join(body_lines).strip()
    body_nonempty_line_count = sum(1 for line in body_lines if line.strip())
    heading = standardized_heading(spec)
    section_text = heading if not body_text else f"{heading}\n{body_text}"
    return section_text, body_text, body_start + 1, body_end, body_nonempty_line_count


def determine_section_status(expected_items: list[str], extracted_items: list[str]) -> str:
    expected = set(expected_items)
    actual = set(extracted_items)
    if not actual:
        return "failed"
    if actual >= expected:
        return "success"
    return "partial"


def build_section_payload(
    *,
    spec: ItemSpec,
    start_index: int,
    section_text: str,
    body_text: str,
    body_line_start: int,
    body_line_end: int,
    body_nonempty_line_count: int,
) -> dict[str, Any]:
    char_count = len(section_text)
    body_char_count = len(body_text)
    return {
        "item_key": spec.item_key,
        "part_name": spec.part_name,
        "item_code": spec.item_code,
        "canonical_heading": spec.canonical_heading,
        "is_primary_target": spec.is_primary_target,
        "section_text": section_text,
        "char_count": char_count,
        "body_char_count": body_char_count,
        "body_nonempty_line_count": body_nonempty_line_count,
        "line_start": start_index + 1,
        "line_end": body_line_end,
        "body_line_start": body_line_start,
        "resolution_method": "clean_text_heading_scan_v1",
        "preview": preview_text(section_text, 240),
        "body_preview": preview_text(body_text, 240),
    }


def build_section_detail(
    *,
    spec: ItemSpec,
    matched: bool,
    start_meta: dict[str, Any],
    boundary_meta: dict[str, Any] | None,
    section_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detail = {
        "matched": matched,
        "part_name": spec.part_name,
        "item_code": spec.item_code,
        "start_meta": start_meta,
        "boundary_meta": boundary_meta,
    }
    if section_payload is None:
        return detail
    detail.update(
        {
            "char_count": int(section_payload["char_count"]),
            "body_char_count": int(section_payload["body_char_count"]),
            "line_start": int(section_payload["line_start"]),
            "line_end": int(section_payload["line_end"]),
        }
    )
    return detail


def extract_items_from_filing(*, filing: FilingRow, item_specs: list[ItemSpec]) -> dict[str, Any]:
    lines = filing.filing_text.splitlines()
    sections: list[dict[str, Any]] = []
    section_details: dict[str, Any] = {}

    for spec in item_specs:
        start_index, start_meta = find_best_item_start(lines, spec)
        if start_index is None:
            section_details[spec.item_key] = build_section_detail(
                spec=spec,
                matched=False,
                start_meta=start_meta,
                boundary_meta=None,
            )
            continue

        boundary_index, boundary_meta = find_next_boundary(lines, start_index, spec)
        section_text, body_text, body_line_start, body_line_end, body_nonempty_line_count = build_section_text(
            lines,
            start_index,
            boundary_index,
            spec,
        )
        section_payload = build_section_payload(
            spec=spec,
            start_index=start_index,
            section_text=section_text,
            body_text=body_text,
            body_line_start=body_line_start,
            body_line_end=body_line_end,
            body_nonempty_line_count=body_nonempty_line_count,
        )
        sections.append(section_payload)
        section_details[spec.item_key] = build_section_detail(
            spec=spec,
            matched=True,
            start_meta=start_meta,
            boundary_meta=boundary_meta,
            section_payload=section_payload,
        )

    sections.sort(key=lambda item: int(item["line_start"]))
    for order, section in enumerate(sections, start=1):
        section["section_order"] = order

    extracted_items = [section["item_key"] for section in sections]
    requested_items = [spec.item_key for spec in item_specs]
    missing_items = [item for item in requested_items if item not in set(extracted_items)]
    primary_targets = [spec.item_key for spec in item_specs if spec.is_primary_target]
    primary_targets_found = [item for item in extracted_items if item in set(primary_targets)]
    section_status = determine_section_status(requested_items, extracted_items)

    return {
        "section_status": section_status,
        "requested_items": requested_items,
        "extracted_items": extracted_items,
        "missing_items": missing_items,
        "primary_targets": primary_targets,
        "primary_targets_found": primary_targets_found,
        "sections": sections,
        "full_line_count": len(lines),
        "debug": {
            "section_details": section_details,
        },
    }


# Audit pipeline: apply a lightweight item-specific check after extraction so we
# can reject TOC matches, empty bodies, suspiciously short sections, and overlaps.
def section_body_text(section: dict[str, Any]) -> str:
    text = str(section.get("section_text") or "")
    if "\n" not in text:
        return ""
    return text.split("\n", 1)[1].strip()


def section_overlap_line_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    overlap_start = max(int(left["line_start"]), int(right["line_start"]))
    overlap_end = min(int(left["line_end"]), int(right["line_end"]))
    return max(0, overlap_end - overlap_start + 1)


def detect_section_overlaps(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A finalized filing should not assign the same body region to multiple items.
    ordered = sorted(
        sections,
        key=lambda item: (
            int(item["line_start"]),
            int(item["line_end"]),
            str(item["item_key"]),
        ),
    )
    overlaps: list[SectionOverlap] = []
    for left_index, left in enumerate(ordered):
        left_end = int(left["line_end"])
        for right in ordered[left_index + 1 :]:
            right_start = int(right["line_start"])
            if right_start > left_end:
                break
            overlap_lines = section_overlap_line_count(left, right)
            if overlap_lines <= 0:
                continue
            severity = "warn" if overlap_lines <= 2 else "fail"
            overlaps.append(
                SectionOverlap(
                    left_item_key=left["item_key"],
                    right_item_key=right["item_key"],
                    left_line_start=int(left["line_start"]),
                    left_line_end=left_end,
                    right_line_start=right_start,
                    right_line_end=int(right["line_end"]),
                    overlap_line_count=overlap_lines,
                    severity=severity,
                )
            )
    return [overlap.to_dict() for overlap in overlaps]


def looks_like_reference_stub(text: str) -> bool:
    lowered = normalized_text(text).lower()
    if lowered.startswith(
        (
            "see ",
            "refer to ",
            "included in ",
            "described in ",
            "for a description",
        )
    ):
        return True
    if len(lowered) <= 400 and (
        "incorporated by reference" in lowered
        or "included immediately following" in lowered
    ):
        return True
    return False


def looks_like_short_stub_value(text: str) -> bool:
    lowered = normalized_text(text).lower().strip(" .")
    return lowered in {
        "none",
        "not applicable",
        "not required",
        "no material changes",
        "no",
    }


def meaningful_body_lines(body_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in body_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or is_decorative_line(stripped):
            continue
        lines.append(stripped)
    return lines


def body_open_lines(body_text: str, max_lines: int = 5) -> list[str]:
    lines = meaningful_body_lines(body_text)
    return lines[:max_lines]


def body_open_prose_count(body_text: str, max_lines: int = 5) -> int:
    return sum(1 for line in body_open_lines(body_text, max_lines=max_lines) if looks_like_prose_line(line))


def looks_like_short_stub_cluster(body_text: str) -> bool:
    # Some valid short disclosures are multi-line stubs, such as "None." plus a
    # short sub-label, so we recognize that cluster shape explicitly.
    lines = meaningful_body_lines(body_text)
    if not lines or len(lines) > 4:
        return False
    if any(looks_like_prose_line(line) for line in lines):
        return False
    stub_count = sum(1 for line in lines if looks_like_short_stub_value(line))
    if stub_count == 0:
        return False
    non_stub_lines = [line for line in lines if not looks_like_short_stub_value(line)]
    return all(len(normalized_text(line)) <= 80 for line in non_stub_lines)


def is_likely_early_toc_match(
    *,
    section: dict[str, Any],
    spec: ItemSpec,
    total_line_count: int,
) -> bool:
    line_start = int(section.get("line_start") or 0)
    body_char_count = int(section.get("body_char_count") or 0)
    body_nonempty_line_count = int(section.get("body_nonempty_line_count") or 0)
    if line_start <= 0:
        return False
    early_absolute = line_start <= 220
    early_relative = total_line_count > 0 and line_start <= max(80, int(total_line_count * 0.12))
    if not (early_absolute or early_relative):
        return False
    if body_char_count == 0:
        return True
    if spec.body_profile == BODY_PROFILE_TABLE_HEAVY:
        return body_nonempty_line_count <= 3 and body_char_count < max(spec.min_chars // 2, 220)
    if spec.body_profile == BODY_PROFILE_SHORT_DISCLOSURE:
        return body_nonempty_line_count <= 1 and body_char_count < 40
    return body_nonempty_line_count <= 3 and body_char_count < max(spec.min_chars // 2, 160)


def audit_narrative_item(
    *,
    spec: ItemSpec,
    section: dict[str, Any],
    total_line_count: int,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    status = "pass"
    body_text = section_body_text(section)
    body_char_count = int(section.get("body_char_count") or 0)
    if body_char_count == 0:
        flags.append("heading_only")
        return "fail", flags
    if is_likely_early_toc_match(section=section, spec=spec, total_line_count=total_line_count):
        flags.append("early_toc_match")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= max(180, int(total_line_count * 0.10)):
        flags.append("too_early_for_item")
        status = promote_status(status, "warn")
    if spec.prefer_prose_start and body_open_prose_count(body_text, max_lines=4) == 0:
        flags.append("weak_prose_open")
        status = promote_status(status, "warn")
    if body_char_count < spec.min_chars:
        flags.append("too_short")
        status = promote_status(status, "warn")
    return status, flags


def audit_table_heavy_item(
    *,
    spec: ItemSpec,
    section: dict[str, Any],
    total_line_count: int,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    status = "pass"
    body_text = section_body_text(section)
    body_char_count = int(section.get("body_char_count") or 0)
    body_nonempty_line_count = int(section.get("body_nonempty_line_count") or 0)
    if body_char_count == 0 or body_nonempty_line_count == 0:
        flags.append("heading_only")
        return "fail", flags
    cross_item_markers = leading_cross_item_markers(body_text, spec)
    if len(cross_item_markers) >= 2:
        flags.append("cross_item_toc_head")
        return "fail", flags
    if is_likely_early_toc_match(section=section, spec=spec, total_line_count=total_line_count):
        flags.append("early_toc_match")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= max(180, int(total_line_count * 0.10)):
        flags.append("too_early_for_item")
        status = promote_status(status, "warn")
    if spec.allow_reference_only and looks_like_reference_stub(body_text):
        flags.append("allowed_reference_only")
        return status, flags
    if "table of contents" in body_text.lower():
        flags.append("toc_text_leak")
        status = promote_status(status, "fail")
    if body_char_count < spec.min_chars:
        flags.append("too_short")
        status = promote_status(status, "warn")
    return status, flags


def audit_short_disclosure_item(
    *,
    spec: ItemSpec,
    section: dict[str, Any],
    total_line_count: int,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    status = "pass"
    body_text = section_body_text(section)
    body_char_count = int(section.get("body_char_count") or 0)
    if body_char_count == 0:
        flags.append("heading_only")
        return "fail", flags
    if is_likely_early_toc_match(section=section, spec=spec, total_line_count=total_line_count):
        flags.append("early_toc_match")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= max(180, int(total_line_count * 0.10)):
        flags.append("too_early_for_item")
        status = promote_status(status, "warn")
    if looks_like_short_stub_value(body_text) or looks_like_short_stub_cluster(body_text):
        flags.append("allowed_short_stub")
        return status, flags
    if spec.allow_reference_only and looks_like_reference_stub(body_text):
        flags.append("allowed_reference_only")
        return status, flags
    if body_char_count < spec.min_chars:
        flags.append("too_short")
        status = promote_status(status, "warn")
    return status, flags


def build_overlap_maps(
    overlap_pairs: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    overlap_status_by_item: dict[str, str] = {}
    overlap_flags_by_item: dict[str, list[str]] = {}
    for overlap in overlap_pairs:
        overlap_flag = "section_overlap" if overlap["severity"] == "fail" else "boundary_overlap"
        for item_key in (overlap["left_item_key"], overlap["right_item_key"]):
            overlap_status_by_item[item_key] = promote_status(
                overlap_status_by_item.get(item_key, "pass"),
                overlap["severity"],
            )
            overlap_flags_by_item.setdefault(item_key, [])
            if overlap_flag not in overlap_flags_by_item[item_key]:
                overlap_flags_by_item[item_key].append(overlap_flag)
    return overlap_status_by_item, overlap_flags_by_item


def audit_section_by_profile(
    *,
    spec: ItemSpec,
    section: dict[str, Any],
    total_line_count: int,
) -> tuple[str, list[str]]:
    if spec.body_profile == BODY_PROFILE_NARRATIVE:
        return audit_narrative_item(
            spec=spec,
            section=section,
            total_line_count=total_line_count,
        )
    if spec.body_profile == BODY_PROFILE_TABLE_HEAVY:
        return audit_table_heavy_item(
            spec=spec,
            section=section,
            total_line_count=total_line_count,
        )
    return audit_short_disclosure_item(
        spec=spec,
        section=section,
        total_line_count=total_line_count,
    )


def build_item_audit_record(
    *,
    spec: ItemSpec,
    section: dict[str, Any] | None,
    total_line_count: int,
    overlap_status_by_item: dict[str, str],
    overlap_flags_by_item: dict[str, list[str]],
) -> ItemAuditRecord:
    flags: list[str] = []
    status = "pass"
    if section is None:
        flags.append("missing_item")
        status = "fail" if spec.is_primary_target else "warn"
    else:
        text = str(section["section_text"])
        head = "\n".join(text.splitlines()[:5]).lower()
        if "table of contents" in head:
            flags.append("toc_like_start")
            status = "fail" if spec.is_primary_target else "warn"
        else:
            status, flags = audit_section_by_profile(
                spec=spec,
                section=section,
                total_line_count=total_line_count,
            )

        tail = "\n".join(text.splitlines()[-5:]).lower()
        if "table of contents" in tail:
            flags.append("toc_like_end")
            status = promote_status(status, "fail")

        for overlap_flag in overlap_flags_by_item.get(spec.item_key, []):
            if overlap_flag not in flags:
                flags.append(overlap_flag)
        status = promote_status(status, overlap_status_by_item.get(spec.item_key, "pass"))

    return ItemAuditRecord(
        item_key=spec.item_key,
        body_profile=spec.body_profile,
        status=status,
        flags=tuple(flags),
        char_count=0 if section is None else int(section["char_count"]),
        body_char_count=0 if section is None else int(section.get("body_char_count") or 0),
    )


# Output writers: keep machine-readable JSON for debugging, plus direct txt
# artifacts for the extracted item bodies.
def section_text_file_name(item_key: str) -> str:
    return f"{item_key}.txt"


def write_section_text_files(artifact_dir: Path, sections: list[dict[str, Any]]) -> dict[str, str]:
    section_dir = artifact_dir / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    section_files: dict[str, str] = {}
    for section in sections:
        item_key = str(section["item_key"])
        file_name = section_text_file_name(item_key)
        file_path = section_dir / file_name
        file_path.write_text(str(section["section_text"]).rstrip("\n") + "\n", encoding="utf-8")
        section_files[item_key] = str(file_path.relative_to(artifact_dir))
    return section_files


def audit_extraction_result(*, filing: FilingRow, extraction_result: dict[str, Any], item_specs: list[ItemSpec]) -> dict[str, Any]:
    spec_by_key = {spec.item_key: spec for spec in item_specs}
    section_by_key = {section["item_key"]: section for section in extraction_result["sections"]}
    overlap_pairs = detect_section_overlaps(extraction_result["sections"])
    overlap_status_by_item, overlap_flags_by_item = build_overlap_maps(overlap_pairs)
    item_results: list[ItemAuditRecord] = []
    overall_flags: list[str] = []
    total_line_count = int(extraction_result.get("full_line_count") or 0)

    for spec in item_specs:
        item_results.append(
            build_item_audit_record(
                spec=spec,
                section=section_by_key.get(spec.item_key),
                total_line_count=total_line_count,
                overlap_status_by_item=overlap_status_by_item,
                overlap_flags_by_item=overlap_flags_by_item,
            )
        )

    if extraction_result["section_status"] == "failed":
        overall_flags.append("no_items_extracted")
    for missing_item in extraction_result["missing_items"]:
        if spec_by_key[missing_item].is_primary_target:
            overall_flags.append(f"missing_primary:{missing_item}")

    audit_status = "pass"
    if any(item.status == "fail" for item in item_results) or any(flag.startswith("missing_primary:") for flag in overall_flags):
        audit_status = "fail"
    elif any(item.status == "warn" for item in item_results) or overall_flags:
        audit_status = "warn"

    return {
        "audit_status": audit_status,
        "overall_flags": overall_flags,
        "item_results": [item.to_dict() for item in item_results],
        "overlap_pairs": overlap_pairs,
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    forms: list[str],
    requested_items: list[str],
    input_files: list[tuple[str, int, Path]],
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "input_root": str(Path(args.input_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "tickers": [normalize_ticker(value) for value in (args.tickers or [])],
        "year_from": int(args.year_from),
        "year_to": int(args.year_to),
        "forms": forms,
        "requested_items": requested_items,
        "limit": int(args.limit),
        "save_full_filing_text": bool(args.save_full_filing_text),
        "dry_run": bool(args.dry_run),
        "input_files": [
            {
                "form": form,
                "year": year,
                "path": str(path),
            }
            for form, year, path in input_files
        ],
        "created_at_utc": utc_now_iso(),
    }


def artifact_dir_for_row(run_dir: Path, filing: FilingRow) -> Path:
    accession = filing.accession or f"row_{filing.source_row_idx}"
    return (
        run_dir
        / "artifacts"
        / f"year={filing.year}"
        / f"form={filing.form}"
        / f"ticker={filing.symbol}"
        / f"accession={accession}"
    )


def build_filing_artifact_payload(filing: FilingRow, *, save_full_filing_text: bool) -> dict[str, Any]:
    filing_payload = asdict(filing)
    if save_full_filing_text:
        return filing_payload
    filing_text = filing_payload.pop("filing_text", "")
    filing_payload["filing_text_char_count"] = len(filing_text)
    return filing_payload


def write_row_artifacts(
    run_dir: Path,
    filing: FilingRow,
    extraction_result: dict[str, Any],
    audit_result: dict[str, Any],
    *,
    save_full_filing_text: bool,
) -> str:
    artifact_dir = artifact_dir_for_row(run_dir, filing)
    # Keep the JSON artifact for structured metadata/debugging, but also write
    # one txt file per item so the extracted sections are directly readable.
    section_files = write_section_text_files(artifact_dir, extraction_result["sections"])
    extraction_payload = {
        "filing": build_filing_artifact_payload(filing, save_full_filing_text=save_full_filing_text),
        "raw_filing_text_saved": bool(save_full_filing_text),
        "section_status": extraction_result["section_status"],
        "requested_items": extraction_result["requested_items"],
        "extracted_items": extraction_result["extracted_items"],
        "missing_items": extraction_result["missing_items"],
        "primary_targets_found": extraction_result["primary_targets_found"],
        "section_files": section_files,
        "sections": extraction_result["sections"],
        "debug": extraction_result["debug"],
    }
    safe_json_dump(artifact_dir / "extraction.json", extraction_payload)
    safe_json_dump(artifact_dir / "audit.json", audit_result)
    return str(artifact_dir)


def build_result_row(
    filing: FilingRow,
    extraction_result: dict[str, Any],
    audit_result: dict[str, Any],
    artifact_dir: str,
) -> dict[str, Any]:
    return {
        "date": filing.date,
        "symbol": filing.symbol,
        "cik": filing.cik,
        "accession": filing.accession,
        "form": filing.form,
        "year": filing.year,
        "filing_idx": filing.filing_idx,
        "requested_items": extraction_result["requested_items"],
        "extracted_items": extraction_result["extracted_items"],
        "missing_items": extraction_result["missing_items"],
        "section_status": extraction_result["section_status"],
        "audit_status": audit_result["audit_status"],
        "primary_targets_found": extraction_result["primary_targets_found"],
        "output_artifact_dir": artifact_dir,
    }


# End-to-end run orchestration.
def main() -> int:
    validate_item_specs()
    args = parse_args()
    forms = [normalize_form(value) for value in args.forms]
    requested_items = resolve_requested_items(forms, args.items)
    item_specs = item_specs_for_request(forms, requested_items)

    run_id = args.run_id or make_run_id()
    object.__setattr__(args, "run_id", run_id)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / run_id
    input_files = input_files_for_forms(input_root, args.year_from, args.year_to, forms)

    manifest = build_manifest(
        args=args,
        forms=forms,
        requested_items=requested_items,
        input_files=input_files,
    )
    manifest["item_registry"] = [asdict(spec) for spec in item_specs]
    safe_json_dump(run_dir / "manifest.json", manifest)

    if args.dry_run:
        print(f"[dry-run] manifest written to {run_dir / 'manifest.json'}")
        print(f"[dry-run] input files discovered: {len(input_files)}")
        print(f"[dry-run] requested items: {', '.join(requested_items)}")
        return 0

    tickers = {normalize_ticker(value) for value in (args.tickers or [])} or None
    results_path = run_dir / "results.jsonl"
    failures_path = run_dir / "failures.jsonl"
    log_path = run_dir / "logs" / "run.log"

    stats = Counter()
    item_extract_counts = Counter()
    item_missing_counts = Counter()

    for filing in iter_filing_rows(
        input_root=input_root,
        year_from=args.year_from,
        year_to=args.year_to,
        forms=forms,
        tickers=tickers,
        limit=args.limit,
    ):
        specs_for_filing = [spec for spec in item_specs if spec.form == filing.form]
        extraction_result = extract_items_from_filing(filing=filing, item_specs=specs_for_filing)
        audit_result = audit_extraction_result(
            filing=filing,
            extraction_result=extraction_result,
            item_specs=specs_for_filing,
        )
        artifact_dir = write_row_artifacts(
            run_dir,
            filing,
            extraction_result,
            audit_result,
            save_full_filing_text=bool(args.save_full_filing_text),
        )
        result_row = build_result_row(filing, extraction_result, audit_result, artifact_dir)
        append_jsonl(results_path, result_row)

        stats["input_rows"] += 1
        stats[f"section_status::{extraction_result['section_status']}"] += 1
        stats[f"audit_status::{audit_result['audit_status']}"] += 1
        for item_key in extraction_result["extracted_items"]:
            item_extract_counts[item_key] += 1
        for item_key in extraction_result["missing_items"]:
            item_missing_counts[item_key] += 1

        if extraction_result["section_status"] != "success" or audit_result["audit_status"] != "pass":
            failure_row = dict(result_row)
            failure_flags = list(audit_result["overall_flags"])
            if not failure_flags:
                for item_result in audit_result["item_results"]:
                    if item_result["status"] == "pass":
                        continue
                    for flag in item_result["flags"]:
                        failure_flags.append(f"{item_result['item_key']}::{flag}")
            failure_row["failure_reason"] = "|".join(failure_flags) or extraction_result["section_status"]
            failure_row["failure_details"] = {
                "missing_items": extraction_result["missing_items"],
                "audit_overall_flags": audit_result["overall_flags"],
                "audit_item_results": audit_result["item_results"],
            }
            append_jsonl(failures_path, failure_row)

        append_log_line(
            log_path,
            (
                f"{utc_now_iso()} symbol={filing.symbol} form={filing.form} accession={filing.accession} "
                f"section_status={extraction_result['section_status']} audit_status={audit_result['audit_status']} "
                f"extracted={','.join(extraction_result['extracted_items']) or '<none>'}"
            ),
        )

    run_summary = {
        "run_id": run_id,
        "input_rows": stats["input_rows"],
        "success_rows": stats["section_status::success"],
        "partial_rows": stats["section_status::partial"],
        "failed_rows": stats["section_status::failed"],
        "audit_pass_rows": stats["audit_status::pass"],
        "audit_warn_rows": stats["audit_status::warn"],
        "audit_fail_rows": stats["audit_status::fail"],
        "extracted_item_counts": dict(item_extract_counts),
        "missing_item_counts": dict(item_missing_counts),
        "finished_at_utc": utc_now_iso(),
    }
    safe_json_dump(run_dir / "run_summary.json", run_summary)

    print(f"[run] results written to {results_path}")
    print(f"[run] summary written to {run_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
