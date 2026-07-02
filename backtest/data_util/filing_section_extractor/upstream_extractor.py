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
    r"(?i)\bi\s*t\s*e\s*m(?:s)?\s*\.?\s*([0-9]{1,2}[A-Za-z]?)\b"
)
ITEM_PREFIX_START_RE = re.compile(
    r"(?i)^i\s*t\s*e\s*m(?:s)?\s*\.?\s*([0-9]{1,2}[A-Za-z]?)\b"
)
BARE_ITEM_PREFIX_START_RE = re.compile(r"^([0-9]{1,2}[A-Za-z]?)\.\s*$")
STACKED_ITEM_CODE_START_RE = re.compile(r"^([0-9]{1,2}[A-Za-z]?)\.(?:\s|$)")
PART_RE = re.compile(r"(?i)\bpart\b\s*([ivx]+)\b")
PART_START_RE = re.compile(r"(?i)^part\s*([ivx]+)\b")
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_LOCATOR_SUFFIX_RE = re.compile(r"\b(?:[A-Z]{1,4}\s*-\s*)?\d{1,4}\s*$")
PAGE_REFERENCE_RE = re.compile(r"\bpages?\s+\d{1,4}\b", flags=re.IGNORECASE)
CROSS_REFERENCE_INDEX_RE = re.compile(r"cross[\s-]*reference\s+index", flags=re.IGNORECASE)
CROSS_REFERENCE_INTRO_LINE_RE = re.compile(
    r"(?:\bsee(?: also)?|\brefer to|\bincluded in|\bdescribed in|\bdiscussed in|\bprovided in|\bwithin)\s*$",
    flags=re.IGNORECASE,
)
STATEMENT_BUNDLE_SIGNALS = (
    "index to consolidated financial statements",
    "consolidated balance sheets",
    "consolidated statements of operations",
    "consolidated statements of income",
    "notes to consolidated financial statements",
    "report of independent registered public accounting firm",
)
STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}
POSITION_RELATIVE_LINE_COUNT_CAP = 4000
INVISIBLE_TEXT_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")


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
    normalized = INVISIBLE_TEXT_RE.sub("", normalized)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.translate(DASH_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def collapsed_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalized_text(text).lower())


COLLAPSED_STATEMENT_BUNDLE_SIGNALS = tuple(
    collapsed_text(signal) for signal in STATEMENT_BUNDLE_SIGNALS
)


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


def parse_bare_item_code_from_leading_text(text: str | None) -> str | None:
    if not text:
        return None
    match = BARE_ITEM_PREFIX_START_RE.search(normalized_text(text))
    if not match:
        return None
    return normalize_item_code(match.group(1))


def parse_item_code_anchor_from_leading_text(text: str | None) -> str | None:
    return parse_item_code_from_leading_text(text) or parse_bare_item_code_from_leading_text(text)


def is_standalone_item_lead(text: str | None) -> bool:
    if not text:
        return False
    normalized = normalized_text(text).lower().rstrip(".:")
    return normalized in {"item", "items"}


def parse_stacked_item_code_from_index(lines: list[str], index: int) -> str | None:
    current = normalized_text(lines[index])
    if not current:
        return None
    match = STACKED_ITEM_CODE_START_RE.search(current)
    if not match:
        return None
    previous_lines = previous_nonempty_lines(lines, index, max_lines=1)
    if not previous_lines or not is_standalone_item_lead(previous_lines[-1]):
        return None
    return normalize_item_code(match.group(1))


def parse_item_code_anchor_at_index(lines: list[str], index: int) -> str | None:
    current = lines[index]
    return (
        parse_item_code_from_leading_text(current)
        or parse_bare_item_code_from_leading_text(current)
        or parse_stacked_item_code_from_index(lines, index)
    )


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


def is_item_code_anchor_lead(text: str | None) -> bool:
    return parse_item_code_anchor_from_leading_text(text) is not None


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


def ends_with_page_locator(text: str) -> bool:
    return bool(PAGE_LOCATOR_SUFFIX_RE.search(normalized_text(text)))


def looks_like_outline_page_locator_line(text: str) -> bool:
    normalized = normalized_text(text)
    if not normalized:
        return False
    if PAGE_NUMBER_RE.fullmatch(normalized):
        return True
    if not ends_with_page_locator(normalized):
        return False
    if re.search(r"\d\.\d", normalized):
        return False
    numeric_tokens = re.findall(r"\d{1,4}", normalized)
    if len(numeric_tokens) > 2:
        return False
    return not looks_like_prose_line(normalized)


def ends_with_page_number(text: str) -> bool:
    return ends_with_page_locator(text)


def looks_like_page_column_header(text: str) -> bool:
    lowered = normalized_text(text).lower().rstrip(".:")
    return lowered in {"page", "page no", "page no."}


def find_title_fragment_support(
    lines: list[str],
    index: int,
    spec: ItemSpec,
    *,
    direction: str,
    max_nonempty: int = 4,
) -> tuple[int, str | None]:
    step = -1 if direction == "before" else 1
    seen_nonempty = 0
    cursor = index + step

    while 0 <= cursor < len(lines) and seen_nonempty < max_nonempty:
        stripped = lines[cursor].strip()
        cursor += step
        if not stripped:
            continue
        seen_nonempty += 1

        normalized = normalized_text(stripped)
        if not normalized:
            continue
        if ends_with_page_locator(normalized) or looks_like_page_column_header(normalized):
            continue
        if parse_part_name_from_leading_text(normalized) is not None:
            break
        if is_item_code_anchor_lead(normalized):
            break
        if len(normalized) > 160:
            break

        fragment_hits = heading_fragment_score(normalized, spec)
        if fragment_hits > 0:
            return fragment_hits, normalized
        if looks_like_prose_line(normalized):
            break

    return 0, None


def starts_with_lowercase_alpha(text: str) -> bool:
    normalized = normalized_text(text).lstrip("([{\"'`")
    for char in normalized:
        if char.isalpha():
            return char.islower()
    return False


def is_decorative_line(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    return not any(char.isalnum() for char in stripped)


def looks_like_toc_banner_line(text: str) -> bool:
    normalized = normalized_text(text)
    lowered = normalized.lower()
    if "table of contents" not in lowered:
        return False
    if re.search(r"\btable of contents\b.*\bitem\s+[0-9]{1,2}[a-z]?\b", lowered):
        return True
    # Keep genuine prose references intact. We only want to strip compact TOC
    # banners or page headers that leak into extracted section bodies.
    if looks_like_prose_line(normalized):
        return False
    if "|" in normalized:
        return True
    if len(normalized) <= 120:
        return True
    return ends_with_page_locator(normalized)


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


def previous_nonempty_index(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index - 1, -1, -1):
        if lines[index].strip():
            return index
    return None


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


def resolve_item_key_from_anchor(form: str, part_name: str | None, item_code: str | None) -> str | None:
    if item_code is None:
        return None

    normalized_part = normalize_part_name(part_name)
    exact_matches = [
        spec.item_key
        for spec in ITEM_SPECS[form]
        if normalize_item_code(spec.item_code) == item_code
        and normalize_part_name(spec.part_name) == normalized_part
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    if normalized_part is not None:
        return None

    loose_matches = [
        spec.item_key
        for spec in ITEM_SPECS[form]
        if normalize_item_code(spec.item_code) == item_code
    ]
    if len(loose_matches) == 1:
        return loose_matches[0]
    return None


def item_spec_by_key(form: str, item_key: str) -> ItemSpec | None:
    return next((spec for spec in ITEM_SPECS[form] if spec.item_key == item_key), None)


def resolve_supported_item_anchor(
    lines: list[str],
    index: int,
    form: str,
) -> tuple[str | None, dict[str, Any]]:
    current_text = lines[index]
    explicit_anchor = looks_like_explicit_heading_lead(current_text)
    anchor_item_code = parse_item_code_anchor_at_index(lines, index)
    if not explicit_anchor and anchor_item_code is None:
        return None, {"matched": False, "reason": "no_item_code_anchor"}

    heading_window = heading_window_text(lines, index, max_nonempty=4)
    item_code = parse_item_code_from_text(heading_window) if explicit_anchor else anchor_item_code
    if item_code is None:
        return None, {"matched": False, "reason": "no_item_code_anchor"}

    if looks_like_toc_heading_candidate(lines, index, form=form):
        return None, {"matched": False, "reason": "toc_candidate"}
    if looks_like_cross_reference_index_heading_candidate(lines, index, heading_window):
        return None, {"matched": False, "reason": "cross_reference_index"}
    if looks_like_inline_cross_reference_heading_candidate(lines, index):
        return None, {"matched": False, "reason": "inline_cross_reference"}

    part_name = parse_part_name_near_index(lines, index)
    item_key = resolve_item_key_from_anchor(form, part_name, item_code)
    if item_key is None:
        return None, {
            "matched": False,
            "reason": "unsupported_item_anchor",
            "item_code": item_code,
            "part_name": part_name,
        }

    target_spec = item_spec_by_key(form, item_key)
    base_hits = heading_fragment_score(heading_window, target_spec) if target_spec else 0
    before_hits = (
        find_title_fragment_support(lines, index, target_spec, direction="before", max_nonempty=4)[0]
        if target_spec
        else 0
    )
    after_hits = (
        find_title_fragment_support(lines, index, target_spec, direction="after", max_nonempty=4)[0]
        if target_spec
        else 0
    )
    fragment_hits = max(base_hits, before_hits, after_hits)

    if parse_bare_item_code_from_leading_text(current_text) is not None and fragment_hits == 0:
        return None, {
            "matched": False,
            "reason": "bare_item_anchor_without_title_support",
            "item_key": item_key,
        }

    return item_key, {
        "matched": True,
        "item_key": item_key,
        "item_code": item_code,
        "part_name": part_name,
        "fragment_hits": fragment_hits,
        "heading_preview": preview_text(heading_window, 200),
    }


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


def position_reference_line_count(total_line_count: int) -> int:
    if total_line_count <= 0:
        return 0
    # Sparse annual-report style text can explode raw line counts into the
    # 100k range, which makes relative "early/late" position heuristics stop
    # reflecting actual document structure. Cap the reference count so the
    # relative threshold remains a coarse ordering signal rather than a parser
    # artifact.
    return min(total_line_count, POSITION_RELATIVE_LINE_COUNT_CAP)


def early_relative_position_threshold(total_line_count: int) -> int:
    reference_line_count = position_reference_line_count(total_line_count)
    if reference_line_count <= 0:
        return 80
    return max(80, int(reference_line_count * 0.12))


def late_position_warning_threshold(total_line_count: int) -> int:
    reference_line_count = position_reference_line_count(total_line_count)
    if reference_line_count <= 0:
        return 180
    return max(180, int(reference_line_count * 0.10))


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
    early_relative = total_line_count > 0 and line_index <= early_relative_position_threshold(total_line_count)
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
    is_toc_candidate = looks_like_toc_heading_candidate(lines, index, form=spec.form)
    if looks_like_item8_front_index_statement_directory(provisional_body_text, spec):
        is_toc_candidate = True
        score_notes = [*score_notes, "statement_directory_toc"]
    if looks_like_item8_statement_page_map_proxy_candidate(
        lines=lines,
        index=index,
        spec=spec,
        provisional_body_text=provisional_body_text,
        provisional_boundary_index=provisional_boundary_index,
    ):
        is_toc_candidate = False
        score_notes = [*score_notes, "statement_page_map_proxy"]
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


def build_index_only_start_meta(candidates: list[HeadingCandidate]) -> dict[str, Any]:
    previews = [candidate.to_debug_payload() for candidate in sorted(candidates, key=lambda item: item.sort_key(), reverse=True)[:3]]
    return {
        "matched": False,
        "reason": "only_index_candidates",
        "candidate_count": len(candidates),
        "top_candidates": previews,
    }


def find_best_item_start(lines: list[str], spec: ItemSpec) -> tuple[int | None, dict[str, Any]]:
    candidates: list[HeadingCandidate] = []
    target_item = normalize_item_code(spec.item_code)
    target_part = normalize_part_name(spec.part_name)
    total_line_count = len(lines)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        explicit_anchor = looks_like_explicit_heading_lead(line)
        anchor_item_code = parse_item_code_anchor_at_index(lines, index)
        if not explicit_anchor and anchor_item_code is None:
            continue
        if (
            explicit_anchor
            and anchor_item_code is None
            and parse_part_name_from_leading_text(line) is not None
            and next_explicit_item_anchor_index(lines, index, max_nonempty=2) is not None
        ):
            continue
        window_text = heading_window_text(lines, index, max_nonempty=5, spec=spec)
        if not window_text:
            continue
        if explicit_anchor and looks_like_cross_reference_index_heading_candidate(lines, index, window_text):
            continue
        if explicit_anchor and looks_like_inline_cross_reference_heading_candidate(lines, index):
            continue

        item_code = parse_item_code_from_text(window_text) if explicit_anchor else anchor_item_code
        if item_code != target_item:
            continue

        part_name = parse_part_name_near_index(lines, index)
        if target_part is not None and part_name not in {target_part, None}:
            continue

        fragment_hits = heading_fragment_score(window_text, spec)
        before_hits, _ = find_title_fragment_support(lines, index, spec, direction="before")
        after_hits, _ = find_title_fragment_support(lines, index, spec, direction="after")
        fragment_hits = max(fragment_hits, before_hits, after_hits)
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

    # If every explicit heading candidate is index-like, we prefer to report
    # "no usable heading" instead of materializing a fake zero-body section.
    preferred = [candidate for candidate in candidates if not candidate.is_toc_candidate]
    if not preferred:
        return None, build_index_only_start_meta(candidates)
    candidates = preferred

    best = max(candidates, key=lambda item: item.sort_key())
    return int(best.line_index), {"matched": True, "best_candidate": best.to_debug_payload(), "candidate_count": len(candidates)}


def matches_boundary(lines: list[str], index: int, boundary_part: str | None, boundary_item_code: str | None) -> bool:
    current_text = normalized_text(lines[index])
    current_item_code = parse_item_code_anchor_at_index(lines, index)
    if not looks_like_explicit_heading_lead(current_text) and current_item_code is None:
        return False
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


def looks_like_toc_heading_candidate(
    lines: list[str],
    index: int,
    *,
    form: str | None = None,
    lookback: int = 30,
) -> bool:
    line = lines[index]
    if not looks_like_explicit_heading_lead(line) and parse_item_code_anchor_at_index(lines, index) is None:
        return False
    start = max(0, index - lookback)
    recent_window = lines[start:index]
    if form != "10-K":
        # Some 10-Q filings legitimately use a top index row as the only clean
        # anchor for Part I Item 1 before the statement block starts.
        if not ends_with_page_number(line):
            return False
        return any("table of contents" in normalized_text(candidate).lower() for candidate in recent_window)

    heading_window = heading_window_text(lines, index, max_nonempty=4)
    if not heading_window:
        return False
    if not looks_like_page_mapped_heading_entry(lines, index):
        return False
    if any("table of contents" in normalized_text(candidate).lower() for candidate in recent_window):
        return True
    if looks_like_split_toc_anchor_candidate(lines, index):
        return True
    return looks_like_front_index_heading_candidate(lines, index, heading_window)


def looks_like_front_index_heading_candidate(lines: list[str], index: int, heading_window: str | None = None) -> bool:
    if index > max(320, int(len(lines) * 0.12)):
        return False
    if heading_window is None:
        heading_window = heading_window_text(lines, index, max_nonempty=4)
    if not heading_window:
        return False
    if not looks_like_page_mapped_heading_entry(lines, index):
        return False

    start = max(0, index - 18)
    end = min(len(lines), index + 18)
    page_mapped_heading_count = 0
    part_heading_count = 0
    has_page_column_header = False

    for probe_index in range(start, end):
        probe = normalized_text(lines[probe_index])
        if not probe:
            continue
        if looks_like_page_column_header(probe):
            has_page_column_header = True
        if parse_part_name_from_leading_text(probe) is not None:
            part_heading_count += 1
        if looks_like_page_mapped_heading_entry(lines, probe_index):
            page_mapped_heading_count += 1

    if page_mapped_heading_count < 3:
        return False
    if has_page_column_header:
        return True
    if part_heading_count >= 2:
        return True
    return page_mapped_heading_count >= 5


def looks_like_split_toc_anchor_candidate(lines: list[str], index: int) -> bool:
    # Annual-report TOCs sometimes split a row across:
    #   7.
    #   Management's Discussion and Analysis ...
    #   43
    # and then immediately continue into the next item anchor. That shape is
    # much closer to a page-mapped TOC row than a genuine stacked heading.
    if index > max(320, int(len(lines) * 0.12)):
        return False

    current = normalized_text(lines[index])
    if not current:
        return False
    if parse_bare_item_code_from_leading_text(current) is None and parse_stacked_item_code_from_index(lines, index) is None:
        return False
    if not looks_like_page_mapped_heading_entry(lines, index):
        return False

    next_lines: list[tuple[int, str]] = []
    for cursor in range(index + 1, len(lines)):
        stripped = lines[cursor].strip()
        if not stripped:
            continue
        next_lines.append((cursor, normalized_text(stripped)))
        if len(next_lines) >= 2:
            break
    if len(next_lines) < 2:
        return False

    _, title_line = next_lines[0]
    page_index, page_line = next_lines[1]
    if not looks_like_front_index_title_line(title_line):
        return False
    if not ends_with_page_locator(page_line):
        return False

    return next_explicit_item_anchor_index(lines, page_index, max_nonempty=2) is not None


def looks_like_front_index_title_line(text: str) -> bool:
    normalized = normalized_text(text)
    if not normalized or len(normalized) > 180:
        return False
    if parse_part_name_from_leading_text(normalized) is not None:
        return False
    if parse_item_code_anchor_from_leading_text(normalized) is not None:
        return False
    if ends_with_page_locator(normalized) or looks_like_page_column_header(normalized):
        return False
    if normalized[-1:] in {".", ";", ":", "?", "!"}:
        return False
    first_alpha = next((char for char in normalized if char.isalpha()), "")
    if not first_alpha or not first_alpha.isupper():
        return False
    return True


def looks_like_page_mapped_heading_entry(lines: list[str], index: int) -> bool:
    probe = normalized_text(lines[index])
    if not probe:
        return False
    if parse_item_code_anchor_at_index(lines, index) is None:
        return False
    window = heading_window_text(lines, index, max_nonempty=4)
    if window and ends_with_page_locator(window):
        return True

    next_lines = next_nonempty_lines(lines, index + 1, max_lines=2)
    if len(next_lines) < 2:
        return False
    title_line, page_line = next_lines[0], next_lines[1]
    if not looks_like_front_index_title_line(title_line):
        return False
    return ends_with_page_locator(page_line)


def next_explicit_item_anchor_index(lines: list[str], index: int, max_nonempty: int = 2) -> int | None:
    seen_nonempty = 0
    for cursor in range(index + 1, len(lines)):
        stripped = lines[cursor].strip()
        if not stripped:
            continue
        seen_nonempty += 1
        if parse_item_code_anchor_at_index(lines, cursor) is not None:
            return cursor
        if parse_part_name_from_leading_text(stripped) is not None:
            return None
        if seen_nonempty >= max_nonempty:
            return None
    return None


def count_statement_bundle_signal_hits(lines: Iterable[str]) -> int:
    collapsed = " ".join(collapsed_text(line) for line in lines if line)
    return sum(1 for signal in COLLAPSED_STATEMENT_BUNDLE_SIGNALS if signal in collapsed)


def is_statement_bundle_signal_line(text: str) -> bool:
    collapsed = collapsed_text(text)
    return any(signal in collapsed for signal in COLLAPSED_STATEMENT_BUNDLE_SIGNALS)


def looks_like_item8_statement_page_map_proxy_candidate(
    *,
    lines: list[str],
    index: int,
    spec: ItemSpec,
    provisional_body_text: str,
    provisional_boundary_index: int | None,
) -> bool:
    # This is intentionally narrow: only a late 10-K Item 8 heading that is
    # immediately followed by a statement page-map and then a real statement
    # bundle should bypass the generic TOC-heading rejection.
    if spec.form != "10-K" or spec.item_key != "item_8":
        return False
    if provisional_boundary_index is None:
        return False
    if index <= max(320, int(len(lines) * 0.12)):
        return False
    if not looks_like_statement_bundle_body(provisional_body_text):
        return False

    probe_lines = next_nonempty_lines(lines, index + 1, max_lines=24)
    if len(probe_lines) < 6:
        return False
    if not any(looks_like_page_column_header(line) for line in probe_lines[:6]):
        return False

    page_locator_count = sum(1 for line in probe_lines[:20] if ends_with_page_locator(line))
    if page_locator_count < 4:
        return False

    if count_statement_bundle_signal_hits(probe_lines[:20]) < 3:
        return False
    return True


def looks_like_statement_bundle_page_map_line(text: str) -> bool:
    normalized = normalized_text(text)
    if not normalized:
        return False
    if looks_like_page_column_header(normalized):
        return True
    if len(normalized) <= 60 and normalized.endswith(":"):
        return True
    if is_decorative_line(normalized):
        return True
    return looks_like_outline_page_locator_line(normalized)


def looks_like_statement_bundle_page_map_start_line(text: str) -> bool:
    normalized = normalized_text(text)
    if not normalized:
        return False
    if looks_like_page_column_header(normalized):
        return True
    if len(normalized) <= 60 and normalized.endswith(":"):
        return True
    return looks_like_outline_page_locator_line(normalized) and is_statement_bundle_signal_line(normalized)


def headingless_item8_page_map_search_start(lines: list[str]) -> int:
    # Keep a late-document floor so front TOC blocks stay out, but do not let a
    # pure proportional gate skip compact 10-K appendices whose statement page
    # map starts shortly after line ~180.
    return max(120, min(int(len(lines) * 0.12), 180))


def find_headingless_item8_statement_page_map_proxy(
    *,
    lines: list[str],
    spec: ItemSpec,
) -> tuple[int, int] | None:
    # Some 10-Ks never repeat a canonical late Item 8 heading. Instead they
    # expose a late statement page-map, then flow directly into the statement
    # bundle. We only recognize that narrow shape here.
    if spec.form != "10-K" or spec.item_key != "item_8":
        return None

    start_threshold = headingless_item8_page_map_search_start(lines)
    for index in range(start_threshold, len(lines)):
        current_line = normalized_text(lines[index])
        if not current_line or not looks_like_statement_bundle_page_map_start_line(current_line):
            continue

        probe_lines = next_nonempty_lines(lines, index, max_lines=40)
        if len(probe_lines) < 6:
            continue
        page_locator_count = sum(1 for line in probe_lines[:32] if ends_with_page_locator(line))
        signal_count = sum(1 for line in probe_lines[:32] if is_statement_bundle_signal_line(line))
        if page_locator_count < 4 or signal_count < 3:
            continue

        cursor = index
        while cursor < len(lines):
            probe = normalized_text(lines[cursor])
            if not probe:
                cursor += 1
                continue
            if looks_like_statement_bundle_page_map_line(probe):
                cursor += 1
                continue
            break

        body_start = cursor
        while body_start < len(lines):
            probe = normalized_text(lines[body_start])
            if not probe or is_decorative_line(probe):
                body_start += 1
                continue
            break
        if body_start >= len(lines):
            return None

        body_probe = "\n".join(next_nonempty_lines(lines, body_start, max_lines=48))
        if body_probe and not looks_like_financial_statement_payload_body(body_probe):
            continue
        return index, body_start
    return None


def looks_like_page_mapping_text(text: str) -> bool:
    lowered = normalized_text(text).lower()
    return bool(PAGE_REFERENCE_RE.search(lowered) or "not applicable" in lowered)


def looks_like_cross_reference_index_heading_candidate(
    lines: list[str],
    index: int,
    heading_window: str,
) -> bool:
    if not looks_like_explicit_heading_lead(lines[index]):
        return False

    heading_text = normalized_text(heading_window)
    probe_text = " ".join(next_nonempty_lines(lines, index + 1, max_lines=4))
    if not (
        looks_like_page_mapping_text(heading_text)
        or looks_like_page_mapping_text(probe_text)
    ):
        return False

    lookback_start = max(0, index - 30)
    recent_context_lower = " ".join(
        normalized_text(line).lower() for line in lines[lookback_start:index]
    )
    if CROSS_REFERENCE_INDEX_RE.search(recent_context_lower):
        return True
    return "item number" in recent_context_lower


def looks_like_inline_cross_reference_heading_candidate(lines: list[str], index: int) -> bool:
    if not looks_like_explicit_heading_lead(lines[index]):
        return False

    next_lines = next_nonempty_lines(lines, index + 1, max_lines=2)
    if not next_lines or not starts_with_lowercase_alpha(next_lines[0]):
        return False

    for previous_line in reversed(previous_nonempty_lines(lines, index, max_lines=2)):
        if CROSS_REFERENCE_INTRO_LINE_RE.search(normalized_text(previous_line).lower()):
            return True
    return False


def find_next_boundary(lines: list[str], start_index: int, spec: ItemSpec) -> tuple[int | None, dict[str, Any]]:
    boundary_keys = spec.boundary_candidates
    boundary_specs = [item_key_to_part_and_code(item_key) for item_key in boundary_keys]

    for index in range(start_index + 1, len(lines)):
        if not lines[index].strip():
            continue
        for boundary_key, (boundary_part, boundary_item_code) in zip(boundary_keys, boundary_specs):
            if matches_boundary(lines, index, boundary_part, boundary_item_code):
                if looks_like_toc_heading_candidate(lines, index, form=spec.form):
                    continue
                slice_boundary_index = index
                previous_index = previous_nonempty_index(lines, index)
                if (
                    parse_stacked_item_code_from_index(lines, index) is not None
                    and previous_index is not None
                    and is_standalone_item_lead(lines[previous_index])
                ):
                    slice_boundary_index = previous_index
                return slice_boundary_index, {
                    "matched": True,
                    "boundary_item_key": boundary_key,
                    "boundary_line_number": index + 1,
                    "boundary_preview": preview_text(heading_window_text(lines, index, max_nonempty=4), 200),
                    "boundary_mode": "canonical_boundary",
                }
        # Keep this fallback scoped to 10-K for now. The second-class family we
        # are targeting is the reordered 10-K body shape (for example EIX),
        # while 10-Q Item 1 still has legitimate top-index proxy starts (for
        # example NFLX) that become over-expanded if we let any later supported
        # anchor cut the section.
        if spec.form == "10-K":
            supported_item_key, supported_anchor_meta = resolve_supported_item_anchor(lines, index, spec.form)
            if supported_item_key and supported_item_key != spec.item_key:
                return index, {
                    "matched": True,
                    "boundary_item_key": supported_item_key,
                    "boundary_line_number": index + 1,
                    "boundary_preview": supported_anchor_meta["heading_preview"],
                    "boundary_mode": "supported_item_anchor",
                    "boundary_fragment_hits": supported_anchor_meta["fragment_hits"],
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
        if looks_like_toc_banner_line(normalized):
            continue
        cleaned.append(normalized)

    while cleaned and not cleaned[0].strip():
        cleaned = cleaned[1:]
    while cleaned and not cleaned[-1].strip():
        cleaned = cleaned[:-1]
    return cleaned


def trim_leading_toc_body_lines(body_lines: list[str], spec: ItemSpec) -> tuple[list[str], int]:
    toc_index: int | None = None
    for index, line in enumerate(body_lines[:60]):
        # Keep leading-trim intentionally conservative. Compact TOC banners can
        # appear inside valid table-heavy bodies, so the broader banner matcher
        # is only safe for mid-body cleanup in normalize_body_lines().
        if normalized_text(line).lower() == "table of contents":
            toc_index = index
            break
    if toc_index is None:
        return body_lines, 0

    prefix_lines = [line for line in body_lines[:toc_index] if line.strip()]
    if not prefix_lines:
        return body_lines, 0

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
        return body_lines, 0

    trimmed_index = toc_index + 1
    while trimmed_index < len(body_lines) and not body_lines[trimmed_index].strip():
        trimmed_index += 1
    return body_lines[trimmed_index:], trimmed_index


def materialize_body_from_line_range(
    lines: list[str],
    body_start: int,
    body_end: int,
    spec: ItemSpec,
    *,
    trim_leading_toc: bool,
) -> tuple[str, int, int, int]:
    raw_body_lines = lines[body_start:body_end]
    leading_trim_offset = 0
    if trim_leading_toc:
        raw_body_lines, leading_trim_offset = trim_leading_toc_body_lines(raw_body_lines, spec)
    body_lines = normalize_body_lines(raw_body_lines)
    body_text = "\n".join(body_lines).strip()
    body_nonempty_line_count = sum(1 for line in body_lines if line.strip())
    return body_text, body_start + leading_trim_offset + 1, body_end, body_nonempty_line_count


def build_section_text(lines: list[str], start_index: int, boundary_index: int | None, spec: ItemSpec) -> tuple[str, str, int, int, int]:
    heading_end = locate_heading_end(lines, start_index, spec)
    body_start = heading_end + 1
    body_end = len(lines) if boundary_index is None else boundary_index
    body_text, body_line_start, body_line_end, body_nonempty_line_count = materialize_body_from_line_range(
        lines,
        body_start,
        body_end,
        spec,
        trim_leading_toc=True,
    )
    heading = standardized_heading(spec)
    section_text = heading if not body_text else f"{heading}\n{body_text}"
    return section_text, body_text, body_line_start, body_line_end, body_nonempty_line_count


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
    resolution_method: str = "clean_text_heading_scan_v1",
    resolution_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    char_count = len(section_text)
    body_char_count = len(body_text)
    payload = {
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
        "resolution_method": resolution_method,
        "preview": preview_text(section_text, 240),
        "body_preview": preview_text(body_text, 240),
    }
    if resolution_debug:
        payload["resolution_debug"] = resolution_debug
    return payload


def build_section_detail(
    *,
    spec: ItemSpec,
    matched: bool,
    start_meta: dict[str, Any],
    boundary_meta: dict[str, Any] | None,
    section_payload: dict[str, Any] | None = None,
    effective_start_meta: dict[str, Any] | None = None,
    effective_boundary_meta: dict[str, Any] | None = None,
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
    if effective_start_meta is not None:
        detail["effective_start_meta"] = effective_start_meta
    if effective_boundary_meta is not None:
        detail["effective_boundary_meta"] = effective_boundary_meta
    return detail


def build_effective_fallback_detail_metas(
    *,
    lines: list[str],
    resolution_method: str,
    resolution_debug: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not resolution_debug:
        return None, None

    if resolution_method == "nontraditional_10q_main_body_fallback_v1":
        fallback_start_line = int(resolution_debug["fallback_line_start"])
        fallback_start_index = fallback_start_line - 1
        effective_start_meta = {
            "matched": True,
            "reason": "fallback_structural_heading",
            "line_number": fallback_start_line,
            "heading_preview": preview_text(
                heading_window_text(lines, fallback_start_index, max_nonempty=4),
                240,
            ),
            "fallback_heading_label": str(resolution_debug["fallback_heading_label"]),
        }
        fallback_boundary_line = resolution_debug.get("fallback_boundary_heading_line")
        if fallback_boundary_line is None:
            return effective_start_meta, None
        fallback_boundary_index = int(fallback_boundary_line) - 1
        effective_boundary_meta = {
            "matched": True,
            "reason": "fallback_structural_boundary",
            "boundary_line_number": int(fallback_boundary_line),
            "boundary_preview": preview_text(
                heading_window_text(lines, fallback_boundary_index, max_nonempty=4),
                200,
            ),
            "fallback_boundary_label": str(resolution_debug["fallback_boundary_label"]),
        }
        return effective_start_meta, effective_boundary_meta

    if resolution_method == "item8_headingless_statement_page_map_proxy_v1":
        fallback_start_line = int(resolution_debug["fallback_line_start"])
        fallback_start_index = fallback_start_line - 1
        effective_start_meta = {
            "matched": True,
            "reason": "fallback_proxy_body_start",
            "proxy_page_map_start_line": int(resolution_debug["proxy_page_map_start_line"]),
            "line_number": fallback_start_line,
            "heading_preview": preview_text(
                heading_window_text(lines, fallback_start_index, max_nonempty=4),
                240,
            ),
        }

        fallback_boundary_line = resolution_debug.get("fallback_boundary_heading_line")
        if fallback_boundary_line is not None:
            fallback_boundary_index = int(fallback_boundary_line) - 1
            effective_boundary_meta = {
                "matched": True,
                "reason": "fallback_proxy_boundary",
                "boundary_item_key": "item_9",
                "boundary_line_number": int(fallback_boundary_line),
                "boundary_preview": preview_text(
                    heading_window_text(lines, fallback_boundary_index, max_nonempty=4),
                    200,
                ),
            }
            return effective_start_meta, effective_boundary_meta

        signature_boundary_line = resolution_debug.get("fallback_signature_boundary_line")
        if signature_boundary_line is not None:
            signature_boundary_index = int(signature_boundary_line) - 1
            effective_boundary_meta = {
                "matched": True,
                "reason": "fallback_signature_cluster_boundary",
                "boundary_line_number": int(signature_boundary_line),
                "boundary_preview": preview_text(
                    heading_window_text(lines, signature_boundary_index, max_nonempty=4),
                    200,
                ),
            }
            return effective_start_meta, effective_boundary_meta
        return effective_start_meta, None

    if resolution_method != "item8_part_iv_item15_fallback_v1":
        return None, None

    fallback_start_line = int(resolution_debug["fallback_line_start"])
    fallback_boundary_line = int(resolution_debug["fallback_boundary_heading_line"])
    fallback_start_index = fallback_start_line - 1
    fallback_boundary_index = fallback_boundary_line - 1

    effective_start_meta = {
        "matched": True,
        "reason": "fallback_proxy_heading",
        "fallback_item_key": str(resolution_debug["fallback_item_key"]),
        "line_number": fallback_start_line,
        "heading_preview": preview_text(
            heading_window_text(lines, fallback_start_index, max_nonempty=4),
            240,
        ),
    }
    effective_boundary_meta = {
        "matched": True,
        "reason": "fallback_proxy_boundary",
        "boundary_item_key": str(resolution_debug["fallback_boundary_item_key"]),
        "boundary_line_number": fallback_boundary_line,
        "boundary_preview": preview_text(
            heading_window_text(lines, fallback_boundary_index, max_nonempty=4),
            200,
        ),
    }
    return effective_start_meta, effective_boundary_meta


def extract_items_from_filing(*, filing: FilingRow, item_specs: list[ItemSpec]) -> dict[str, Any]:
    lines = filing.filing_text.splitlines()
    sections: list[dict[str, Any]] = []
    section_details: dict[str, Any] = {}

    for spec in item_specs:
        start_index, start_meta = find_best_item_start(lines, spec)
        if start_index is None:
            fallback_payload = maybe_build_nontraditional_10q_main_body_fallback(
                lines=lines,
                spec=spec,
                start_meta=start_meta,
            )
            if fallback_payload is not None:
                (
                    payload_start_index,
                    section_text,
                    body_text,
                    body_line_start,
                    body_line_end,
                    body_nonempty_line_count,
                    resolution_method,
                    resolution_debug,
                ) = fallback_payload
                section_payload = build_section_payload(
                    spec=spec,
                    start_index=payload_start_index,
                    section_text=section_text,
                    body_text=body_text,
                    body_line_start=body_line_start,
                    body_line_end=body_line_end,
                    body_nonempty_line_count=body_nonempty_line_count,
                    resolution_method=resolution_method,
                    resolution_debug=resolution_debug,
                )
                effective_start_meta, effective_boundary_meta = build_effective_fallback_detail_metas(
                    lines=lines,
                    resolution_method=resolution_method,
                    resolution_debug=resolution_debug,
                )
                sections.append(section_payload)
                section_details[spec.item_key] = build_section_detail(
                    spec=spec,
                    matched=True,
                    start_meta=start_meta,
                    boundary_meta=None,
                    section_payload=section_payload,
                    effective_start_meta=effective_start_meta,
                    effective_boundary_meta=effective_boundary_meta,
                )
                continue
            fallback_payload = maybe_build_item8_headingless_statement_page_map_proxy(
                lines=lines,
                spec=spec,
                start_meta=start_meta,
            )
            if fallback_payload is not None:
                (
                    payload_start_index,
                    section_text,
                    body_text,
                    body_line_start,
                    body_line_end,
                    body_nonempty_line_count,
                    resolution_method,
                    resolution_debug,
                ) = fallback_payload
                section_payload = build_section_payload(
                    spec=spec,
                    start_index=payload_start_index,
                    section_text=section_text,
                    body_text=body_text,
                    body_line_start=body_line_start,
                    body_line_end=body_line_end,
                    body_nonempty_line_count=body_nonempty_line_count,
                    resolution_method=resolution_method,
                    resolution_debug=resolution_debug,
                )
                effective_start_meta, effective_boundary_meta = build_effective_fallback_detail_metas(
                    lines=lines,
                    resolution_method=resolution_method,
                    resolution_debug=resolution_debug,
                )
                sections.append(section_payload)
                section_details[spec.item_key] = build_section_detail(
                    spec=spec,
                    matched=True,
                    start_meta=start_meta,
                    boundary_meta=None,
                    section_payload=section_payload,
                    effective_start_meta=effective_start_meta,
                    effective_boundary_meta=effective_boundary_meta,
                )
                continue
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
        payload_start_index = start_index
        resolution_method = "clean_text_heading_scan_v1"
        resolution_debug: dict[str, Any] | None = None
        fallback_payload = maybe_build_item8_part_iv_fallback(
            lines=lines,
            spec=spec,
            body_text=body_text,
            boundary_index=boundary_index,
        )
        if fallback_payload is not None:
            (
                payload_start_index,
                section_text,
                body_text,
                body_line_start,
                body_line_end,
                body_nonempty_line_count,
                resolution_method,
                resolution_debug,
            ) = fallback_payload
        section_payload = build_section_payload(
            spec=spec,
            start_index=payload_start_index,
            section_text=section_text,
            body_text=body_text,
            body_line_start=body_line_start,
            body_line_end=body_line_end,
            body_nonempty_line_count=body_nonempty_line_count,
            resolution_method=resolution_method,
            resolution_debug=resolution_debug,
        )
        effective_start_meta, effective_boundary_meta = build_effective_fallback_detail_metas(
            lines=lines,
            resolution_method=resolution_method,
            resolution_debug=resolution_debug,
        )
        sections.append(section_payload)
        section_details[spec.item_key] = build_section_detail(
            spec=spec,
            matched=True,
            start_meta=start_meta,
            boundary_meta=boundary_meta,
            section_payload=section_payload,
            effective_start_meta=effective_start_meta,
            effective_boundary_meta=effective_boundary_meta,
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
    if len(lowered) > 500:
        return False
    if (
        "incorporated by reference" in lowered
        or "incorporated herein by reference" in lowered
        or "included immediately following" in lowered
    ):
        return True
    financial_statement_markers = (
        "financial statements",
        "supplementary data",
        "notes thereto",
        "notes to those statements",
        "financial statement schedules",
        "financial statement schedule",
        "index to financial statements",
    )
    if not any(marker in lowered for marker in financial_statement_markers):
        return False
    reference_verbs = (
        "included",
        "set forth",
        "incorporated",
        "commencing",
        "beginning",
        "contained",
        "located",
    )
    if not any(marker in lowered for marker in reference_verbs):
        return False
    location_markers = (
        "part iv, item 15",
        "part iv item 15",
        "item 15(a)(1)",
        "item 15(a)",
        "page f-1",
        "page f 1",
        "commencing on page",
        "beginning on page",
        "annual report on form 10-k",
        "quarterly report on form 10-q",
        "this annual report on form 10-k",
        "this quarterly report on form 10-q",
        "this form 10-k",
        "this form 10-q",
        "this report",
    )
    if any(marker in lowered for marker in location_markers):
        return True
    return False


def looks_like_item8_part_iv_stub(text: str) -> bool:
    lowered = normalized_text(text).lower()
    if len(lowered) > 400:
        return False
    if "the information required by this item is set forth" not in lowered:
        return False
    if "consolidated financial statements" not in lowered:
        return False
    return "annual report on form 10-k" in lowered


def looks_like_statement_bundle_body(text: str) -> bool:
    collapsed = collapsed_text(text)
    return any(signal in collapsed for signal in COLLAPSED_STATEMENT_BUNDLE_SIGNALS)


def looks_like_financial_statement_payload_body(text: str) -> bool:
    if looks_like_statement_bundle_body(text):
        return True
    lowered = normalized_text(text).lower()
    if "assets" in lowered and "liabilities" in lowered:
        return True
    cues = (
        "year ended december",
        "december 31",
        "cash and cash equivalents",
        "total assets",
        "total liabilities",
        "stockholders' equity",
        "net income",
    )
    return sum(1 for cue in cues if cue in lowered) >= 2


def find_explicit_item_heading_after(
    lines: list[str],
    *,
    start_index: int,
    item_code: str,
    required_part_name: str | None = None,
) -> int | None:
    target_item_code = normalize_item_code(item_code)
    target_part_name = normalize_part_name(required_part_name)

    for index in range(max(0, start_index), len(lines)):
        line = lines[index].strip()
        if not line:
            continue
        if not looks_like_explicit_heading_lead(line):
            continue
        if parse_item_code_from_leading_text(line) != target_item_code:
            continue
        if target_part_name is not None and parse_part_name_near_index(lines, index) != target_part_name:
            continue
        if looks_like_toc_heading_candidate(lines, index, form="10-K"):
            continue
        return index
    return None


def find_signature_cluster_start(
    lines: list[str],
    *,
    start_index: int,
) -> int | None:
    for index in range(max(0, start_index), len(lines)):
        line = normalized_text(lines[index])
        if not line:
            continue
        lowered = line.lower()
        if lowered == "signatures":
            return index
        if "/s/" not in lowered:
            continue
        window = next_nonempty_lines(lines, index, max_lines=24)
        signature_marker_count = sum(1 for candidate in window if "/s/" in normalized_text(candidate).lower())
        if signature_marker_count >= 2:
            return index
    return None


def normalized_heading_key(text: str | None) -> str:
    return normalized_text(text or "").lower().strip(" .:-")


def find_exact_heading_line(
    lines: list[str],
    heading: str,
    *,
    start_index: int = 0,
) -> int | None:
    target = normalized_heading_key(heading)
    if not target:
        return None
    for index in range(max(0, start_index), len(lines)):
        if normalized_heading_key(lines[index]) == target:
            return index
    return None


def find_first_heading_in_set(
    lines: list[str],
    headings: Iterable[str],
    *,
    start_index: int = 0,
) -> tuple[int | None, str | None]:
    normalized_targets = [(normalized_heading_key(heading), heading) for heading in headings]
    for index in range(max(0, start_index), len(lines)):
        current = normalized_heading_key(lines[index])
        for normalized_target, original_heading in normalized_targets:
            if current == normalized_target:
                return index, original_heading
    return None, None


def find_nontraditional_10q_statement_block_start(
    lines: list[str],
    *,
    start_index: int = 0,
) -> int | None:
    for index in range(max(0, start_index), len(lines)):
        lowered = normalized_text(lines[index]).lower()
        if not (
            lowered.startswith("consolidated condensed statements of ")
            or lowered == "consolidated condensed balance sheets"
        ):
            continue
        following_lines = next_nonempty_lines(lines, index + 1, max_lines=8)
        following_probe = " ".join(normalized_text(candidate).lower() for candidate in following_lines[:6])
        if any(
            marker in following_probe
            for marker in (
                "three months ended",
                "nine months ended",
                "net revenue",
                "total assets",
                "net income",
                "cash and cash equivalents",
                "$",
            )
        ):
            return index
        page_like_count = sum(
            1
            for candidate in following_lines
            if ends_with_page_number(candidate) and len(normalized_text(candidate)) <= 90
        )
        if page_like_count >= 2:
            continue
        return index
    return None


def nontraditional_10q_structure_search_start(lines: list[str]) -> int:
    total_line_count = len(lines)
    return min(100, max(20, total_line_count // 6))


def looks_like_nontraditional_10q_main_body_family(lines: list[str]) -> bool:
    structure_start = nontraditional_10q_structure_search_start(lines)
    lowered_lines = [normalized_text(line).lower() for line in lines]
    if not any("traditional sec form 10-q format" in line for line in lowered_lines):
        return False
    if find_exact_heading_line(lines, "Form 10-Q Cross-Reference Index", start_index=structure_start) is None:
        return False
    if find_exact_heading_line(
        lines,
        "Consolidated Condensed Financial Statements and Supplemental Details",
        start_index=0,
    ) is None:
        return False
    statement_start = find_nontraditional_10q_statement_block_start(lines, start_index=0)
    if statement_start is None:
        return False
    mda_start, _ = find_first_heading_in_set(
        lines,
        (
            "Management's Discussion and Analysis",
            "Management's Discussion and Analysis (MD& A)",
        ),
        start_index=structure_start,
    )
    if mda_start is None or statement_start >= mda_start:
        return False
    controls_start = find_exact_heading_line(lines, "Controls and Procedures", start_index=mda_start + 1)
    if controls_start is None or mda_start >= controls_start:
        return False
    return True


def find_nontraditional_10q_item_boundaries(
    lines: list[str],
) -> dict[str, tuple[int, int, str, str]] | None:
    if not looks_like_nontraditional_10q_main_body_family(lines):
        return None

    structure_start = nontraditional_10q_structure_search_start(lines)
    statement_start = find_nontraditional_10q_statement_block_start(lines, start_index=0)
    mda_start, mda_label = find_first_heading_in_set(
        lines,
        (
            "Management's Discussion and Analysis",
            "Management's Discussion and Analysis (MD& A)",
        ),
        start_index=structure_start,
    )
    if statement_start is None or mda_start is None or mda_label is None or statement_start >= mda_start:
        return None

    item2_boundary, item2_boundary_label = find_first_heading_in_set(
        lines,
        (
            "Risk Factors and Other Key Information",
            "Other Key Information",
        ),
        start_index=mda_start + 1,
    )
    if item2_boundary is None or item2_boundary_label is None or mda_start >= item2_boundary:
        return None

    risk_start = find_exact_heading_line(lines, "Risk Factors", start_index=item2_boundary + 1)
    controls_start = find_exact_heading_line(lines, "Controls and Procedures", start_index=item2_boundary + 1)
    if risk_start is None or controls_start is None or risk_start >= controls_start:
        return None

    return {
        "part_i_item_1": (
            statement_start,
            mda_start,
            "Consolidated Condensed Statements",
            mda_label,
        ),
        "part_i_item_2": (
            mda_start,
            item2_boundary,
            mda_label,
            item2_boundary_label,
        ),
        "part_ii_item_1a": (
            risk_start,
            controls_start,
            "Risk Factors",
            "Controls and Procedures",
        ),
    }


def maybe_build_item8_part_iv_fallback(
    *,
    lines: list[str],
    spec: ItemSpec,
    body_text: str,
    boundary_index: int | None,
) -> tuple[int, str, str, int, int, int, str, dict[str, Any]] | None:
    if spec.form != "10-K" or spec.item_key != "item_8":
        return None
    if boundary_index is None:
        return None
    if not looks_like_item8_part_iv_stub(body_text):
        return None

    item15_start = find_explicit_item_heading_after(
        lines,
        start_index=boundary_index,
        item_code="15",
        required_part_name="Part IV",
    )
    if item15_start is None:
        return None
    item16_start = find_explicit_item_heading_after(
        lines,
        start_index=item15_start + 1,
        item_code="16",
    )
    if item16_start is None or item16_start <= item15_start + 1:
        return None

    fallback_body_text, fallback_body_line_start, fallback_body_line_end, fallback_body_nonempty_line_count = (
        materialize_body_from_line_range(
            lines,
            item15_start + 1,
            item16_start,
            spec,
            trim_leading_toc=False,
        )
    )
    if len(fallback_body_text) < max(spec.min_chars, 1000):
        return None
    if fallback_body_nonempty_line_count < 8:
        return None
    if not looks_like_financial_statement_payload_body(fallback_body_text):
        return None

    fallback_section_text = f"{standardized_heading(spec)}\n{fallback_body_text}"
    resolution_debug = {
        "fallback_item_key": "item_15",
        "fallback_boundary_item_key": "item_16",
        "fallback_line_start": item15_start + 1,
        "fallback_line_end": item16_start,
        "fallback_boundary_heading_line": item16_start + 1,
        "fallback_body_char_count": len(fallback_body_text),
    }
    return (
        item15_start,
        fallback_section_text,
        fallback_body_text,
        fallback_body_line_start,
        fallback_body_line_end,
        fallback_body_nonempty_line_count,
        "item8_part_iv_item15_fallback_v1",
        resolution_debug,
    )


def build_structural_fallback_payload(
    *,
    lines: list[str],
    spec: ItemSpec,
    start_index: int,
    boundary_index: int,
    resolution_method: str,
    fallback_heading_label: str,
    fallback_boundary_label: str,
) -> tuple[int, str, str, int, int, int, str, dict[str, Any]] | None:
    section_text, body_text, body_line_start, body_line_end, body_nonempty_line_count = build_section_text(
        lines,
        start_index,
        boundary_index,
        spec,
    )
    min_body_chars = max(spec.min_chars, 1000 if spec.body_profile == BODY_PROFILE_TABLE_HEAVY else 600)
    min_nonempty_lines = 8
    if spec.body_profile != BODY_PROFILE_TABLE_HEAVY:
        min_nonempty_lines = 1 if spec.min_chars <= 200 else 3
    if len(body_text) < min_body_chars:
        return None
    if body_nonempty_line_count < min_nonempty_lines:
        return None
    if spec.body_profile == BODY_PROFILE_TABLE_HEAVY and not looks_like_financial_statement_payload_body(body_text):
        return None

    resolution_debug = {
        "fallback_line_start": start_index + 1,
        "fallback_line_end": body_line_end,
        "fallback_boundary_heading_line": boundary_index + 1,
        "fallback_body_char_count": len(body_text),
        "fallback_heading_label": fallback_heading_label,
        "fallback_boundary_label": fallback_boundary_label,
    }
    return (
        start_index,
        section_text,
        body_text,
        body_line_start,
        body_line_end,
        body_nonempty_line_count,
        resolution_method,
        resolution_debug,
    )


def maybe_build_nontraditional_10q_main_body_fallback(
    *,
    lines: list[str],
    spec: ItemSpec,
    start_meta: dict[str, Any],
) -> tuple[int, str, str, int, int, int, str, dict[str, Any]] | None:
    if spec.form != "10-Q":
        return None
    if spec.item_key not in {"part_i_item_1", "part_i_item_2", "part_ii_item_1a"}:
        return None
    if start_meta.get("reason") != "no_heading_candidate":
        return None

    structural_boundaries = find_nontraditional_10q_item_boundaries(lines)
    if structural_boundaries is None:
        return None

    start_index, boundary_index, fallback_heading_label, fallback_boundary_label = structural_boundaries[spec.item_key]
    if start_index >= boundary_index:
        return None
    return build_structural_fallback_payload(
        lines=lines,
        spec=spec,
        start_index=start_index,
        boundary_index=boundary_index,
        resolution_method="nontraditional_10q_main_body_fallback_v1",
        fallback_heading_label=fallback_heading_label,
        fallback_boundary_label=fallback_boundary_label,
    )


def maybe_build_item8_headingless_statement_page_map_proxy(
    *,
    lines: list[str],
    spec: ItemSpec,
    start_meta: dict[str, Any],
) -> tuple[int, str, str, int, int, int, str, dict[str, Any]] | None:
    if spec.form != "10-K" or spec.item_key != "item_8":
        return None
    if start_meta.get("reason") != "only_index_candidates":
        return None

    proxy_shape = find_headingless_item8_statement_page_map_proxy(lines=lines, spec=spec)
    if proxy_shape is None:
        return None
    page_map_start, body_start = proxy_shape

    explicit_item9_start = find_explicit_item_heading_after(
        lines,
        start_index=body_start,
        item_code="9",
    )
    signature_cluster_start = find_signature_cluster_start(
        lines,
        start_index=body_start,
    )

    boundary_candidates = [
        candidate
        for candidate in (explicit_item9_start, signature_cluster_start)
        if candidate is not None and candidate > body_start
    ]
    body_end = min(boundary_candidates) if boundary_candidates else len(lines)
    if body_end <= body_start + 1:
        return None

    fallback_body_text, fallback_body_line_start, fallback_body_line_end, fallback_body_nonempty_line_count = (
        materialize_body_from_line_range(
            lines,
            body_start,
            body_end,
            spec,
            trim_leading_toc=False,
        )
    )
    if len(fallback_body_text) < max(spec.min_chars, 1000):
        return None
    if fallback_body_nonempty_line_count < 8:
        return None
    if not looks_like_financial_statement_payload_body(fallback_body_text):
        return None

    fallback_section_text = f"{standardized_heading(spec)}\n{fallback_body_text}"
    resolution_debug = {
        "proxy_page_map_start_line": page_map_start + 1,
        "fallback_line_start": body_start + 1,
        "fallback_line_end": body_end,
        "fallback_boundary_heading_line": None if explicit_item9_start is None else explicit_item9_start + 1,
        "fallback_signature_boundary_line": None if signature_cluster_start is None else signature_cluster_start + 1,
        "fallback_body_char_count": len(fallback_body_text),
    }
    return (
        body_start,
        fallback_section_text,
        fallback_body_text,
        fallback_body_line_start,
        fallback_body_line_end,
        fallback_body_nonempty_line_count,
        "item8_headingless_statement_page_map_proxy_v1",
        resolution_debug,
    )


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
        normalized = normalized_text(raw_line)
        if not normalized or is_decorative_line(normalized):
            continue
        lines.append(normalized)
    return lines


def body_open_lines(body_text: str, max_lines: int = 5) -> list[str]:
    lines = meaningful_body_lines(body_text)
    return lines[:max_lines]


def body_open_prose_count(body_text: str, max_lines: int = 5) -> int:
    return sum(1 for line in body_open_lines(body_text, max_lines=max_lines) if looks_like_prose_line(line))


def body_contains_toc_banner_leak(body_text: str) -> bool:
    return any(looks_like_toc_banner_line(line) for line in meaningful_body_lines(body_text))


def looks_like_statement_bundle_index_heading_line(text: str) -> bool:
    lowered = normalized_text(text).lower()
    if not lowered:
        return False
    return (
        "index to consolidated financial statements" in lowered
        or "index to financial statements" in lowered
    )


def looks_like_registered_auditor_report_heading(text: str) -> bool:
    lowered = normalized_text(text).lower()
    if not lowered:
        return False
    return "independent registered public accounting firm" in lowered


def statement_title_like_count(lines: Iterable[str]) -> int:
    return sum(
        1
        for line in lines
        if normalized_text(line).lower().startswith("consolidated ")
        or normalized_text(line).lower().startswith("notes to consolidated ")
    )


def looks_like_leading_statement_bundle_directory_prefix(
    body_text: str,
    spec: ItemSpec,
) -> bool:
    if spec.form != "10-K" or spec.item_key != "item_8":
        return False

    probe_lines = body_open_lines(body_text, max_lines=24)
    if len(probe_lines) < 6:
        return False

    opening_lines = probe_lines[:6]
    report_heading_present = any(
        looks_like_registered_auditor_report_heading(line) for line in probe_lines[:16]
    )
    title_like_count = statement_title_like_count(probe_lines[:16])
    has_directory_scaffold = (
        any(looks_like_page_column_header(line) for line in opening_lines)
        or any(looks_like_statement_bundle_index_heading_line(line) for line in opening_lines)
        or (report_heading_present and title_like_count >= 3)
    )
    if not has_directory_scaffold:
        return False

    early_lines = probe_lines[:20]
    page_locator_count = sum(1 for line in early_lines if looks_like_outline_page_locator_line(line))
    if page_locator_count < 4:
        return False
    if count_statement_bundle_signal_hits(early_lines) < 3 and title_like_count < 3:
        return False

    if not report_heading_present:
        return False

    return len(leading_cross_item_markers(body_text, spec, max_lines=12)) == 0


def looks_like_item8_front_index_statement_directory(
    body_text: str,
    spec: ItemSpec,
) -> bool:
    if spec.form != "10-K" or spec.item_key != "item_8":
        return False

    probe_lines = body_open_lines(body_text, max_lines=16)
    if len(probe_lines) < 6:
        return False

    early_lines = probe_lines[:12]
    if any(looks_like_registered_auditor_report_heading(line) for line in probe_lines[:24]):
        return False

    page_locator_count = sum(1 for line in early_lines if looks_like_outline_page_locator_line(line))
    if page_locator_count < 3:
        return False

    signal_count = sum(1 for line in early_lines if is_statement_bundle_signal_line(line))
    title_like_count = statement_title_like_count(early_lines)
    if signal_count < 2 and title_like_count < 3:
        return False

    return True


def leading_prose_run_before_outline_signal(body_text: str, max_lines: int = 16) -> int:
    run = 0
    for line in body_open_lines(body_text, max_lines=max_lines):
        if looks_like_prose_line(line):
            if (
                looks_like_toc_line(line)
                or looks_like_statement_bundle_page_map_line(line)
                or looks_like_outline_page_locator_line(line)
            ):
                break
            run += 1
            continue
        if (
            looks_like_toc_line(line)
            or looks_like_statement_bundle_page_map_line(line)
            or looks_like_outline_page_locator_line(line)
        ):
            break
        return 0
    return run


def looks_like_internal_outline_open(body_text: str, spec: ItemSpec, max_lines: int = 16) -> bool:
    if spec.form != "10-K":
        return False

    if looks_like_leading_statement_bundle_directory_prefix(body_text, spec):
        return False

    probe_lines = body_open_lines(body_text, max_lines=max_lines)
    if len(probe_lines) < 5:
        return False

    if leading_prose_run_before_outline_signal(body_text, max_lines=max_lines) >= 2:
        return False

    page_marker_count = sum(
        1
        for line in probe_lines
        if looks_like_outline_page_locator_line(line)
    )
    prose_count = sum(
        1
        for line in probe_lines
        if looks_like_prose_line(line)
        and not looks_like_toc_line(line)
        and not looks_like_statement_bundle_page_map_line(line)
        and not looks_like_outline_page_locator_line(line)
    )
    if page_marker_count >= 3:
        return True
    if prose_count >= 2:
        return False

    short_heading_count = sum(
        1
        for line in probe_lines
        if len(line) <= 120
        and not looks_like_prose_line(line)
        and not looks_like_explicit_heading_lead(line)
    )
    toc_like_count = sum(
        1
        for line in probe_lines
        if looks_like_toc_line(line) or looks_like_statement_bundle_page_map_line(line)
    )
    cross_item_count = len(leading_cross_item_markers(body_text, spec, max_lines=max_lines))

    if short_heading_count < 6:
        return False
    if spec.body_profile == BODY_PROFILE_TABLE_HEAVY:
        return cross_item_count >= 1 and short_heading_count >= 7
    if (
        prose_count == 0
        and short_heading_count >= 8
    ):
        return True
    if cross_item_count >= 1 and short_heading_count >= 7:
        return True
    return toc_like_count >= 2


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


def looks_like_risk_factor_carry_forward_stub(body_text: str) -> bool:
    lowered = normalized_text(body_text).lower()
    if len(lowered) > 260:
        return False
    risk_anchor = "item 1a" in lowered and "risk factors" in lowered
    if not risk_anchor:
        return False
    no_change_markers = (
        "there are no material changes",
        "no material changes to the risk factors",
    )
    if any(marker in lowered for marker in no_change_markers):
        return "form 10-k" in lowered or "previously disclosed" in lowered
    discussion_markers = (
        "for a discussion of our risk factors",
        "for a discussion of the risk factors",
        "see part i, item 1a",
        "see part i item 1a",
    )
    if any(marker in lowered for marker in discussion_markers):
        return "form 10-k" in lowered or "annual report" in lowered
    return False


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
    early_relative = total_line_count > 0 and line_start <= early_relative_position_threshold(total_line_count)
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
    if looks_like_internal_outline_open(body_text, spec):
        flags.append("internal_outline_open")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= late_position_warning_threshold(total_line_count):
        flags.append("too_early_for_item")
        status = promote_status(status, "warn")
    if spec.prefer_prose_start and body_open_prose_count(body_text, max_lines=4) == 0:
        flags.append("weak_prose_open")
        status = promote_status(status, "warn")
    if spec.item_key == "part_ii_item_1a" and looks_like_risk_factor_carry_forward_stub(body_text):
        flags.append("allowed_carry_forward_stub")
        return status, flags
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
    if looks_like_internal_outline_open(body_text, spec):
        flags.append("internal_outline_open")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= late_position_warning_threshold(total_line_count):
        flags.append("too_early_for_item")
        status = promote_status(status, "warn")
    if spec.allow_reference_only and looks_like_reference_stub(body_text):
        flags.append("allowed_reference_only")
        return status, flags
    if body_contains_toc_banner_leak(body_text):
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
    if looks_like_internal_outline_open(body_text, spec):
        flags.append("internal_outline_open")
        return "fail", flags
    if spec.require_late_document_position and int(section.get("line_start") or 0) <= late_position_warning_threshold(total_line_count):
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
