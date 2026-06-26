# Filing Section Extractor

This directory contains the filing-section extraction code used by FINSABER to
turn full-text SEC filings into item-level section text.

The extractor is designed for one specific job:

- take clean-text `10-K` or `10-Q` filing content,
- locate a requested SEC item,
- cut the corresponding body section,
- run a lightweight quality audit,
- and expose the result to downstream strategies without changing the core
  `TradingData` contract.

The current integration target is FinMem, but the extraction layer itself is
generic. Any strategy can request a different filing item as long as it is part
of the supported item registry.

## What Is In This Directory

- `upstream_extractor.py`
  A self-contained extractor that includes item definitions, heading matching,
  boundary detection, section building, auditing, and an optional standalone
  CLI.
- `dataset_overlay.py`
  A `TradingData` wrapper that replaces raw filing modalities with extracted
  section text at runtime.
- `__init__.py`
  Public exports for the extractor primitives and the dataset overlay.

This folder intentionally keeps only one README. All local design notes and
copied upstream markdown files were removed so the implementation and its usage
stay easy to understand in one place.

## Runtime Role Inside FINSABER

FINSABER does not rewrite its global data model for filing sections.
Instead, the extractor is used as a thin overlay on top of an existing
`TradingData` loader.

The runtime flow is:

1. a base loader provides normal daily modalities such as `price`, `news`,
   `filing_k`, and `filing_q`,
2. `FilingSectionOverlayDataset` intercepts the requested filing modalities,
3. the raw full-text filing is converted into the requested item section,
4. all other modalities pass through unchanged.

This keeps the change local to the data-util layer and avoids broad framework
changes.

## Current FinMem Mapping

The current FinMem integration uses the extractor in the following way:

- `filing_k` -> `10-K / item_7`
- `filing_q` -> `10-Q / part_i_item_2`

That mapping lives outside this directory in the FinMem data-loading path. The
extractor itself does not hardcode any FinMem-specific behavior.

## Core Concepts

### 1. `ItemSpec`

Each supported filing item is described by an `ItemSpec`. An item spec defines:

- the target form (`10-K` or `10-Q`),
- the target part name when applicable,
- the item code,
- canonical heading text,
- heading fragments used for matching,
- candidate boundary items,
- the expected body profile,
- primary-target status,
- minimum text-length expectations,
- and a few audit-related tolerances.

This makes the extractor item-driven instead of regex-driven.

### 2. Body Profiles

The extractor uses three body profiles:

- `narrative`
  Used for long prose sections such as MD&A.
- `table_heavy`
  Used for sections that often open with statements or tables.
- `short_disclosure`
  Used for naturally short items or allowed stubs.

The body profile changes how candidate headings are scored and how suspicious
early matches are treated.

### 3. Runtime Overlay

`FilingSectionOverlayDataset` wraps any `TradingData` implementation and keeps
subset behavior intact. That matters because FINSABER frequently narrows a data
loader by time range and by ticker during backtests.

The overlay:

- preserves `TradingData` methods such as `get_subset_by_time_range(...)`,
- copies day-level filing payloads before replacing them,
- caches extraction results by `(modality, form, item, filing-text hash)`,
- and delegates everything unrelated to filing sections back to the base loader.

## Supported Items

### 10-K

- `item_1`
- `item_1a`
- `item_5`
- `item_7`
- `item_7a`
- `item_8`
- `item_9a`

### 10-Q

- `part_i_item_1`
- `part_i_item_2`
- `part_i_item_3`
- `part_i_item_4`
- `part_ii_item_1`
- `part_ii_item_1a`
- `part_ii_item_2`

These are the items currently encoded in `ITEM_SPECS`.

## How Extraction Works

The extraction pipeline in `upstream_extractor.py` is clean-text based. It does
not parse HTML or EDGAR structure directly.

At a high level, the extractor does this:

1. normalize and split the filing text into lines,
2. scan for explicit item headings,
3. build heading candidates only when the heading text and body text both look
   plausible,
4. score each candidate using heading fragments, part matching, body shape, TOC
   penalties, and document-position heuristics,
5. choose the best candidate,
6. find the next valid item boundary,
7. build the final section text,
8. audit the result.

## Candidate Selection Logic

The extractor does not trust a heading match on its own.

A candidate is preferred when it has signals such as:

- an explicit `Item` or `Part` heading prefix,
- heading fragments that match the target item,
- a plausible body after the heading,
- the right part name,
- and a document position that makes sense for that item.

A candidate is penalized when it looks like:

- a table-of-contents row,
- an early false match,
- a heading with no real body,
- or a cross-item body segment.

This is especially important for filings where the same item name appears in the
table of contents before the real section begins.

## Boundary Detection

Once the best start heading is found, the extractor looks for the next item
boundary listed in the target item's `boundary_candidates`.

The section body is the text between:

- the resolved start heading,
- and the next acceptable boundary heading.

This gives a stable item section without requiring any upstream document
structure beyond clean text.

## Auditing

Each extracted result is audited after extraction.

The audit logic checks things such as:

- whether the section is missing,
- whether it is too short,
- whether it looks like a heading-only result,
- whether it still resembles a TOC match,
- whether short or reference-only content is acceptable for that item,
- and whether overlapping section ranges look suspicious.

The audit output includes:

- an overall `audit_status`,
- item-level audit records,
- and item-level flags.

The runtime overlay uses that audit signal when deciding how to expose the
result back to the strategy.

## Failure Modes

`FilingSectionOverlayDataset` supports three failure policies:

- `empty`
  Return an empty string when extraction fails or the audit fails.
- `raw`
  Fall back to the original full filing text.
- `raise`
  Raise an exception with extraction context.

The current recommended runtime policy for strategy integration is `empty`,
because it avoids silently feeding suspicious raw text into item-based prompts.

## Public API

The main exported objects are:

- `ITEM_SPECS`
- `FilingRow`
- `ItemSpec`
- `resolve_requested_items(...)`
- `item_specs_for_request(...)`
- `extract_items_from_filing(...)`
- `audit_extraction_result(...)`
- `FilingSectionOverlayDataset`
- `with_filing_sections(...)`

## Example: Direct Extraction

```python
from backtest.data_util.filing_section_extractor import (
    FilingRow,
    audit_extraction_result,
    extract_items_from_filing,
    item_specs_for_request,
)

filing = FilingRow(
    date="2024-02-02",
    symbol="AMZN",
    cik="",
    accession="",
    form="10-K",
    year=2024,
    filing_idx=0,
    filing_text=raw_filing_text,
    source_file="",
    source_row_idx=0,
)

specs = item_specs_for_request(["10-K"], ["item_7"])
result = extract_items_from_filing(filing=filing, item_specs=specs)
audit = audit_extraction_result(
    filing=filing,
    extraction_result=result,
    item_specs=specs,
)
```

## Example: Loader Overlay

```python
from backtest.data_util.filing_section_extractor import with_filing_sections

loader = with_filing_sections(
    data_loader=base_loader,
    section_map={
        "filing_k": {"form": "10-K", "item_key": "item_7"},
        "filing_q": {"form": "10-Q", "item_key": "part_i_item_2"},
    },
    failure_mode="empty",
)
```

After wrapping the loader, calls such as `get_data_by_date(...)` will return the
same day structure as before, except that the selected filing modalities now
contain extracted section text instead of the original full filing text.

## Standalone Script Notes

`upstream_extractor.py` still contains a standalone CLI. That script can be used
for offline inspection or batch extraction, but the current FINSABER runtime
integration does not depend on its output files or CLI pipeline.

In other words:

- the standalone script remains available,
- the runtime backtest path uses the importable extraction primitives,
- and the overlay is the main integration point for strategies.

## Scope And Limitations

This extractor is intentionally narrow in scope.

It assumes:

- clean-text filing input,
- supported `10-K` and `10-Q` items only,
- and item-based consumption downstream.

It does not currently try to be:

- a full SEC document parser,
- an HTML-aware section resolver,
- or a universal filing reconstruction framework.

That narrow scope is deliberate. The goal is to provide reliable, low-friction
item extraction for backtesting workflows while keeping the surrounding
architecture stable.
