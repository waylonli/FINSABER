"""Vendored filing section extractor.

This package currently vendors the upstream standalone extractor for local
exploration inside the FINSABER repository. We intentionally expose only the
core extraction and audit primitives here so downstream adapters can build on
them without depending on the upstream CLI/output pipeline.
"""

from .upstream_extractor import (
    ITEM_SPECS,
    FilingRow,
    ItemSpec,
    audit_extraction_result,
    extract_items_from_filing,
    item_specs_for_request,
    resolve_requested_items,
    validate_item_specs,
)
from .dataset_overlay import FilingSectionOverlayDataset, with_filing_sections

__all__ = [
    "ITEM_SPECS",
    "FilingSectionOverlayDataset",
    "FilingRow",
    "ItemSpec",
    "audit_extraction_result",
    "extract_items_from_filing",
    "item_specs_for_request",
    "resolve_requested_items",
    "validate_item_specs",
    "with_filing_sections",
]
