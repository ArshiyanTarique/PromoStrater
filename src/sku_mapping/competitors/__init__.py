"""Master-SKU-centric competitor discovery for business mapping outputs."""

from sku_mapping.competitors.discovery import (
    COMPETITOR_EXPORT_COLUMNS,
    COMPETITOR_LONG_COLUMNS,
    SUPPORTED_COMPETITOR_STATUSES,
    CompetitorDiscoveryResult,
    discover_competitors,
)

__all__ = [
    "COMPETITOR_EXPORT_COLUMNS",
    "COMPETITOR_LONG_COLUMNS",
    "SUPPORTED_COMPETITOR_STATUSES",
    "CompetitorDiscoveryResult",
    "discover_competitors",
]
