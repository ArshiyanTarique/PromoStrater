"""Unit tests for dashboard enum-to-label formatters and badge utils."""

from __future__ import annotations

from dashboard.components.formatters import (
    format_enum_label,
    format_embedding_status,
    get_status_color,
    render_badge_html,
)


def test_format_enum_label_known_values() -> None:
    assert format_enum_label("EMBEDDING_FAILURE_SAFE_FALLBACK") == "Embedding Fallback"
    assert format_enum_label("EMBEDDING_UNAVAILABLE") == "Embedding Unavailable"
    assert format_enum_label("AGREEMENT") == "Agreement Policy"
    assert format_enum_label("AUTO_ACCEPT") == "Auto Accept"


def test_format_enum_label_fallback() -> None:
    assert format_enum_label("SOME_UNKNOWN_ENUM_VALUE") == "Some Unknown Enum Value"
    assert format_enum_label("custom-status_code") == "Custom Status Code"
    assert format_enum_label(None) == "N/A"
    assert format_enum_label("") == "N/A"


def test_format_embedding_status() -> None:
    label, color = format_embedding_status("ACTIVE")
    assert label == "Active / Used"
    assert color == "success"

    label, color = format_embedding_status("EMBEDDING_UNAVAILABLE", "Model missing")
    assert "Embedding Unavailable" in label
    assert "Model missing" in label
    assert color == "error"


def test_render_badge_html() -> None:
    html = render_badge_html("Auto Accept", "success")
    assert "ps-badge-success" in html
    assert "Auto Accept" in html
