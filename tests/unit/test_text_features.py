"""Tests for legacy-compatible text normalization."""

import numpy as np
import pandas as pd

from sku_mapping.features.text_features import clean_offer_text, safe_text


def test_clean_offer_text_removes_brand_fillers_and_measurements() -> None:
    assert clean_offer_text("Al-Kabeer New Chicken Nuggets 400 Gms Offer") == "chicken nuggets"


def test_clean_offer_text_retains_business_tokens() -> None:
    assert clean_offer_text("Chicken Seekh Kebab Spicy") == "chicken seekh kebab spicy"


def test_missing_text_is_safe() -> None:
    assert safe_text(None) == ""
    assert safe_text(np.nan) == ""
    assert safe_text(pd.NA) == ""
    assert clean_offer_text(np.nan) == ""
    assert clean_offer_text(pd.NA) == ""
