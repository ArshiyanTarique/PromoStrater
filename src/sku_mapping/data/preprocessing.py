"""Reusable, file-independent preprocessing from the production pipeline."""

from __future__ import annotations

import re

import pandas as pd

from sku_mapping.data.validators import normalize_itemcode_series
from sku_mapping.features.measurement_features import (
    collapse_to_simple,
    extract_flyer_measures,
    extract_master_measures,
)
from sku_mapping.features.semantic_features import _resolve_semantic_variant
from sku_mapping.features.text_features import clean_offer_text, safe_text

OWN_BRAND_ALIASES = {"al kabeer", "alkabeer", "al kabeer foods", "al kabeer group"}

CATEGORY_RULES = [
    ("Chicken", {"chicken"}),
    ("Meat", {"beef", "veal", "lamb", "mutton", "kofta", "kibbeh"}),
    ("Seafood", {"fish", "shrimp", "shrimps", "prawn", "prawns", "seafood", "dori", "tilapia", "salmon"}),
    ("Veg", {"vegetable", "vegetables", "veg", "peas", "corn", "spinach"}),
    ("Potato", {"potato", "potatoes", "fries", "wedges"}),
    ("Fruits", {"fruit", "fruits", "berry", "berries", "strawberry", "mango"}),
    ("Dough", {"paratha", "tortilla", "flour", "dough", "samosa", "wrap", "wraps"}),
]
CAT2_MAP = {
    "Chicken": "Chicken", "Chicken-Mince": "Chicken", "Chicken-Commodity": "Chicken", "Chicken-ZNG": "Chicken",
    "Meat": "Meat", "Meat-Commodity": "Meat", "Seafood": "Seafood", "Seafood-ZING": "Seafood", "Seafood-Commodity": "Seafood",
    "Fruits": "Fruits", "Veg": "Veg", "Potato": "Potato", "Potato-Commodity": "Potato", "Potato-Spl": "Potato", "Dough": "Dough",
}
PHRASE_RULES = [("Dough", ["spring roll", "spring rolls"])]
PRODUCT_TOKEN_ALIASES = {
    "nugget": "nuggets", "burger": "burgers", "patty": "patties", "sausage": "sausages",
    "strip": "strips", "wing": "wings", "prawn": "prawns", "shrimp": "shrimps",
    "paratha": "parathas", "samosa": "samosas", "roll": "rolls",
}


def normalize_brand(value: object) -> str:
    """Normalize brand spelling exactly as the legacy own-brand check does."""
    normalized = re.sub(r"[\s\-]+", " ", safe_text(value).lower()).strip()
    return re.sub(r"[^a-z0-9 ]", "", normalized)


def categorize(text: object) -> str:
    """Assign the existing hard category gate from product text."""
    normalized = safe_text(text).lower()
    for category, phrases in PHRASE_RULES:
        if any(phrase in normalized for phrase in phrases):
            return category
    tokens = set(re.findall(r"[a-z]+", normalized))
    for category, keywords in CATEGORY_RULES:
        if tokens & keywords:
            return category
    return "Other"


def normalize_product_family(text: object) -> str:
    """Normalize controlled product-family text without broad stemming."""
    normalized = re.sub(r"\bfrozen\b", " ", safe_text(text).lower())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(PRODUCT_TOKEN_ALIASES.get(token, token) for token in normalized.split()).strip()


def preprocess_clickflyer(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a prepared ClickFlyer copy using Phase 1 parsers and legacy rules."""
    prepared = frame.copy()
    for column in ("Offer Name", "Product", "Brand Name", "Variant", "Base Packsize"):
        prepared[column] = prepared[column].map(safe_text)
    prepared["brand_normalized"] = prepared["Brand Name"].map(normalize_brand)
    prepared["is_own"] = prepared["brand_normalized"].isin(OWN_BRAND_ALIASES)
    prepared["category"] = prepared["Product"].map(categorize)
    prepared["product_family"] = prepared["Product"].map(normalize_product_family)
    prepared["clean_offer_text"] = prepared["Offer Name"].map(clean_offer_text)
    semantic_variant = pd.Series(
        [
            _resolve_semantic_variant(offer_name, product, variant)
            for offer_name, product, variant in zip(
                prepared["Offer Name"],
                prepared["Product"],
                prepared["Variant"],
            )
        ],
        index=prepared.index,
        dtype="string",
    ).str.lower()
    semantic_variant = semantic_variant.where(
        ~semantic_variant.isin(["no variant", ""]), ""
    )
    prepared["match_text"] = (
        prepared["clean_offer_text"]
        + " "
        + prepared["Product"].str.lower().str.replace("-frozen", "", regex=False)
        + " "
        + semantic_variant
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared["offer_measures_detailed"] = (
        prepared["Base Packsize"] + " " + prepared["Offer Name"]
    ).map(extract_flyer_measures)
    prepared["offer_measures"] = prepared["offer_measures_detailed"].map(collapse_to_simple)
    return prepared


def preprocess_product_master(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a prepared Product Master copy using retail-safe master parsing."""
    prepared = frame.copy()
    prepared["Itemcode"] = normalize_itemcode_series(prepared["Itemcode"])
    for column in ("Itemname", "Item-Cat-2", "Item-Cat-4", "Item Description", "Item-Spec"):
        prepared[column] = prepared[column].map(safe_text)
    prepared["category"] = prepared["Item-Cat-2"].map(CAT2_MAP).fillna(
        prepared["Item-Cat-2"].map(categorize)
    )
    prepared["match_text"] = (
        prepared["Itemname"].str.lower()
        + " "
        + prepared["Item-Cat-4"].str.lower()
        + " "
        + prepared["Item Description"].str.lower()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared["master_measures_detailed"] = prepared["Item-Spec"].map(extract_master_measures)
    prepared["master_measures"] = prepared["master_measures_detailed"].map(collapse_to_simple)
    return prepared
