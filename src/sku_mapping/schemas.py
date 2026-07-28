"""Input schema definitions shared by loaders and validators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSchema:
    """Required and supported optional column names for one input table."""

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()


CLICKFLYER_SCHEMA = TableSchema(
    name="ClickFlyer CSV",
    required_columns=(
        "Offer Name",
        "Product",
        "Brand Name",
        "Variant",
        "Base Packsize",
        "Country",
        "Retailer Name",
        "Flyer Name",
        "offerid",
        "Offer Price",
        "Regular Price",
    ),
)

PRODUCT_MASTER_SCHEMA = TableSchema(
    name="Product Master Excel",
    required_columns=(
        "Itemcode",
        "Itemname",
        "Item-Cat-2",
        "Item-Cat-4",
        "Item Description",
        "Item-Spec",
    ),
)

GOLD_PAIRS_SCHEMA = TableSchema(
    name="Gold training-pairs CSV",
    required_columns=(
        "offer_group_id",
        "offer_text",
        "master_itemcode",
        "pair_label",
        "use_for_binary_pair_training",
    ),
    optional_columns=(
        "record_id",
        "source_dataset",
        "product_class_offer",
        "variant_offer",
        "recommended_split",
        "split_group",
        "label_provenance",
        "label_confidence",
    ),
)

MODEL_PACKAGE_REQUIRED_KEYS = ("model", "feature_columns")
