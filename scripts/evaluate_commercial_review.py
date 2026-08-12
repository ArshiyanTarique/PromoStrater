"""Evaluate reviewed proposal rows with generalized commercial evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sku_mapping.features.commercial_attributes import (  # noqa: E402
    compare_commercial_attributes,
    parse_master_attributes,
    parse_source_attributes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reviewed = pd.read_csv(args.review_summary, keep_default_na=False)
    records: list[dict[str, object]] = []
    for row in reviewed.to_dict(orient="records"):
        source = parse_source_attributes(
            {
                "Offer Name": row["source_offer_name"],
                "Product": "",
                "Variant": "",
                "Base Packsize": "",
            }
        )
        master = parse_master_attributes(
            {
                "Itemname": row["reviewed_master_name"],
                "Item-Cat-2": "",
                "Item-Cat-4": "",
                "Item Description": row["reviewed_master_description"],
                "Item-Spec": row["reviewed_master_description"],
            }
        )
        comparison = compare_commercial_attributes(source, master)
        records.append(
            {
                "workbook_row": row["workbook_row"],
                "source_offer_name": row["source_offer_name"],
                "reviewed_master_name": row["reviewed_master_name"],
                "reviewed_master_description": row[
                    "reviewed_master_description"
                ],
                "human_label": row["human_label"],
                "before_final_decision": row["final_decisions"],
                "before_reason": row["final_decision_reasons"],
                "before_hard_conflict": row["hard_conflict_any"],
                "after_mapping_outcome": comparison.outcome,
                "after_severity": comparison.severity,
                "after_measurement_match": comparison.measurement_match,
                "after_hard_conflict": comparison.hard_conflict,
                "after_exact_match_eligible": (
                    comparison.exact_match_eligible
                ),
                "after_reason_codes": "|".join(comparison.reason_codes),
                "parsed_source_family": "|".join(source.family),
                "parsed_source_protein": "|".join(source.protein),
                "parsed_source_variants": "|".join(source.variants),
                "parsed_source_base_measure": source.base_measure,
                "parsed_source_bonus_measure": source.bonus_measure,
                "parsed_source_total_measure": source.total_measure,
                "parsed_source_pack_count": source.pack_count,
                "parsed_source_piece_count": source.piece_count,
                "parsed_source_bundle_structure": source.bundle_structure,
                "parsed_source_ambiguity": (
                    source.multi_product or source.slash_ambiguity
                ),
                "source_parse_confidence": source.confidence,
            }
        )
    output = pd.DataFrame(records)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = args.output_directory / "reviewed_before_after.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    red = output["human_label"].eq("RED")
    yellow = output["human_label"].eq("YELLOW")
    green = output["human_label"].eq("GREEN")
    detected_red = output["after_mapping_outcome"].eq(
        "UNACCEPTABLE_MATCH"
    )
    detected_yellow = output["after_mapping_outcome"].eq("ADAPTED_MATCH")
    exact_green = output["after_mapping_outcome"].eq("EXACT_MATCH")
    metrics = {
        "rows": int(len(output)),
        "green_rows": int(green.sum()),
        "yellow_rows": int(yellow.sum()),
        "red_rows": int(red.sum()),
        "green_exact_detection": int((green & exact_green).sum()),
        "green_exact_rate": float(
            (green & exact_green).sum() / green.sum()
        ) if green.any() else None,
        "yellow_adapted_detection": int((yellow & detected_yellow).sum()),
        "yellow_adapted_rate": float(
            (yellow & detected_yellow).sum() / yellow.sum()
        ) if yellow.any() else None,
        "red_unacceptable_detection": int((red & detected_red).sum()),
        "red_unacceptable_recall": float(
            (red & detected_red).sum() / red.sum()
        ) if red.any() else None,
        "unacceptable_precision": float(
            (red & detected_red).sum() / detected_red.sum()
        ) if detected_red.any() else None,
        "unsafe_red_exact_eligible": int(
            (red & output["after_exact_match_eligible"]).sum()
        ),
        "model_retrained": False,
        "model_feature_contract_changed": False,
    }
    (args.output_directory / "reviewed_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
