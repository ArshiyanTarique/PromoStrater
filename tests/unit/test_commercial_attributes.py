from sku_mapping.features.commercial_attributes import (
    MappingOutcome,
    MeasurementMatch,
    compare_commercial_attributes,
    parse_master_attributes,
    parse_source_attributes,
)


def _master(name: str, spec: str = "") -> dict[str, str]:
    return {
        "Itemname": name, "Item-Cat-2": "", "Item-Cat-4": "",
        "Item Description": "", "Item-Spec": spec,
    }


def test_source_protein_never_inherits_candidate_and_flags_bad_variant() -> None:
    source = parse_source_attributes({
        "Offer Name": "Chicken Samosas 240 gm",
        "Product": "Chicken Samosas", "Variant": "Mutton",
        "Base Packsize": "240 gm",
    })
    assert source.protein == ("chicken",)
    assert source.source_field_conflict is True
    wrong = compare_commercial_attributes(
        source, parse_master_attributes(_master("Mutton Samosas", "240 gm"))
    )
    assert wrong.outcome == MappingOutcome.UNACCEPTABLE_MATCH.value
    assert "PROTEIN_CONFLICT" in wrong.reason_codes


def test_measurement_roles_distinguish_promotion_from_exact_unit() -> None:
    source = parse_source_attributes({
        "Offer Name": "Chicken Nuggets twin promo 2 x 400 gm",
        "Product": "Chicken Nuggets", "Variant": "",
        "Base Packsize": "2 x 400 gm",
    })
    result = compare_commercial_attributes(
        source, parse_master_attributes(_master("Chicken Nuggets", "400 gm"))
    )
    assert result.measurement_match == MeasurementMatch.PROMOTION_MISMATCH.value
    assert result.outcome == MappingOutcome.ADAPTED_MATCH.value


def test_exact_candidate_and_wrong_family() -> None:
    source = parse_source_attributes({
        "Offer Name": "Chicken Nuggets 400 gm", "Product": "Chicken Nuggets",
        "Variant": "", "Base Packsize": "400 gm",
    })
    exact = compare_commercial_attributes(
        source, parse_master_attributes(_master("Chicken Nuggets", "0.4 kg"))
    )
    wrong = compare_commercial_attributes(
        source, parse_master_attributes(_master("Chicken Burger", "400 gm"))
    )
    assert exact.outcome == MappingOutcome.EXACT_MATCH.value
    assert wrong.outcome == MappingOutcome.UNACCEPTABLE_MATCH.value


def test_slash_bundle_is_adapted_review_evidence() -> None:
    source = parse_source_attributes({
        "Offer Name": "Chicken Nuggets / Chicken Samosas 400 gm",
        "Product": "", "Variant": "", "Base Packsize": "",
    })
    result = compare_commercial_attributes(
        source, parse_master_attributes(_master("Chicken Nuggets", "400 gm"))
    )
    assert source.slash_ambiguity is True
    assert result.outcome == MappingOutcome.ADAPTED_MATCH.value
    assert result.exact_match_eligible is False
