import pytest

from sku_mapping.features.commercial_entities import (
    decompose_commercial_entities,
)


def _entities(text: str):
    return decompose_commercial_entities(
        {"offerid": "12345", "Offer Name": text}
    )


def test_shared_family_different_proteins_and_size() -> None:
    entities = _entities("Chicken / Beef Burger Patty 400 gm")
    assert len(entities) == 2
    assert [entity.protein for entity in entities] == [
        ("chicken",),
        ("beef",),
    ]
    assert all(entity.product_family == ("burger patty",) for entity in entities)
    assert all(entity.retail_weight_g == 400 for entity in entities)
    assert all("retail_weight_g" in entity.inherited_attributes for entity in entities[:1])


def test_separate_sizes_are_not_inherited() -> None:
    entities = _entities(
        "Chicken Nuggets 400 gm + Chicken Popcorn 500 gm"
    )
    assert len(entities) == 2
    assert [entity.retail_weight_g for entity in entities] == [400, 500]


def test_shared_protein_and_size_are_inherited() -> None:
    entities = _entities("Chicken Nuggets / Popcorn 400 gm")
    assert len(entities) == 2
    assert all(entity.protein == ("chicken",) for entity in entities)
    assert [entity.product_family for entity in entities] == [
        ("nuggets",),
        ("popcorn",),
    ]
    assert all(entity.retail_weight_g == 400 for entity in entities)


@pytest.mark.parametrize(
    "text",
    [
        "Chicken and Cheese Sausages 400 gm",
        "Sweet and Spicy Chicken Wings",
        "Beef and Herb Meatballs",
    ],
)
def test_descriptor_conjunctions_do_not_false_split(text: str) -> None:
    assert len(_entities(text)) == 1


def test_or_preserves_alternative_semantics() -> None:
    entities = _entities("Chicken Samosas or Beef Samosas 240 gm")
    assert len(entities) == 2
    assert {entity.conjunction_type for entity in entities} == {"OR"}
    assert all(entity.retail_weight_g == 240 for entity in entities)


def test_promotional_quantity_does_not_create_fake_entity() -> None:
    entities = _entities("Chicken Nuggets 800 gm + 200 gm Free")
    assert len(entities) == 1


def test_free_product_is_separate_bonus_entity() -> None:
    entities = _entities("Chicken Nuggets + Free Chicken Popcorn")
    assert len(entities) == 2
    assert entities[1].entity_type == "PROMOTIONAL_BONUS"
    assert entities[1].conjunction_type == "PROMOTIONAL_BONUS"


def test_mixed_pack_without_evidence_is_not_split() -> None:
    assert len(_entities("Mixed Grill Pack")) == 1


def test_distinct_families_and_proteins_split() -> None:
    entities = _entities("Chicken Nuggets and Beef Burger Patties")
    assert len(entities) == 2
    assert entities[0].product_family == ("nuggets",)
    assert entities[1].product_family == ("burger patty",)
