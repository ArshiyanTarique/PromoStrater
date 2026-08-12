"""Product-family normalisation shared by competitor discovery.

Stage 3 groups competitor offers against own-SKU targets with an EXACT
string match on the normalised family, so a family string that differs by
a plural or a romanisation puts the competitor in a bucket that is never
compared. Widening the alias table is therefore a recall fix, not cosmetics.

Two rules keep the table safe, and both are enforced by
``audit_alias_collisions`` rather than by care alone:

1. An alias may only join SPELLINGS OF THE SAME THING - a plural, or a
   different romanisation of one dish. Synonyms for genuinely different
   products never belong here: "chicken fingers" and "chicken fries" are
   different products even though they read alike, and merging them would
   invent competitors that do not exist.
2. An alias may never map a token that is a different real word in some
   other product context. This is why generic stemming is refused
   ("patty" -> "party") and why the Urdu/Arabic "seekh" is left alone
   rather than aliased from "sikh".

Run ``audit_alias_collisions`` against every new client's data before
trusting the table on it - it reports exactly which distinct families the
aliases merged, so an unsafe merge is visible instead of silent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping

#: Plural/singular pairs. Canonical form is the PLURAL, matching the
#: original table's convention. Only nouns whose singular and plural denote
#: the same product appear here.
_PLURAL_ALIASES: dict[str, str] = {
    "nugget": "nuggets",
    "burger": "burgers",
    "patty": "patties",
    "pattie": "patties",
    "sausage": "sausages",
    "strip": "strips",
    "wing": "wings",
    "prawn": "prawns",
    "shrimp": "shrimps",
    "roll": "rolls",
    "fillet": "fillets",
    "filet": "fillets",
    "finger": "fingers",
    "fry": "fries",
    "wrap": "wraps",
    "wedge": "wedges",
    "cube": "cubes",
    "ball": "balls",
    "ring": "rings",
    "bite": "bites",
    "stick": "sticks",
    "frank": "franks",
    "frankfurter": "franks",
    "frankfurters": "franks",
    "hotdog": "hotdogs",
    "meatball": "meatballs",
    "tender": "tenders",
    "escalope": "escalopes",
    "lollipop": "lollipops",
    "pea": "peas",
    "vegetable": "vegetables",
    "cutlet": "cutlets",
    "skewer": "skewers",
    "steak": "steaks",
    "slice": "slices",
    "chunk": "chunks",
    "portion": "portions",
}

#: Romanisation variants. Each group is one dish spelled several ways in
#: Gulf/South-Asian product data. Grouped by canonical form so the intent
#: is auditable; flattened below.
_TRANSLITERATION_GROUPS: dict[str, tuple[str, ...]] = {
    # Stuffed pastry triangle. "sambosa" is the common Gulf spelling.
    "samosas": ("samosa", "samosas", "sambosa", "sambosas", "samboosa",
                "sambousa", "sambusa", "samoosa", "samosay"),
    # Bulgur-and-meat croquette. NOTE: distinct from kebab - do not merge.
    "kibbeh": ("kibbeh", "kibbe", "kibba", "kibbah", "kubba", "kubbe",
               "kubbeh", "kebbeh", "kubbah"),
    # Grilled skewer. NOTE: distinct from kibbeh - do not merge.
    "kebabs": ("kebab", "kebabs", "kabab", "kababs", "kabob", "kabobs",
               "kebob", "kebobs", "kebaab"),
    # Spiced meat patty/skewer.
    "kofta": ("kofta", "koftas", "kufta", "kuftas", "koftah", "kofte"),
    # Spit-roasted meat.
    "shawarma": ("shawarma", "shawarmas", "shawerma", "shwarma", "shawrma",
                 "shaworma", "chawarma"),
    # Chickpea fritter.
    "falafel": ("falafel", "falafels", "felafel", "falafil", "filafil"),
    # Marinated chicken (shish tawook).
    "tawook": ("tawook", "tawouk", "taouk", "tawuk", "tawook's", "tauk"),
    # Skewer prefix in "shish tawook" / "shish kebab".
    "shish": ("shish", "sheesh", "shesh", "shis"),
    # Flatbreads.
    "parathas": ("paratha", "parathas", "parata", "paratta", "porotha",
                 "porota", "prantha", "parantha"),
    "chapatti": ("chapatti", "chapati", "chappati", "chappathi", "chapathi"),
    # Layered pastry dessert.
    "kunafa": ("kunafa", "kunafah", "knafeh", "kanafeh", "konafa"),
    # Spiced rice dish.
    "biryani": ("biryani", "biriyani", "briyani", "biriani", "beriani"),
    # Fresh cheese.
    "paneer": ("paneer", "panir", "panner"),
    # Yoghurt drink / dip is NOT included: "lassi"/"laban" denote different
    # products in Gulf ranges, so they are synonyms, not spellings.
}

_TRANSLITERATION_ALIASES: dict[str, str] = {
    variant: canonical
    for canonical, variants in _TRANSLITERATION_GROUPS.items()
    for variant in variants
}

#: The public table. Kept flat and token-level so it stays a drop-in
#: replacement for the original dict.
PRODUCT_TOKEN_ALIASES: dict[str, str] = {
    **_PLURAL_ALIASES,
    **_TRANSLITERATION_ALIASES,
}

#: Families that must NEVER collapse together. These are real pairs that
#: read alike but are different products; they are asserted in the tests so
#: a future alias cannot quietly merge them.
MUST_STAY_DISTINCT: tuple[tuple[str, str], ...] = (
    ("chicken fingers", "chicken fries"),
    ("chicken rings", "chicken fries"),
    ("vegetable nuggets", "vegetable burgers"),
    ("chicken kibbeh", "chicken kebabs"),
    ("beef burger", "chicken burger"),
    ("chicken strips", "chicken fries"),
    ("beef kibbeh", "beef kofta"),
    ("chicken patty", "chicken party"),
    ("potato wedges", "potato bites"),
    ("fish fingers", "fish fillets"),
)


def normalize_product_family(
    text: object,
    aliases: Mapping[str, str] = PRODUCT_TOKEN_ALIASES,
) -> str:
    """Normalise a raw Product value into its bucket key.

    Lower-cases, splits on punctuation, drops the ``frozen`` marker, then
    maps each token through *aliases*. Token order is preserved and tokens
    are not de-duplicated, so the output stays a predictable function of
    the input. (De-duplication is deliberately refused: it would turn
    "meat kofta arabic kofta" into "meat kofta arabic" and lose meaning.)

    Punctuation is split BEFORE the ``frozen`` check. The previous
    ``\\bfrozen\\b`` form ran on the raw string, where ``_`` is a word
    character, so "Nuggets_Frozen" kept the marker while "Nuggets Frozen"
    dropped it - the same product in two buckets. Splitting first makes
    every separator behave identically.
    """
    value = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    tokens = [token for token in value.split() if token != "frozen"]
    return " ".join(aliases.get(token, token) for token in tokens).strip()


def audit_alias_collisions(
    raw_families: Iterable[object],
    aliases: Mapping[str, str] = PRODUCT_TOKEN_ALIASES,
) -> list[dict[str, object]]:
    """Report every group of distinct raw families the aliases merged.

    Returns one record per canonical family that more than one distinct raw
    family maps onto. A merge is expected and wanted when the raw strings
    are spelling variants; it is a BUG when they are different products.
    Reviewing this list is how the table is kept honest on data that was
    never seen when the table was written.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for raw in raw_families:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(raw).lower())
        cleaned = " ".join(cleaned.split()).strip()
        if not cleaned:
            continue
        grouped[normalize_product_family(raw, aliases)].add(cleaned)

    collisions = []
    for canonical, members in sorted(grouped.items()):
        if len(members) > 1:
            collisions.append(
                {"canonical": canonical, "merged": sorted(members),
                 "count": len(members)}
            )
    return collisions
