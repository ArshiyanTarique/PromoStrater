"""Check the product-family alias table against a NEW client's data.

The alias table widens competitor-discovery buckets. On data it was not
written against, the risk is that it merges two families that are actually
different products, which would invent competitors that do not exist.

This script makes that visible before the pipeline runs: it lists every
merge the table performs on the supplied file, together with the exact
token pair that caused it, so a human can confirm each one once per client.

It deliberately does NOT auto-classify a merge as safe or unsafe. String
similarity cannot make that call - measured on this domain, genuine
variants span 67-92 ("kunafa"/"knafeh" = 67) while genuinely different
products span 36-80 ("patty"/"party" = 80). The ranges overlap, so any
threshold both clears bad merges and flags good ones. The only automatic
check that is sound is the MUST_STAY_DISTINCT list, which is asserted
here and in the unit tests; everything else is reported for human review.

Usage:
    python scripts/audit_product_families.py <file.csv|file.xlsx> [--column Product]
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from sku_mapping.features.product_family import (  # noqa: E402
    MUST_STAY_DISTINCT,
    PRODUCT_TOKEN_ALIASES,
    audit_alias_collisions,
    normalize_product_family,
)


def _load(path: str, column: str) -> pd.Series:
    if path.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if column not in frame.columns:
        raise SystemExit(
            f"column {column!r} not found. Available: {list(frame.columns)[:25]}"
        )
    return frame[column].dropna()


def _responsible_tokens(members: list[str]) -> list[tuple[str, str]]:
    """The token pairs that differ between merged members.

    This is the actionable part of a merge: 'beef kibbeh' and 'beef kubbe'
    were joined because kibbeh<->kubbe are aliased, and that token pair is
    what a reviewer needs to judge - not the whole family string, whose
    shared words inflate any similarity score.
    """
    pairs: list[tuple[str, str]] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            left, right = members[i].split(), members[j].split()
            only_left = [t for t in left if t not in set(right)]
            only_right = [t for t in right if t not in set(left)]
            if only_left or only_right:
                pairs.append((" ".join(only_left) or "-",
                              " ".join(only_right) or "-"))
    return sorted(set(pairs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--column", default="Product")
    args = parser.parse_args()

    values = _load(args.path, args.column)
    raw = values.unique()
    counts = values.value_counts()

    families = {r: normalize_product_family(r) for r in raw}
    print(f"rows                 {len(values):,}")
    print(f"distinct raw values  {len(raw):,}")
    print(f"distinct families    {len(set(families.values())):,}")
    print(f"alias entries        {len(PRODUCT_TOKEN_ALIASES):,}")

    # The one sound automatic check: a merge that violates a pair we have
    # already declared must never join is a hard error, not a judgement call.
    violations = [
        (left, right) for left, right in MUST_STAY_DISTINCT
        if normalize_product_family(left) == normalize_product_family(right)
    ]
    if violations:
        print("\nFAIL - the alias table merges families declared distinct:")
        for left, right in violations:
            print(f"  {left!r} == {right!r}")
        print("\nFix src/sku_mapping/features/product_family.py before running "
              "the pipeline.")
        return 1

    collisions = audit_alias_collisions(raw)
    if not collisions:
        print("\nNo merges performed - the alias table changed nothing here.")
        return 0

    print(f"\n{len(collisions)} merge group(s) - confirm each is one product:\n")
    for group in collisions:
        members = [str(m) for m in group["merged"]]
        rows = sum(int(counts.get(r, 0)) for r in raw
                   if normalize_product_family(r) == group["canonical"])
        print(f"  {group['canonical']}   ({rows:,} rows)")
        for member in members:
            print(f"      <- {member}")
        for left, right in _responsible_tokens(members):
            print(f"      caused by:  {left!r} <-> {right!r}")
        print()

    print("Each merge above says 'these are the same product'. If any is "
          "wrong, remove the\nresponsible alias from "
          "src/sku_mapping/features/product_family.py and add the pair\n"
          "to MUST_STAY_DISTINCT so a test keeps it separated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
