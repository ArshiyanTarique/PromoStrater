import pandas as pd
import numpy as np
import re
import time
import json
import requests
from rapidfuzz import fuzz, process

FLYER_PATH = "Alkabeer_Export_Data_Clickflyer.csv"
MASTER_PATH = "Product_Master.xlsx"

OUT_CSV = "outputs/sku_mapping_output.csv"
COMPETITOR_LONG_CSV = "outputs/competitor_matches_long.csv"
VALIDATION_SAMPLE_CSV = "outputs/manual_validation_sample.csv"
MANUAL_REVIEW_CSV = "outputs/manual_mapping_review_v2.csv"

# -----------------------------------------------------------------
# Ollama LLM revision settings (Stage 2.5)
# -----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60          # seconds per request
OLLAMA_RETRIES = 1           # retries on timeout/connection error
OLLAMA_BATCH_SIZE = 50       # review many rows in one model request
OLLAMA_NUM_PREDICT = 260     # enough for compact "id:YES/NO" answers
LLM_CHECKPOINT_CSV = "outputs/ollama_revision_checkpoint.csv"
# Tiers that will be sent to the LLM for revision.
LLM_REVIEW_TIERS = {"medium", "low", "low_pack_conflict", "low_structure_conflict"}
# After LLM says YES, the row gets this tier (treated as auto-accepted).
LLM_HIGH_REVISED = "high (revised)"
# After LLM says NO, the row keeps this tier (stays as review-required).
LLM_LOW_REVISED  = "low (revised)"

# Which confidence tiers get auto-accepted (a real matched_itemcode/matched_itemname
# and are used as competitor-discovery targets) vs REVIEW_REQUIRED.
# "high (revised)" is added here so LLM-confirmed medium/low rows are also
# auto-accepted after Stage 2.5 revision.
AUTO_ACCEPT_TIERS = ["high", "high (revised)"]

print("RUNNING SKU MAPPING PIPELINE V2 - ITEM DESCRIPTION REVIEW EXPORT")
t_start = time.time()
df = pd.read_csv(FLYER_PATH, low_memory=False, dtype={"offerid": str})
master = pd.read_excel(MASTER_PATH, dtype={"Itemcode": str})

for col in ["Offer Name", "Product", "Brand Name", "Variant", "Base Packsize"]:
    df[col] = df[col].fillna("")
for col in ["Itemname", "Item-Cat-2", "Item-Cat-4", "Item Description", "Item-Spec"]:
    master[col] = master[col].fillna("")

# -----------------------------------------------------------------
# Fix (Itemcode normalization): Excel can coerce identifier columns to
# floats (e.g. 12345.0) or strip leading zeros. Normalize defensively.
# -----------------------------------------------------------------
master["Itemcode"] = (
    master["Itemcode"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
)

# -----------------------------------------------------------------
# Fix (brand matching too fragile / too broad): normalize + compare
# against an explicit alias set instead of a loose substring regex.
# -----------------------------------------------------------------
OWN_BRAND_ALIASES = {"al kabeer", "alkabeer", "al kabeer foods", "al kabeer group"}

def normalize_brand(x):
    x = re.sub(r"[\s\-]+", " ", str(x).lower()).strip()
    x = re.sub(r"[^a-z0-9 ]", "", x)
    return x

df["brand_normalized"] = df["Brand Name"].apply(normalize_brand)
# Fix (#1 - is_own must be a real column, not an external Series that can
# desync from df after merges/filters): compute once, store as a column,
# always reference df["is_own"] from here on.
df["is_own"] = df["brand_normalized"].isin(OWN_BRAND_ALIASES)
print(f"Own-brand rows identified: {df['is_own'].sum():,}")

FILLER_WORDS = {"krazee","jumbo","promo","free","new","nispc","special","offer","combo","value","mega","family"}

def clean_offer_text(text):
    text = re.sub(r"al[\s\-]?kabeer", " ", str(text).lower())
    words = re.findall(r"[a-z]+", text)
    return " ".join(w for w in words if w not in FILLER_WORDS
                     and w not in ("gm","kg","g","kgs","gms","lb","lbs","oz","ml","l"))

# -----------------------------------------------------------------
# Fix (ml/l wrongly treated as grams): keep weight and volume as
# separate dimensions. Only compare weight-to-weight, volume-to-volume.
# -----------------------------------------------------------------
WEIGHT_UNITS = {"kg":1000, "kgs":1000, "g":1, "gm":1, "gms":1, "lb":453.592, "lbs":453.592, "oz":28.3495}
VOLUME_UNITS = {"ml":1, "l":1000, "ltr":1000, "litre":1000, "liter":1000}
ALL_UNITS = "|".join(sorted(set(list(WEIGHT_UNITS) + list(VOLUME_UNITS)), key=len, reverse=True))

def unit_dim_value(unit):
    if unit in WEIGHT_UNITS:
        return "weight", WEIGHT_UNITS[unit]
    return "volume", VOLUME_UNITS[unit]

def pack_is_compatible(offer_measures, master_measures, tol=0.10):
    """None = unknown (can't verify either side). True/False otherwise.
    Only compares values within the SAME dimension (weight vs weight,
    volume vs volume) so a 1L bottle is never treated as a 1kg bag."""
    if not offer_measures or not master_measures:
        return None
    for ov, od in offer_measures:
        for mv, md in master_measures:
            if od == md and abs(ov - mv) / max(ov, mv) <= tol:
                return True
    return False

def extract_measures_detailed(text, include_total=True, include_unit=True):
    """Returns list of (total_value, dimension, unit_count) tuples.
    For any "N x size" expression: include_total controls whether the
    bundle/case total (count=N) is emitted, include_unit controls whether
    the per-unit size (count=1) is emitted.

    Fix (source-specific interpretation): the SAME "N x size" notation means
    two different things on each side of this pipeline -- a flyer's
    "2 x 500g" is usually a genuine promotional twin-pack (the total is a
    meaningful, real quantity a shopper buys), while this master's
    Item-Spec is virtually always case/carton packaging like
    "270 Gms x 20 Pkts" (270g retail packet, 20 per distributor case) where
    the CASE TOTAL (5400g) is meaningless for retail matching -- no flyer
    ever describes a 5.4kg retail purchase of nuggets. Emitting master's
    case total as a candidate reading created a narrow but real false-match
    risk (an unrelated flyer's own large bulk total coincidentally landing
    within tolerance of a master case-total). Use extract_flyer_measures /
    extract_master_measures below rather than calling this directly."""
    if not isinstance(text, str):
        return []
    t = text.lower().replace(",", "")
    out = []
    occupied = []

    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*[x×\*]\s*(\d+(?:\.\d+)?)\s*({ALL_UNITS})\b", t):
        occupied.append(m.span())
        a, b, unit = m.groups()
        dim, factor = unit_dim_value(unit)
        count = int(round(float(a)))
        if include_total:
            out.append((float(a) * float(b) * factor, dim, count))   # bundle/case total
        if include_unit:
            out.append((float(b) * factor, dim, 1))                   # per-unit retail size
    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*({ALL_UNITS})\s*[x×\*]\s*(\d+(?:\.\d+)?)\b", t):
        occupied.append(m.span())
        size, unit, mult = m.groups()
        dim, factor = unit_dim_value(unit)
        count = int(round(float(mult)))
        if include_total:
            out.append((float(size) * float(mult) * factor, dim, count))  # bundle/case total
        if include_unit:
            out.append((float(size) * factor, dim, 1))                     # per-unit retail size

    def inside_multipack(span):
        return any(span[0] >= a and span[1] <= b for a, b in occupied)

    for m in re.finditer(rf"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*({ALL_UNITS})\b", t):
        if inside_multipack(m.span()):
            continue
        n1, n2, unit = m.groups()
        dim, factor = unit_dim_value(unit)
        out.append((float(n1) * factor, dim, 1))
        if n2:
            out.append((float(n2) * factor, dim, 1))
    return sorted(set((round(v, 1), d, c) for v, d, c in out if 1 <= v <= 30000))

def extract_flyer_measures(text):
    """Flyer offer text: a promotional 'N x size' is a genuine bundle a
    shopper buys, so both the total and the per-unit size are meaningful."""
    return extract_measures_detailed(text, include_total=True, include_unit=True)

def extract_master_measures(text):
    """Master Item-Spec: virtually always distributor case/carton config
    (verified: 231/237 rows follow 'unit_size x case_count'), so the case
    TOTAL is not a retail-comparable quantity -- only the per-unit size is."""
    return extract_measures_detailed(text, include_total=False, include_unit=True)

def collapse_to_simple(detailed_measures):
    """Derive the (value, dimension) view -- used by the loose,
    count-ignorant pack_is_compatible check -- from whichever detailed list
    (flyer or master) it's given, so the two views can never drift out of
    sync with each other the way they briefly did."""
    return sorted(set((v, d) for v, d, c in detailed_measures))

def pack_structure_agrees(offer_details, master_details, tol=0.10):
    """Stricter than pack_is_compatible: also requires the unit_count to
    match (both single-pack, or both the same N-pack) wherever a total-size
    match was found. Returns True/False/None (None = can't tell)."""
    if not offer_details or not master_details:
        return None
    any_total_match = False
    for ov, od, oc in offer_details:
        for mv, md, mc in master_details:
            if od == md and abs(ov - mv) / max(ov, mv) <= tol:
                any_total_match = True
                if oc == mc:
                    return True
    return False if any_total_match else None

# -----------------------------------------------------------------
# Fix (category false positives from substring matching, e.g. "corn"
# inside "corned"): match whole tokens only, not substrings.
# -----------------------------------------------------------------
CATEGORY_RULES = [
    ("Chicken", {"chicken"}),
    ("Meat", {"beef", "veal", "lamb", "mutton", "kofta", "kibbeh"}),
    ("Seafood", {"fish", "shrimp", "shrimps", "prawn", "prawns", "seafood", "dori", "tilapia", "salmon"}),
    ("Veg", {"vegetable", "vegetables", "veg", "peas", "corn", "spinach"}),
    ("Potato", {"potato", "potatoes", "fries", "wedges"}),
    ("Fruits", {"fruit", "fruits", "berry", "berries", "strawberry", "mango"}),
    ("Dough", {"paratha", "tortilla", "flour", "dough", "samosa", "wrap", "wraps"}),
]
CAT2_MAP = {"Chicken":"Chicken","Chicken-Mince":"Chicken","Chicken-Commodity":"Chicken","Chicken-ZNG":"Chicken",
            "Meat":"Meat","Meat-Commodity":"Meat","Seafood":"Seafood","Seafood-ZING":"Seafood","Seafood-Commodity":"Seafood",
            "Fruits":"Fruits","Veg":"Veg","Potato":"Potato","Potato-Commodity":"Potato","Potato-Spl":"Potato","Dough":"Dough"}
# phrase rules checked separately since "spring roll" is two tokens
PHRASE_RULES = [("Dough", ["spring roll", "spring rolls"])]

def categorize(text):
    text_l = str(text).lower()
    for cat, phrases in PHRASE_RULES:
        if any(p in text_l for p in phrases):
            return cat
    tokens = set(re.findall(r"[a-z]+", text_l))
    for cat, kws in CATEGORY_RULES:
        if tokens & kws:
            return cat
    return "Other"

df["category"] = df["Product"].apply(categorize)
master["category"] = master["Item-Cat-2"].map(CAT2_MAP).fillna(master["Item-Cat-2"].apply(categorize))

def normalize_product_family(text):
    # Fix: .replace("-frozen","") only handled the hyphenated form. Checked
    # against this dump's real Product values -- "Breaded shrimp frozen"
    # uses a space (no hyphen), so it wasn't being stripped and would have
    # formed its own spurious competitor group. Word-boundary regex catches
    # any separator (hyphen, space, underscore, none).
    text = str(text).lower()
    text = re.sub(r"\bfrozen\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [PRODUCT_TOKEN_ALIASES.get(tok, tok) for tok in text.split()]
    return " ".join(tokens).strip()

# Fix: plain punctuation/suffix stripping alone does NOT merge singular vs
# plural spellings ("chicken nugget" vs "chicken nuggets" would still be
# two groups). Checked against this dump's actual Product field: no such
# duplicate pair currently exists there (it's a controlled taxonomy), so
# this alias table has zero effect on today's grouping -- it's here as
# defensive normalization for future data, deliberately kept to a small,
# verified list rather than generic stemming (which can wrongly merge
# unrelated families, e.g. "patty" vs "party").
PRODUCT_TOKEN_ALIASES = {
    "nugget": "nuggets", "burger": "burgers", "patty": "patties", "sausage": "sausages",
    "strip": "strips", "wing": "wings", "prawn": "prawns", "shrimp": "shrimps",
    "paratha": "parathas", "samosa": "samosas", "roll": "rolls",
}

df["product_family"] = df["Product"].apply(normalize_product_family)

df["clean_offer_text"] = df["Offer Name"].apply(clean_offer_text)
df["match_text"] = (
    df["clean_offer_text"] + " " +
    df["Product"].str.lower().str.replace("-frozen", "", regex=False) + " " +
    df["Variant"].str.lower().where(~df["Variant"].str.lower().isin(["no variant", ""]), "")
)
df["match_text"] = df["match_text"].str.replace(r"\s+", " ", regex=True).str.strip()
df["offer_measures_detailed"] = (df["Base Packsize"] + " " + df["Offer Name"]).apply(extract_flyer_measures)
df["offer_measures"] = df["offer_measures_detailed"].apply(collapse_to_simple)

master["match_text"] = master["Itemname"].str.lower() + " " + master["Item-Cat-4"].str.lower() + " " + master["Item Description"].str.lower()
master["match_text"] = master["match_text"].str.replace(r"\s+", " ", regex=True).str.strip()
master["master_measures_detailed"] = master["Item-Spec"].apply(extract_master_measures)
master["master_measures"] = master["master_measures_detailed"].apply(collapse_to_simple)

print(f"Stage 1 (cleaning/parsing): {time.time()-t_start:.1f}s")

# ===================================================================
# STAGE 2: SKU matching
# Fix (performance): batch-score with rapidfuzz.process.cdist instead
# of a python-level double loop, once per category (category filter
# is still a hard gate: chicken can never match beef/fish/veg/etc).
# Fix (Other-category danger): the "Other" bucket is where the
# category parser already failed, so require a much higher bar.
# Fix (unverified pack size): small score adjustment instead of
# treating "unknown" identically to "confirmed compatible".
# ===================================================================
t0 = time.time()
master_by_cat = {cat: grp.reset_index(drop=True) for cat, grp in master.groupby("category")}

def match_batch(sub_df, cat):
    """sub_df: unique own combos in this category. Returns list of result dicts, same order."""
    pool = master_by_cat.get(cat)
    if pool is None or pool.empty:
        if cat == "Other":
            # flyer-side categorizer couldn't identify a protein/category — fall back
            # to searching the entire master, but match_batch applies a much stricter
            # score/margin bar for cat == "Other" below.
            pool = master.reset_index(drop=True)
        else:
            # a real category (Chicken/Meat/etc.) with literally no master rows in it
            return [dict(matched_itemcode="NO_MATCH", matched_itemname="None", match_score=0.0, margin=0.0,
                         raw_margin=0.0, pack_status=None, confidence_tier="no_match_category",
                         master_match_text="", master_measures=[])] * len(sub_df)

    queries = sub_df["match_text"].tolist()
    choices = pool["match_text"].tolist()
    score_matrix = process.cdist(queries, choices, scorer=fuzz.token_sort_ratio) + \
                   process.cdist(queries, choices, scorer=fuzz.token_set_ratio)
    score_matrix = score_matrix / 2.0

    results = []
    is_other = (cat == "Other")
    for i, row in enumerate(sub_df.itertuples()):
        # Fix (#14): reject near-empty text outright instead of letting it
        # produce an arbitrary top match against noise.
        if len(row.match_text.split()) < 2:
            results.append(dict(matched_itemcode="NO_MATCH", matched_itemname="None", match_score=0.0,
                                 margin=0.0, raw_margin=0.0, pack_status=None, confidence_tier="no_match_empty_text",
                                 master_match_text="", master_measures=[]))
            continue

        row_scores = score_matrix[i].copy()
        pack_flags = np.array([pack_is_compatible(row.offer_measures, mm) for mm in pool["master_measures"]], dtype=object)
        adj_scores = row_scores + np.where(pack_flags == True, 4, np.where(pack_flags == None, -3, 0)).astype(float)
        # known-incompatible pack sizes are excluded outright unless nothing else is available
        known_incompatible = (pack_flags == False)
        all_incompatible = known_incompatible.all()
        if (~known_incompatible).any():
            adj_scores = np.where(known_incompatible, -1e9, adj_scores)

        # Fix: compute margin only among ELIGIBLE candidates (not the -1e9
        # excluded ones). Sorting all of adj_scores and taking order[1] could
        # land on an excluded candidate when only one real option existed,
        # producing an absurd ~1e9 margin -- confirmed this in real output
        # (two rows) before this fix, both harmless by luck since their text
        # score was already below the no-match floor, but a real risk in
        # general since it would trivially satisfy any margin>=N check.
        eligible = np.flatnonzero(adj_scores > -1e8)
        eligible_order = eligible[np.argsort(-adj_scores[eligible])]
        best_i = eligible_order[0]
        # Fix (#6): a category with only one ELIGIBLE candidate has no real
        # rival to be ambiguous against -- forcing margin to 0 in that case
        # would permanently block "high" confidence for a completely
        # unambiguous match. Use a large synthetic margin instead; the
        # score/pack checks still have to pass on their own merits.
        if len(eligible_order) > 1:
            second_i = eligible_order[1]
            margin = adj_scores[best_i] - adj_scores[second_i]
            raw_margin = row_scores[best_i] - row_scores[second_i]
        else:
            margin = 100.0
            raw_margin = 100.0
        best_score = row_scores[best_i]
        # Fix (#1 continued): margin is computed on adj_scores (which drove
        # the ranking) so a pack-driven re-ranking can't produce a
        # nonsensical negative margin; raw_margin (unadjusted) stays as a
        # diagnostic only.
        pack_status = pack_flags[best_i]
        best_row = pool.iloc[best_i]
        struct_ok = pack_structure_agrees(row.offer_measures_detailed, best_row["master_measures_detailed"])

        min_score = 85 if is_other else 55
        min_margin = 12 if is_other else 0

        if best_score < min_score or (is_other and margin < min_margin):
            results.append(dict(matched_itemcode="NO_MATCH", matched_itemname="None", match_score=round(float(best_score),2),
                                 margin=round(float(margin),2), raw_margin=round(float(raw_margin),2),
                                 pack_status=pack_status, confidence_tier="no_match",
                                 master_match_text="", master_measures=[]))
            continue

        # Fix (#2): "every candidate had an incompatible pack size" is a
        # materially different failure mode than an ordinary low-confidence
        # text match -- label it distinctly so it can be filtered separately.
        # Fix (#2 continued): a structural conflict (same total weight, but
        # different unit_count -- e.g. 2x500g vs a plain 1x1000g) was only
        # blocked from "high" before; it could still slip into "medium" and
        # get auto-accepted since pack_is_compatible (total-only) still says
        # True. Now blocked from both.
        if all_incompatible:
            conf = "low_pack_conflict"
        elif struct_ok is False:
            conf = "low_structure_conflict"
        elif pack_status is False:
            conf = "low"
        elif best_score >= 80 and margin >= 8 and pack_status is True:
            conf = "high"
        elif best_score >= 65 and (pack_status is True or margin >= 6):
            conf = "medium"
        else:
            conf = "low"

        results.append(dict(matched_itemcode=best_row["Itemcode"], matched_itemname=best_row["Itemname"],
                             match_score=round(float(best_score),2), margin=round(float(margin),2),
                             raw_margin=round(float(raw_margin),2),
                             pack_status=pack_status, confidence_tier=conf,
                             master_match_text=best_row["match_text"], master_measures=best_row["master_measures"]))
    return results

df["measures_key"] = df["offer_measures"].apply(tuple)
own_keys = df.loc[df["is_own"], ["match_text","offer_measures","offer_measures_detailed","measures_key","category"]].drop_duplicates(
    subset=["match_text","measures_key","category"]).reset_index(drop=True)

all_results = []
for cat, grp in own_keys.groupby("category"):
    all_results.append((grp.index, match_batch(grp.reset_index(drop=True), cat)))

res_series = pd.Series([None] * len(own_keys))
for idx, results in all_results:
    for pos, orig_idx in enumerate(idx):
        res_series.iloc[orig_idx] = results[pos]
res_df = pd.DataFrame(list(res_series))
own_keys = pd.concat([own_keys, res_df], axis=1)

print(f"Stage 2 (SKU matching, {len(own_keys):,} unique combos): {time.time()-t0:.1f}s")
print(own_keys["confidence_tier"].value_counts())

# -----------------------------------------------------------------
# Fix (low-confidence rows should not silently carry a real Itemcode):
# split into an always-populated "suggestion" and a gated
# "auto-accepted mapping" that is blank unless confidence is high/medium.
# -----------------------------------------------------------------
own_keys["suggested_itemcode"] = own_keys["matched_itemcode"]
own_keys["suggested_itemname"] = own_keys["matched_itemname"]
# NOTE: The AUTO_ACCEPT_TIERS gate and the merge of own_keys back into df
# are applied in Stage 2.5 (after LLM revision), so we do NOT merge here.
# Stage 2.5 will re-evaluate confidence tiers, then do the definitive merge.

print(f"After stage 2, elapsed: {time.time()-t_start:.1f}s")

# ===================================================================
# STAGE 2.5: Batched LLM revision of medium / low confidence mappings
#
# IMPORTANT PERFORMANCE FIX:
# The old version made one Ollama request per row. At ~14 seconds per row,
# 3,784 rows would take around 15 hours. This version sends 50 comparisons
# in one prompt, asks for compact "id:YES/NO" output, and checkpoints after
# every batch so interrupted runs can resume.
# ===================================================================

def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _build_llm_batch_prompt(batch_records) -> str:
    lines = [
        "You verify frozen-food flyer offers against suggested Al Kabeer master SKUs.",
        "For every numbered record, decide whether both descriptions identify the SAME retail SKU.",
        "",
        "Use YES only when product type, protein, flavour/variant, preparation style and retail pack size agree.",
        "Use NO when any important attribute conflicts.",
        "Ignore brand spelling, punctuation, word order and promotional words.",
        "When evidence is insufficient, use NO rather than guessing.",
        "",
        "Return exactly one result per record in this compact format:",
        "id:YES",
        "id:NO",
        "Do not add explanations, headings, markdown or extra text.",
        "",
        "RECORDS:"
    ]

    for rec in batch_records:
        lines.extend([
            f"{rec['id']}:",
            f"offer={rec['offer_name']}",
            f"offer_product={rec['product']}",
            f"offer_variant={rec['variant']}",
            f"offer_pack={rec['packsize']}",
            f"sku={rec['suggested_itemname']}",
            f"sku_description={rec['item_description']}",
            ""
        ])

    return "\n".join(lines)


def _call_ollama_batch(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_ctx": 4096,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT * 4)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _parse_batch_verdicts(raw_response: str, valid_ids) -> dict:
    valid_ids = {int(x) for x in valid_ids}
    parsed = {}

    # Accept compact forms such as 12:YES, 12 - YES, or "12": "YES".
    for match in re.finditer(
        r'(?m)^\s*["\']?(\d+)["\']?\s*[:=\-]\s*["\']?(YES|NO)["\']?\s*[,;]?\s*$',
        raw_response,
        flags=re.IGNORECASE,
    ):
        row_id = int(match.group(1))
        if row_id in valid_ids:
            parsed[row_id] = match.group(2).upper()

    # Fallback for responses that put several pairs on one line.
    if len(parsed) < len(valid_ids):
        for match in re.finditer(
            r'["\']?(\d+)["\']?\s*[:=\-]\s*["\']?(YES|NO)["\']?',
            raw_response,
            flags=re.IGNORECASE,
        ):
            row_id = int(match.group(1))
            if row_id in valid_ids:
                parsed[row_id] = match.group(2).upper()

    return parsed


def _ollama_is_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _checkpoint_key(row) -> str:
    return "||".join([
        _safe_text(row.get("match_text", "")),
        _safe_text(row.get("measures_key", "")),
        _safe_text(row.get("category", "")),
        _safe_text(row.get("suggested_itemcode", "")),
    ])


t0 = time.time()
print("\n=== STAGE 2.5: BATCHED LLM revision via Ollama ===")

if not _ollama_is_available():
    print("WARNING: Ollama not reachable at localhost:11434 — skipping LLM revision stage.")
    own_keys["llm_verdict"] = ""
else:
    _item_desc_lookup = (
        master.drop_duplicates(subset=["Itemcode"])
        .set_index("Itemcode")["Item Description"]
        .to_dict()
    )

    # Recover the original flyer fields for each unique matching combination.
    _representative = (
        df[df["is_own"]]
        .drop_duplicates(subset=["match_text", "measures_key", "category"])
        [["match_text", "measures_key", "category",
          "Offer Name", "Product", "Variant", "Base Packsize"]]
    )
    own_keys = own_keys.merge(
        _representative,
        on=["match_text", "measures_key", "category"],
        how="left",
    )

    review_mask = (
        own_keys["confidence_tier"].isin(LLM_REVIEW_TIERS)
        & own_keys["suggested_itemcode"].notna()
        & ~own_keys["suggested_itemcode"].isin(["NO_MATCH", "REVIEW_REQUIRED"])
    )
    review_idx = own_keys[review_mask].index.tolist()

    own_keys["llm_key"] = own_keys.apply(_checkpoint_key, axis=1)
    own_keys["llm_verdict"] = ""

    # Resume successful decisions from a prior interrupted run.
    completed = {}
    try:
        checkpoint = pd.read_csv(LLM_CHECKPOINT_CSV, dtype=str).fillna("")
        completed = dict(zip(checkpoint["llm_key"], checkpoint["llm_verdict"]))
        print(f"Loaded {len(completed):,} prior LLM decisions from checkpoint.")
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"WARNING: Could not load checkpoint: {exc}")

    for idx in review_idx:
        old = completed.get(own_keys.at[idx, "llm_key"], "")
        if old in {"YES", "NO"}:
            own_keys.at[idx, "llm_verdict"] = old

    pending_idx = [
        idx for idx in review_idx
        if own_keys.at[idx, "llm_verdict"] not in {"YES", "NO"}
    ]

    print(
        f"Rows eligible for LLM review: {len(review_idx):,}; "
        f"already completed: {len(review_idx) - len(pending_idx):,}; "
        f"remaining: {len(pending_idx):,}; batch size: {OLLAMA_BATCH_SIZE}"
    )

    total_batches = (len(pending_idx) + OLLAMA_BATCH_SIZE - 1) // OLLAMA_BATCH_SIZE

    for batch_no, batch_start in enumerate(
        range(0, len(pending_idx), OLLAMA_BATCH_SIZE), start=1
    ):
        batch_indices = pending_idx[batch_start:batch_start + OLLAMA_BATCH_SIZE]
        records = []

        for local_id, idx in enumerate(batch_indices):
            row = own_keys.loc[idx]
            records.append({
                "id": local_id,
                "offer_name": _safe_text(row.get("Offer Name", row.get("match_text", ""))),
                "product": _safe_text(row.get("Product", row.get("category", ""))),
                "variant": _safe_text(row.get("Variant", "")),
                "packsize": _safe_text(row.get("Base Packsize", row.get("measures_key", ""))),
                "suggested_itemname": _safe_text(row.get("suggested_itemname", "")),
                "item_description": _safe_text(
                    _item_desc_lookup.get(str(row.get("suggested_itemcode", "")), "")
                ),
            })

        prompt = _build_llm_batch_prompt(records)
        raw = ""
        parsed = {}

        for attempt in range(1, OLLAMA_RETRIES + 2):
            try:
                raw = _call_ollama_batch(prompt)
                parsed = _parse_batch_verdicts(
                    raw, [r["id"] for r in records]
                )
                if len(parsed) == len(records):
                    break
                print(
                    f"  Batch {batch_no}/{total_batches}: parsed "
                    f"{len(parsed)}/{len(records)} results on attempt {attempt}."
                )
            except requests.exceptions.Timeout:
                print(f"  Batch {batch_no}/{total_batches}: timeout on attempt {attempt}.")
            except Exception as exc:
                print(f"  Batch {batch_no}/{total_batches}: error on attempt {attempt}: {exc}")

        # Apply parsed responses. Missing/malformed answers remain pending and
        # will be retried automatically the next time the script is run.
        for local_id, verdict in parsed.items():
            idx = batch_indices[local_id]
            own_keys.at[idx, "llm_verdict"] = verdict

        # Save all completed results after every batch.
        checkpoint_out = own_keys.loc[
            own_keys["llm_verdict"].isin(["YES", "NO"]),
            ["llm_key", "llm_verdict"],
        ].drop_duplicates(subset=["llm_key"])
        checkpoint_out.to_csv(LLM_CHECKPOINT_CSV, index=False)

        done_now = own_keys.loc[review_idx, "llm_verdict"].isin(["YES", "NO"]).sum()
        elapsed = time.time() - t0
        rate = done_now / elapsed if elapsed > 0 else 0
        remaining = len(review_idx) - done_now
        eta_minutes = (remaining / rate / 60) if rate > 0 else float("inf")

        print(
            f"  Batch {batch_no}/{total_batches}: "
            f"{done_now}/{len(review_idx)} rows completed; "
            f"{elapsed:.1f}s elapsed; approx. {eta_minutes:.1f} min remaining"
        )

    # Update revised tiers after all available batch responses.
    yes_mask = own_keys["llm_verdict"].eq("YES")
    no_mask = own_keys["llm_verdict"].eq("NO")
    own_keys.loc[yes_mask, "confidence_tier"] = LLM_HIGH_REVISED
    own_keys.loc[no_mask, "confidence_tier"] = LLM_LOW_REVISED

    auto_ok_revised = own_keys["confidence_tier"].isin(AUTO_ACCEPT_TIERS)
    own_keys["matched_itemcode"] = np.where(
        auto_ok_revised, own_keys["suggested_itemcode"], "REVIEW_REQUIRED"
    )
    own_keys["matched_itemname"] = np.where(
        auto_ok_revised, own_keys["suggested_itemname"], "REVIEW_REQUIRED"
    )

    revised_counts = own_keys.loc[review_mask, "confidence_tier"].value_counts()
    print(f"Stage 2.5 revision summary:\n{revised_counts.to_string()}")
    print(f"Stage 2.5 elapsed: {time.time()-t0:.1f}s")

    own_keys.drop(
        columns=["llm_key", "Offer Name", "Product", "Variant", "Base Packsize"],
        inplace=True,
        errors="ignore",
    )

# Re-merge own_keys back into df after tier + matched_itemcode updates
# (drop the old match columns from df first to avoid _x/_y suffixes)
_merge_cols = ["matched_itemcode", "matched_itemname", "suggested_itemcode", "suggested_itemname",
               "match_score", "margin", "raw_margin", "pack_status", "confidence_tier",
               "master_match_text", "master_measures", "llm_verdict"]
# Only drop columns that already exist in df (llm_verdict won't be there yet)
df.drop(columns=[c for c in _merge_cols if c in df.columns], inplace=True)

# Add llm_verdict to own_keys if it wasn't created (Ollama unavailable path)
if "llm_verdict" not in own_keys.columns:
    own_keys["llm_verdict"] = ""

df = df.merge(
    own_keys.drop(columns=["offer_measures", "offer_measures_detailed"]),
    on=["match_text", "measures_key", "category"], how="left"
)
not_own_cols_revised = ["matched_itemcode", "matched_itemname", "suggested_itemcode", "suggested_itemname",
                        "match_score", "margin", "raw_margin", "pack_status", "confidence_tier",
                        "master_match_text", "master_measures", "llm_verdict"]
df.loc[~df["is_own"], not_own_cols_revised] = [
    "N/A - competitor brand row", "N/A", "N/A", "N/A",
    np.nan, np.nan, np.nan, None, "n/a", "", None, ""
]

print(f"After stage 2.5, elapsed: {time.time()-t_start:.1f}s")

# ===================================================================
# STAGE 3: SKU-level competitor discovery
# Fixes applied:
#  - target profile = the MATCHED MASTER SKU's text (Itemname + Item-Cat-4
#    + Description -- deliberately NOT Item-Spec, since measurement tokens
#    are already parsed and compared separately via pack_is_compatible;
#    mixing raw spec text back in would distort the fuzzy text score),
#    scored with each specific flyer offer's OWN pack size (not the
#    master's, which may be a generic entry -- see note above).
#  - scope = Country + product_family (normalized "Chicken Nuggets-Frozen"
#    -> "chicken nuggets"; also passes known plural aliases, see
#    PRODUCT_TOKEN_ALIASES above).
#  - competitors must clear a RAW text-score floor (>=60) AND an
#    adjusted-score floor (>=65), and a known pack-size MISMATCH is
#    excluded outright. A pack size that's simply unverifiable (neither
#    side parsed a weight) is NOT the same as a mismatch, so it is kept
#    but explicitly labeled "Pack unverified" rather than silently mixed
#    in with confirmed "Direct competitor" matches -- an earlier version
#    of this comment claimed unverified rows were excluded; they weren't,
#    which was the actual bug.
#  - batched with rapidfuzz.process.cdist instead of a per-row python loop.
# Only runs for rows that were auto-accepted (high/medium) -- a SKU that
# isn't itself confidently identified has no reliable profile to find
# competitors against.
# ===================================================================
t0 = time.time()
RAW_FLOOR, ADJ_FLOOR = 60, 65

df["measures_key"] = df["offer_measures"].apply(tuple)  # recompute after merge changed row order/columns
candidates = df[df["is_own"] & df["confidence_tier"].isin(AUTO_ACCEPT_TIERS)].copy()
# NOTE: dedup key includes measures_key (the flyer row's OWN pack size), not just
# matched_itemcode, because one master SKU can be sold/promoted at several real
# pack sizes (270g / 500g / 750g+250g) and those must not collapse into one
# generic competitor list -- that was exactly the original bug being fixed.
unique_targets = candidates.drop_duplicates(
    subset=["Country", "product_family", "matched_itemcode", "measures_key"]
)[["Country", "product_family", "Product", "matched_itemcode", "matched_itemname", "master_match_text",
   "offer_measures", "offer_measures_detailed", "measures_key", "Base Packsize"]].reset_index(drop=True)
print(f"Unique (country, family, own-SKU, pack) competitor targets: {len(unique_targets):,}")

comp_source = df[~df["is_own"]].copy()
comp_source = comp_source.drop_duplicates(subset=["Country", "product_family", "Brand Name", "clean_offer_text", "measures_key"])
comp_groups = {k: v.reset_index(drop=True) for k, v in comp_source.groupby(["Country", "product_family"])}

def format_retail_unit(detailed_measures):
    """Fix (dashboard clarity): render the comparable per-unit retail size
    explicitly (e.g. '270g') rather than leaving the reader to infer it from
    raw pack text like '2x270gm' vs '750 gm' -- those can look directly
    price-comparable when they aren't."""
    units = [(v, d) for v, d, c in detailed_measures if c == 1]
    if not units:
        return ""
    v, d = sorted(units)[0]
    return f"{v:.0f}{'g' if d == 'weight' else 'ml'}"

def find_competitors_batch(sub_targets, country, family):
    pool = comp_groups.get((country, family))
    if pool is None or pool.empty:
        return [[] for _ in range(len(sub_targets))]

    queries = sub_targets["master_match_text"].tolist()
    choices = pool["match_text"].tolist()
    score_matrix = (process.cdist(queries, choices, scorer=fuzz.token_sort_ratio) +
                    process.cdist(queries, choices, scorer=fuzz.token_set_ratio)) / 2.0

    out = []
    for i, target_measures in enumerate(sub_targets["offer_measures"]):
        pack_flags = np.array([pack_is_compatible(target_measures, cm) for cm in pool["offer_measures"]], dtype=object)
        raw = score_matrix[i]
        adj = raw + np.where(pack_flags == True, 3, np.where(pack_flags == None, -3, 0))
        scored = pool.assign(_score=adj, _raw=raw, _pack=pack_flags)
        # Fix (#3): a known MISMATCH is excluded outright; "unverifiable"
        # (neither side had a parseable weight) is kept but labeled
        # separately below, rather than silently passing as if confirmed.
        scored = scored[(scored["_raw"] >= RAW_FLOOR) & (scored["_score"] >= ADJ_FLOOR) & (scored["_pack"] != False)]
        if scored.empty:
            out.append([])
            continue
        best_per_brand = scored.sort_values("_score", ascending=False).drop_duplicates(subset=["Brand Name"])
        top = best_per_brand.sort_values("_score", ascending=False).head(5)
        out.append([(r["Brand Name"], r["Offer Name"], r["Base Packsize"], float(r["_raw"]), float(r["_score"]), r["_pack"],
                     "Direct competitor" if r["_pack"] == True else "Pack unverified", r["offer_measures_detailed"])
                    for _, r in top.iterrows()])
    return out

COMPETITOR_LONG_COLUMNS = ["Country", "Product", "Own Itemcode", "Own Itemname", "Own Pack", "Own Retail Unit",
                           "Competitor Brand", "Competitor Offer", "Competitor Pack", "Competitor Retail Unit",
                           "Raw Score", "Adjusted Score", "Pack Status", "Competitor Type", "Rank"]

results = []
long_rows = []
for (country, family), grp in unique_targets.groupby(["Country", "product_family"]):
    grp = grp.reset_index(drop=True)
    comps = find_competitors_batch(grp, country, family)
    for i in range(len(grp)):
        summary = "; ".join(f"{b}: {o} [{raw:.0f}]" + ("" if ctype == "Direct competitor" else " (pack unverified)")
                             for b, o, p, raw, adj, ps, ctype, cmd in comps[i]) or "None Found"
        results.append({"Country": grp.loc[i,"Country"], "product_family": grp.loc[i,"product_family"],
                         "matched_itemcode": grp.loc[i,"matched_itemcode"], "measures_key": grp.loc[i,"measures_key"],
                         "competitor_matches": summary})
        own_retail_unit = format_retail_unit(grp.loc[i,"offer_measures_detailed"])
        for rank, (b, o, p, raw, adj, ps, ctype, cmd) in enumerate(comps[i], start=1):
            long_rows.append({
                "Country": grp.loc[i,"Country"], "Product": grp.loc[i,"Product"],
                "Own Itemcode": grp.loc[i,"matched_itemcode"], "Own Itemname": grp.loc[i,"matched_itemname"],
                "Own Pack": grp.loc[i,"Base Packsize"], "Own Retail Unit": own_retail_unit,
                "Competitor Brand": b, "Competitor Offer": o,
                "Competitor Pack": p, "Competitor Retail Unit": format_retail_unit(cmd),
                "Raw Score": round(raw,1), "Adjusted Score": round(adj,1),
                "Pack Status": ps, "Competitor Type": ctype, "Rank": rank,
            })
COMP_RESULT_COLUMNS = ["Country", "product_family", "matched_itemcode", "measures_key", "competitor_matches"]
comp_results = pd.DataFrame(results, columns=COMP_RESULT_COLUMNS)
print(f"Stage 3 (competitor discovery): {time.time()-t0:.1f}s")

# Fix (empty-result schema): pass explicit columns so the CSV always has a
# defined header even if long_rows happens to be empty.
long_format = pd.DataFrame(long_rows, columns=COMPETITOR_LONG_COLUMNS)
long_format.to_csv(COMPETITOR_LONG_CSV, index=False)
print(f"Long-format competitor table written: {len(long_format):,} rows")

df = df.merge(comp_results, on=["Country","product_family","matched_itemcode","measures_key"], how="left")
df["competitor_matches"] = df["competitor_matches"].fillna("Not computed (row not auto-accepted, or is a competitor-brand row)")

# ===================================================================
# STAGE 4: export
# ===================================================================
out_cols = ["Country","Retailer Name","Flyer Name","Analysis week","offerid","Offer Name","Brand Name",
            "segment","Product","Variant","Base Packsize","Offer Price","Regular Price","Discount_percent",
            "matched_itemcode","matched_itemname","suggested_itemcode","suggested_itemname",
            "match_score","margin","raw_margin","pack_status","confidence_tier","llm_verdict","competitor_matches"]
final = df[out_cols]
final.to_csv(OUT_CSV, index=False)
print(f"DONE. rows: {len(final):,}  total time: {time.time()-t_start:.1f}s")

# ===================================================================
# Fix (no real accuracy measurement, only confidence-tier counts):
# export a stratified random sample for manual labeling so accuracy can
# actually be measured per tier, instead of just trusting the tiers.
# Fix (#12): match_batch can emit tier variants (no_match_category,
# no_match_empty_text, low_pack_conflict) beyond the four base tiers --
# fold them into their parent tier for sampling so none are silently
# skipped.
# ===================================================================
own_rows = df[df["is_own"]].copy()
own_rows["validation_tier"] = own_rows["confidence_tier"].replace({
    "no_match_category": "no_match", "no_match_empty_text": "no_match",
    "low_pack_conflict": "low", "low_structure_conflict": "low",
    "high (revised)": "high", "low (revised)": "low",
})
sample_parts = []
for tier, n in [("high", 40), ("medium", 60), ("low", 40), ("no_match", 20)]:
    pool = own_rows[own_rows["validation_tier"] == tier].drop_duplicates(subset=["Offer Name","Base Packsize"])
    if not pool.empty:
        sample_parts.append(pool.sample(min(n, len(pool)), random_state=42))

# Fix (#13): don't let an empty sample_parts list produce a malformed concat.
if sample_parts:
    sample = pd.concat(sample_parts, ignore_index=True)[
        ["Offer Name","Base Packsize","Product","confidence_tier",
         "matched_itemcode","matched_itemname","suggested_itemcode","suggested_itemname",
         "match_score","margin"]].copy()
else:
    sample = pd.DataFrame(columns=["Offer Name","Base Packsize","Product","confidence_tier",
                                    "matched_itemcode","matched_itemname","suggested_itemcode","suggested_itemname",
                                    "match_score","margin"])
sample["is_correct_match (fill in Y/N)"] = ""
sample["correct_itemcode_if_wrong"] = ""
sample.to_csv(VALIDATION_SAMPLE_CSV, index=False)
print(f"Validation sample written: {len(sample)} rows across tiers")

# ===================================================================
# STAGE 6: Complete manual mapping review queue
# ===================================================================
# Keep the full sku_mapping_output.csv unchanged.
# This separate file contains one row per UNIQUE mapping combination
# where the matcher found a candidate worth reviewing.
#
# Included:
#   - high
#   - medium
#   - low
#   - low_pack_conflict
#   - low_structure_conflict
#
# Excluded:
#   - no_match rows
#   - competitor-brand rows
#
# The existing manual_validation_sample.csv remains a random stratified
# sample for measuring accuracy. This file is the complete review queue.
# ===================================================================

review_rows = df[
    df["is_own"]
    & df["confidence_tier"].isin([
        "high",
        "high (revised)",
        "medium",
        "low",
        "low (revised)",
        "low_pack_conflict",
        "low_structure_conflict",
    ])
].copy()

review_rows = review_rows.drop_duplicates(
    subset=["match_text", "measures_key", "category"]
).copy()

# Bring the suggested master SKU's Item Description into the review file.
# We use suggested_itemcode rather than matched_itemcode because medium/low
# rows contain REVIEW_REQUIRED in matched_itemcode but still have a real
# suggested master SKU.
item_description_by_code = (
    master.drop_duplicates(subset=["Itemcode"])
    .set_index("Itemcode")["Item Description"]
    .to_dict()
)
review_rows["Item Description"] = (
    review_rows["suggested_itemcode"]
    .astype(str)
    .map(item_description_by_code)
    .fillna("")
)

manual_review_columns = [
    "Offer Name",
    "Product",
    "Variant",
    "Base Packsize",
    "matched_itemcode",
    "matched_itemname",
    "suggested_itemcode",
    "suggested_itemname",
    "Item Description",
    "match_score",
    "margin",
    "raw_margin",
    "pack_status",
    "confidence_tier",
    "llm_verdict",
]

manual_review = review_rows[manual_review_columns].copy()

confidence_order = {
    "high": 1,
    "high (revised)": 2,
    "medium": 3,
    "low": 4,
    "low (revised)": 5,
    "low_structure_conflict": 6,
    "low_pack_conflict": 7,
}

manual_review["confidence_order"] = (
    manual_review["confidence_tier"]
    .map(confidence_order)
    .fillna(99)
)

manual_review = manual_review.sort_values(
    by=["confidence_order", "match_score", "margin"],
    ascending=[True, False, False],
).drop(columns=["confidence_order"])

manual_review["is_correct_match (fill in Y/N)"] = ""
manual_review["correct_itemcode_if_wrong"] = ""
manual_review["review_notes"] = ""

manual_review.to_csv(MANUAL_REVIEW_CSV, index=False)
print("Manual review output columns:", manual_review.columns.tolist())

print(
    f"Manual mapping review written: "
    f"{len(manual_review):,} unique candidate mappings"
)
print("Manual review rows by confidence:")
print(manual_review["confidence_tier"].value_counts())