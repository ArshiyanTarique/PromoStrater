import pandas as pd
import numpy as np
import re
import time
import json
import requests
import joblib
from pathlib import Path
from rapidfuzz import fuzz, process

FLYER_PATH = "Alkabeer_Export_Data_Clickflyer.csv"
MASTER_PATH = "Product_Master.xlsx"

FINAL_CLICKFLYER_CSV = "outputs/FINAL_master_sku_clickflyer_offers.csv"
FINAL_COMPETITOR_CSV = "outputs/FINAL_alkabeer_competitor_offers.csv"

# -----------------------------------------------------------------
# Trained LightGBM SKU model settings
# -----------------------------------------------------------------
MODEL_PATH = Path("models/alkabeer_sku_matcher_v1.joblib")
ML_AUTO_MATCH_THRESHOLD = 0.95
ML_MANUAL_REVIEW_THRESHOLD = 0.70

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained ML model not found: {MODEL_PATH.resolve()}\n"
        "Place alkabeer_sku_matcher_v1.joblib inside the models folder."
    )

_model_package = joblib.load(MODEL_PATH)
sku_model = _model_package["model"]
MODEL_FEATURE_COLUMNS = _model_package["feature_columns"]
ML_AUTO_MATCH_THRESHOLD = _model_package.get(
    "auto_match_threshold", ML_AUTO_MATCH_THRESHOLD
)
ML_MANUAL_REVIEW_THRESHOLD = _model_package.get(
    "manual_review_threshold", ML_MANUAL_REVIEW_THRESHOLD
)
MODEL_VERSION = _model_package.get("model_version", "v1")

print(f"Loaded SKU ML model: {MODEL_PATH} ({MODEL_VERSION})")
print(f"ML feature count: {len(MODEL_FEATURE_COLUMNS)}")

# -----------------------------------------------------------------
# Ollama LLM revision settings (Stage 2.5)
# -----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60          # seconds per request
OLLAMA_RETRIES = 1           # retries on timeout/connection error
OLLAMA_BATCH_SIZE = 50       # review many rows in one model request
OLLAMA_NUM_PREDICT = 260     # enough for compact "id:YES/NO" answers
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
AUTO_ACCEPT_TIERS = ["high (ml)"]

print("RUNNING SKU MAPPING PIPELINE V3 - LIGHTGBM SKU REVIEW")
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
# STAGE 2.5: LightGBM review of the fuzzy match suggestion
# ===================================================================
# Stage 2 still generates the strongest candidate using the existing
# category, text and pack-size logic. The trained model now reviews that
# candidate and decides whether it can be auto-accepted, needs review, or
# should be treated as no match.
# ===================================================================

def _safe_number(value):
    """Convert a value to float while preserving missing values as NaN."""
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_weight_value(details):
    """Return the most likely consumer-facing weight in grams."""
    if not details:
        return np.nan
    retail = [float(v) for v, dim, count in details if dim == "weight" and count == 1]
    if retail:
        return min(retail)
    weights = [float(v) for v, dim, _ in details if dim == "weight"]
    return min(weights) if weights else np.nan


def _offer_unit_count(details):
    """Return the largest explicit flyer multipack count, otherwise 1."""
    if not details:
        return np.nan
    counts = [int(count) for _, _, count in details if int(count) > 1]
    return max(counts) if counts else 1.0


def _offer_total_weight(details):
    if not details:
        return np.nan
    weights = [float(v) for v, dim, _ in details if dim == "weight"]
    return max(weights) if weights else np.nan


def _extract_piece_count(text):
    t = str(text).lower()
    patterns = [
        r"(\d+)\s*(?:pcs?|pieces?)\b",
        r"\b(\d+)\s*['’]s\b",
        r"\bx\s*(\d+)\s*(?:pcs?|pieces?)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            return float(m.group(1))
    return np.nan


def _extract_bonus_weight(text):
    """Extract the second amount in a bonus expression such as 750g+250g."""
    t = str(text).lower().replace(" ", "")
    m = re.search(
        rf"\d+(?:\.\d+)?(?:{ALL_UNITS})\+(\d+(?:\.\d+)?)({ALL_UNITS})",
        t,
    )
    if not m:
        return np.nan
    amount, unit = m.groups()
    dim, factor = unit_dim_value(unit)
    return float(amount) * factor if dim == "weight" else np.nan


def _master_units_per_carton(spec):
    t = str(spec).lower().replace(",", "")
    patterns = [
        rf"\d+(?:\.\d+)?\s*(?:{ALL_UNITS})\s*[x×*]\s*(\d+(?:\.\d+)?)",
        rf"(\d+(?:\.\d+)?)\s*[x×*]\s*\d+(?:\.\d+)?\s*(?:{ALL_UNITS})",
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            return float(m.group(1))
    return np.nan


PROTEIN_WORDS = {
    "chicken", "beef", "veal", "lamb", "mutton", "fish", "shrimp",
    "shrimps", "prawn", "prawns", "turkey",
}
NON_MEAT_WORDS = {
    "cheese", "chocolate", "vegetable", "vegetables", "veg", "peas",
    "corn", "potato", "potatoes", "fries", "paratha", "spinach",
}
FAMILY_PHRASES = [
    "seekh kebab", "spring roll", "chicken fries", "fish fingers",
    "cheese sticks", "chicken strips", "chicken fillet", "chicken fillets",
    "chicken popcorn", "chicken nuggets", "beef burger", "chicken burger",
    "burger", "burgers", "nuggets", "fillet", "fillets", "strips",
    "popcorn", "kibbeh", "kofta", "samosa", "samosas", "paratha",
    "fries", "wedges", "patties", "sausages", "shrimps", "prawns",
]
VARIANT_WORDS = {
    "spicy", "non spicy", "non-spicy", "regular", "hot", "sriracha",
    "buffalo", "bbq", "barbecue", "onion", "breaded", "zing", "krazee",
    "jumbo", "mild",
}


def _protein_set(text):
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    return tokens & PROTEIN_WORDS


def _family_set(text):
    t = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    found = set()
    for phrase in FAMILY_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            found.add(phrase)
    return found


def _variant_set(text):
    t = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    found = set()
    for variant in VARIANT_WORDS:
        if re.search(rf"\b{re.escape(variant.replace('-', ' '))}\b", t):
            found.add(variant.replace("-", " "))
    return found


def _compatibility_flag(left_values, right_values, unspecified_is_match=True):
    if not left_values or not right_values:
        return 1 if unspecified_is_match else 0
    return int(bool(set(left_values) & set(right_values)))


def _expected_match_count(text):
    t = str(text).lower()
    separators = len(re.findall(r"\s(?:/|\bor\b|\band\b|\+)\s", t))
    return float(max(1, min(4, separators + 1)))


def _build_ml_feature_row(offer_row, master_row):
    offer_text = " ".join([
        str(offer_row.get("Offer Name", "")),
        str(offer_row.get("Product", "")),
        str(offer_row.get("Variant", "")),
    ]).strip()
    master_text = " ".join([
        str(master_row.get("Itemname", "")),
        str(master_row.get("Item-Cat-4", "")),
        str(master_row.get("Item Description", "")),
    ]).strip()

    offer_clean = clean_offer_text(offer_text)
    master_clean = clean_offer_text(master_text)

    offer_proteins = _protein_set(offer_text)
    master_proteins = _protein_set(master_text)
    offer_families = _family_set(offer_text)
    master_families = _family_set(master_text)
    offer_variants = _variant_set(offer_text)
    master_variants = _variant_set(master_text)

    offer_details = offer_row.get("offer_measures_detailed", [])
    master_details = master_row.get("master_measures_detailed", [])
    pack_ok = pack_is_compatible(
        collapse_to_simple(offer_details), collapse_to_simple(master_details)
    )
    structure_ok = pack_structure_agrees(offer_details, master_details)

    word_similarity = (
        fuzz.token_sort_ratio(offer_clean, master_clean)
        + fuzz.token_set_ratio(offer_clean, master_clean)
    ) / 2.0
    character_similarity = fuzz.ratio(offer_clean, master_clean)
    token_similarity = fuzz.WRatio(offer_clean, master_clean)

    mixed_protein = int(len(offer_proteins) > 1)
    multi_family = int(len(offer_families) > 1)
    contains_non_meat = int(bool(
        set(re.findall(r"[a-z]+", offer_text.lower())) & NON_MEAT_WORDS
    ))

    features = {
        "protein_match": _compatibility_flag(offer_proteins, master_proteins),
        "family_match": _compatibility_flag(offer_families, master_families),
        "variant_match": _compatibility_flag(offer_variants, master_variants),
        "size_match": int(pack_ok is True),
        "pack_format_match": int(structure_ok is not False),
        "word_similarity": round(float(word_similarity), 1),
        "character_similarity": round(float(character_similarity), 1),
        "token_similarity": round(float(token_similarity), 1),
        "unit_pack_weight_g": _first_weight_value(offer_details),
        "number_of_units": _offer_unit_count(offer_details),
        "bonus_weight_g": _extract_bonus_weight(offer_text),
        "total_offer_weight_g": _offer_total_weight(offer_details),
        "piece_count": _extract_piece_count(offer_text),
        "master_unit_weight_g": _first_weight_value(master_details),
        "master_units_per_carton": _master_units_per_carton(
            master_row.get("Item-Spec", "")
        ),
        "is_mixed_protein_offer": mixed_protein,
        "is_multi_family_offer": multi_family,
        "contains_non_meat_product": contains_non_meat,
        "expected_match_count": _expected_match_count(offer_text),
    }

    return features


def _predict_ml_probability(feature_row):
    feature_frame = pd.DataFrame([feature_row]).reindex(
        columns=MODEL_FEATURE_COLUMNS
    )
    for column in MODEL_FEATURE_COLUMNS:
        feature_frame[column] = pd.to_numeric(
            feature_frame[column], errors="coerce"
        )
    return float(sku_model.predict_proba(feature_frame)[0, 1])


print("\n=== STAGE 2.5: LIGHTGBM SKU REVIEW ===")
t0 = time.time()

# Recover one representative original flyer row for each unique matching key.
_representative = (
    df[df["is_own"]]
    .drop_duplicates(subset=["match_text", "measures_key", "category"])
    [[
        "match_text", "measures_key", "category", "Offer Name", "Product",
        "Variant", "Base Packsize", "offer_measures_detailed",
    ]]
)
own_keys = own_keys.merge(
    _representative,
    on=["match_text", "measures_key", "category"],
    how="left",
)

_master_lookup = (
    master.drop_duplicates(subset=["Itemcode"])
    .set_index("Itemcode")
)

ml_probabilities = []
ml_decisions = []
ml_feature_json = []

for _, offer_row in own_keys.iterrows():
    itemcode = str(offer_row.get("suggested_itemcode", "")).strip()

    if itemcode in {"", "NO_MATCH", "REVIEW_REQUIRED", "nan"}:
        ml_probabilities.append(np.nan)
        ml_decisions.append("NO_CANDIDATE")
        ml_feature_json.append("")
        continue

    if itemcode not in _master_lookup.index:
        ml_probabilities.append(np.nan)
        ml_decisions.append("MASTER_SKU_NOT_FOUND")
        ml_feature_json.append("")
        continue

    master_row = _master_lookup.loc[itemcode]
    if isinstance(master_row, pd.DataFrame):
        master_row = master_row.iloc[0]

    feature_row = _build_ml_feature_row(offer_row, master_row)
    probability = _predict_ml_probability(feature_row)

    if probability >= ML_AUTO_MATCH_THRESHOLD:
        decision = "AUTO_MATCH"
    elif probability >= ML_MANUAL_REVIEW_THRESHOLD:
        decision = "MANUAL_REVIEW"
    else:
        decision = "NO_MATCH"

    ml_probabilities.append(probability)
    ml_decisions.append(decision)
    ml_feature_json.append(json.dumps(feature_row, allow_nan=True))

own_keys["ml_probability"] = ml_probabilities
own_keys["ml_decision"] = ml_decisions
own_keys["ml_feature_json"] = ml_feature_json

# The ML decision is now the definitive auto-acceptance gate.
own_keys["confidence_tier"] = np.select(
    [
        own_keys["ml_decision"].eq("AUTO_MATCH"),
        own_keys["ml_decision"].eq("MANUAL_REVIEW"),
        own_keys["ml_decision"].eq("NO_MATCH"),
    ],
    ["high (ml)", "medium (ml)", "low (ml)"],
    default=own_keys["confidence_tier"],
)

_auto_ok = own_keys["ml_decision"].eq("AUTO_MATCH")
own_keys["matched_itemcode"] = np.where(
    _auto_ok, own_keys["suggested_itemcode"], "REVIEW_REQUIRED"
)
own_keys["matched_itemname"] = np.where(
    _auto_ok, own_keys["suggested_itemname"], "REVIEW_REQUIRED"
)

print("ML decision counts:")
print(own_keys["ml_decision"].value_counts(dropna=False))
print(f"Stage 2.5 elapsed: {time.time()-t0:.1f}s")

# Merge the reviewed mappings back into every original flyer row.
_merge_cols = [
    "matched_itemcode", "matched_itemname", "suggested_itemcode",
    "suggested_itemname", "match_score", "margin", "raw_margin",
    "pack_status", "confidence_tier", "master_match_text",
    "master_measures", "ml_probability", "ml_decision", "ml_feature_json",
]
df.drop(columns=[c for c in _merge_cols if c in df.columns], inplace=True)

df = df.merge(
    own_keys.drop(
        columns=[
            "offer_measures", "offer_measures_detailed", "Offer Name",
            "Product", "Variant", "Base Packsize",
        ],
        errors="ignore",
    ),
    on=["match_text", "measures_key", "category"],
    how="left",
)

not_own_cols_revised = [
    "matched_itemcode", "matched_itemname", "suggested_itemcode",
    "suggested_itemname", "match_score", "margin", "raw_margin",
    "pack_status", "confidence_tier", "master_match_text",
    "master_measures", "ml_probability", "ml_decision", "ml_feature_json",
]
df.loc[~df["is_own"], not_own_cols_revised] = [
    "N/A - competitor brand row", "N/A", "N/A", "N/A",
    np.nan, np.nan, np.nan, None, "n/a", "", None, np.nan, "N/A", "",
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
print(f"Long-format competitor table built in memory: {len(long_format):,} rows")

df = df.merge(comp_results, on=["Country","product_family","matched_itemcode","measures_key"], how="left")
df["competitor_matches"] = df["competitor_matches"].fillna("Not computed (row not auto-accepted, or is a competitor-brand row)")

# ===================================================================
# STAGE 4: export
# ===================================================================
out_cols = ["Country","Retailer Name","Flyer Name","Analysis week","offerid","Offer Name","Brand Name",
            "segment","Product","Variant","Base Packsize","Offer Price","Regular Price","Discount_percent",
            "matched_itemcode","matched_itemname","suggested_itemcode","suggested_itemname",
            "match_score","margin","raw_margin","pack_status","confidence_tier","ml_probability","ml_decision","competitor_matches"]
final = df[out_cols]
print(f"DONE. rows: {len(final):,}  total time: {time.time()-t_start:.1f}s")

# ===================================================================
# STAGE 5: build the two final deliverable CSVs (master SKU x offers)
# ===================================================================
_master = master[["Itemcode", "Itemname"]].copy()
_master["sku_label"] = _master["Itemcode"].astype(str) + " - " + _master["Itemname"].astype(str)


def _single_line_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def _fmt_offer(row):
    price = row["Offer Price"]
    price_str = f"SAR {price:g}" if pd.notna(price) else "N/A"
    offer_name = _single_line_text(row["Offer Name"])
    retailer_name = _single_line_text(row["Retailer Name"])
    return f"{offer_name} — {retailer_name} — {price_str}"


def _numbered_cell(series):
    """Return a numbered offer list that stays on one physical CSV line."""
    items = [_single_line_text(x) for x in series if x and str(x).strip()]
    if not items:
        return ""
    return " | ".join(f"{i}. {v}" for i, v in enumerate(items, 1))


def _validate_master_offer_export(export_df, expected_master_labels):
    expected_columns = [
        "Master SKU",
        "Confirmed Matched Offers",
        "Medium Confidence Offers",
        "Low Confidence Offers",
    ]
    if export_df.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected clickflyer export columns: {export_df.columns.tolist()}"
        )
    actual_labels = export_df["Master SKU"].astype(str).tolist()
    if actual_labels != expected_master_labels:
        raise ValueError(
            "Clickflyer export column A no longer matches Product Master order/content."
        )
    contains_newline = export_df.astype(str).apply(
        lambda col: col.str.contains(r"[\r\n]", regex=True).any()
    ).any()
    if contains_newline:
        raise ValueError(
            "Clickflyer export contains embedded line breaks; CSV rows would be unsafe."
        )


final_offers = final.copy()
final_offers["offer_text"] = final_offers.apply(_fmt_offer, axis=1)

confirmed_grp = (
    final_offers[final_offers["confidence_tier"] == "high (ml)"]
    .groupby("matched_itemcode")["offer_text"].apply(_numbered_cell)
)
medium_grp = (
    final_offers[final_offers["confidence_tier"] == "medium (ml)"]
    .groupby("suggested_itemcode")["offer_text"].apply(_numbered_cell)
)
low_grp = (
    final_offers[final_offers["confidence_tier"] == "low (ml)"]
    .groupby("suggested_itemcode")["offer_text"].apply(_numbered_cell)
)

file1 = _master[["sku_label", "Itemcode"]].copy()
file1["Confirmed Matched Offers"] = file1["Itemcode"].map(confirmed_grp).fillna("")
file1["Medium Confidence Offers"] = file1["Itemcode"].map(medium_grp).fillna("")
file1["Low Confidence Offers"] = file1["Itemcode"].map(low_grp).fillna("")
file1 = file1.rename(columns={"sku_label": "Master SKU"}).drop(columns=["Itemcode"])
_expected_master_labels = _master["sku_label"].astype(str).tolist()
_validate_master_offer_export(file1, _expected_master_labels)
file1.to_csv(FINAL_CLICKFLYER_CSV, index=False, encoding="utf-8-sig")
_written_file1 = pd.read_csv(FINAL_CLICKFLYER_CSV, dtype=str, keep_default_na=False)
_validate_master_offer_export(_written_file1, _expected_master_labels)
print(f"Final clickflyer-offers CSV written: {FINAL_CLICKFLYER_CSV} ({len(file1):,} rows)")

comp2 = long_format[long_format["Competitor Type"].isin(["Direct competitor", "Pack unverified"])].copy()
comp2 = comp2.sort_values(["Own Itemcode", "Rank"])
comp2["competitor_offer_text"] = comp2.apply(
    lambda r: f"{r['Competitor Offer']} ({r['Competitor Pack']})", axis=1
)

brand_grp = comp2.groupby("Own Itemcode")["Competitor Brand"].apply(_numbered_cell)
offer_grp = comp2.groupby("Own Itemcode")["competitor_offer_text"].apply(_numbered_cell)

file2 = _master[["sku_label", "Itemcode"]].copy()
file2["Competitor Brand Names"] = file2["Itemcode"].map(brand_grp).fillna("")
file2["Competitor Offers"] = file2["Itemcode"].map(offer_grp).fillna("")
file2 = file2.rename(columns={"sku_label": "Al Kabeer Master SKU"}).drop(columns=["Itemcode"])
file2.to_csv(FINAL_COMPETITOR_CSV, index=False, encoding="utf-8-sig")
print(f"Final competitor-offers CSV written: {FINAL_COMPETITOR_CSV} ({len(file2):,} rows)")
