"""GPU evaluation harness for a real embedding backend.

Paste each CELL block into its own notebook cell. Runs standalone on Colab, or
against this repo if you run it from the project root.

Why this exists: the configured backend is `local_hashing`, a HashingVectorizer
that its own docstring calls "for tests and dry runs". It has no semantic
content - measured on real product strings it scored a true match (0.1005)
LOWER than two unrelated products (0.1260). Before wiring a real model in, the
question worth answering is not "is it fast" but "does it retrieve the right
master SKU", so this measures recall@k against 1,392 human-labelled pairs and
compares every candidate model to the hashing baseline on the same data.
"""

# ============================================================ CELL 1: setup
# On Colab, uncomment:
# !pip -q install sentence-transformers pandas pyarrow openpyxl

import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def pick_device() -> str:
    """Prefer a real accelerator, in order of how well ST supports it."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"          # Intel Arc / discrete
    if torch.backends.mps.is_available():
        return "mps"          # Apple silicon
    return "cpu"


DEVICE = pick_device()
print(f"torch {torch.__version__} | device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  No CUDA device. Throughput numbers below will be CPU-bound and "
          "are NOT representative of a GPU deployment.")


# ====================================================== CELL 2: load real data
# Point these at the repo. On Colab, upload the two files instead.
MASTER_PATH = "Product_Master.xlsx"
PAIRS_PATH = "data/processed/training_features.parquet"

master_raw = pd.read_excel(MASTER_PATH)

# Column mapping copied from _build_candidate_features_for_slice in
# src/sku_mapping/shadow/pipeline.py, so master text matches what the pipeline
# would actually embed.
master = pd.DataFrame({
    "master_itemcode": master_raw["Itemcode"].astype(str),
    "master_brand": master_raw.get("Brand Name", "Al Kabeer"),
    "master_item_description": master_raw["Itemname"].astype(str),
    "master_item_family": master_raw["Item-Cat-4"].astype(str),
    "master_item_category": master_raw["Item-Cat-2"].astype(str),
    "master_item_long_description": master_raw["Item Description"].astype(str),
    "master_item_spec": master_raw["Item-Spec"].astype(str),
}).drop_duplicates("master_itemcode").reset_index(drop=True)

pairs = pd.read_parquet(PAIRS_PATH)
gold = (
    pairs[pairs["pair_label"] == 1][["offer_text", "master_itemcode"]]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .reset_index(drop=True)
)
# Only keep pairs whose gold SKU is actually in the master, or recall is
# unwinnable for reasons that have nothing to do with the model.
gold = gold[gold["master_itemcode"].isin(set(master["master_itemcode"]))]
gold = gold.reset_index(drop=True)

print(f"master SKUs:  {len(master):,}")
print(f"gold pairs:   {len(gold):,}  (offers with a known correct SKU)")
print(gold.head(3).to_string(index=False))


# ================================================ CELL 3: build embedding text
def master_text(row) -> str:
    """Mirror of prepare_candidate_embedding_text (embedding/text.py)."""
    parts = [
        f"brand={row.master_brand}",
        f"item={row.master_item_description}",
        f"family={row.master_item_family}",
        f"category={row.master_item_category}",
        f"description={row.master_item_long_description}",
        f"pack={row.master_item_spec}",
    ]
    return " | ".join(p for p in parts if not p.endswith("=nan"))


master_texts = [master_text(r) for r in master.itertuples()]
offer_texts = gold["offer_text"].tolist()
gold_codes = gold["master_itemcode"].to_numpy()
code_index = {c: i for i, c in enumerate(master["master_itemcode"])}
gold_positions = np.array([code_index[c] for c in gold_codes])

print(f"offers to encode: {len(offer_texts):,}")
print(f"masters to encode: {len(master_texts):,}")
print("\nexample master text:\n ", master_texts[0][:160])
print("example offer text:\n ", offer_texts[0][:160])
# NOTE: the pipeline builds a richer offer text (brand, entity, variant, pack,
# commercial attributes). Here only offer_text is available, so these recall
# numbers are a LOWER BOUND on what the pipeline would achieve.


# ============================================== CELL 4: metrics + baseline
def recall_at_k(sim: np.ndarray, gold_pos: np.ndarray, ks=(1, 3, 5, 10)):
    """sim: (n_offers, n_masters). Returns recall@k and MRR."""
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == gold_pos[:, None], axis=1) + 1
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks).mean())
    out["median_rank"] = float(np.median(ranks))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a @ b.T


# Baseline: exactly what production is configured to use today.
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(
    n_features=384, alternate_sign=False, norm=None,
    lowercase=False, ngram_range=(1, 2),
)
base_offers = np.asarray(hv.transform(offer_texts).todense(), dtype=np.float32)
base_masters = np.asarray(hv.transform(master_texts).todense(), dtype=np.float32)
baseline = recall_at_k(cosine(base_offers, base_masters), gold_positions)

print("BASELINE - local_hashing (current production config)")
for k, v in baseline.items():
    print(f"  {k:<14} {v:.4f}")


# ============================================ CELL 5: evaluate candidate models
CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 384-dim, matches current dim
    "BAAI/bge-small-en-v1.5",                   # 384-dim, stronger retrieval
    "sentence-transformers/all-mpnet-base-v2",  # 768-dim, best quality/slowest
]
BATCH = 256

results = []
for name in CANDIDATES:
    print(f"\n=== {name} ===")
    model = SentenceTransformer(name, device=DEVICE)
    dim = model.get_sentence_embedding_dimension()

    if DEVICE == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    off_vec = model.encode(offer_texts, batch_size=BATCH,
                           convert_to_numpy=True, normalize_embeddings=False,
                           show_progress_bar=False)
    mas_vec = model.encode(master_texts, batch_size=BATCH,
                           convert_to_numpy=True, normalize_embeddings=False,
                           show_progress_bar=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    n = len(offer_texts) + len(master_texts)
    vram = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0
    metrics = recall_at_k(cosine(off_vec, mas_vec), gold_positions)

    row = {"model": name.split("/")[-1], "dim": dim,
           "texts_per_sec": n / elapsed, "seconds": elapsed,
           "vram_mb": vram, **metrics}
    results.append(row)
    for k in ("recall@1", "recall@5", "MRR", "median_rank"):
        print(f"  {k:<14} {metrics[k]:.4f}"
              f"   (baseline {baseline[k]:.4f})")
    print(f"  {'texts/sec':<14} {n / elapsed:,.0f}   VRAM {vram:.0f} MB")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

summary = pd.DataFrame(
    [{"model": "local_hashing (current)", "dim": 384,
      "texts_per_sec": np.nan, "seconds": np.nan, "vram_mb": 0.0, **baseline}]
    + results
)
print("\n" + summary[
    ["model", "dim", "recall@1", "recall@5", "MRR", "median_rank",
     "texts_per_sec", "vram_mb"]
].to_string(index=False))


# ======================================= CELL 6: batch-size + precision sweep
BEST = summary.iloc[1:].sort_values("recall@5", ascending=False).iloc[0]["model"]
BEST_FULL = next(c for c in CANDIDATES if c.endswith(BEST))
print(f"Sweeping {BEST_FULL} on {DEVICE}\n")

sweep_texts = (offer_texts * 10)[:20_000]   # ~ one production chunk
rows = []
for half in ([False, True] if DEVICE == "cuda" else [False]):
    model = SentenceTransformer(BEST_FULL, device=DEVICE)
    if half:
        model = model.half()
    for bs in (32, 64, 128, 256, 512):
        if DEVICE == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.encode(sweep_texts, batch_size=bs, convert_to_numpy=True,
                     show_progress_bar=False)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - start
        rows.append({
            "precision": "fp16" if half else "fp32", "batch": bs,
            "texts_per_sec": len(sweep_texts) / dt,
            "vram_mb": (torch.cuda.max_memory_allocated() / 1e6)
                       if DEVICE == "cuda" else 0.0,
        })
        print(f"  {'fp16' if half else 'fp32'} bs={bs:<4} "
              f"{len(sweep_texts) / dt:>9,.0f} texts/sec  "
              f"{rows[-1]['vram_mb']:>6.0f} MB")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

sweep = pd.DataFrame(rows)
best_cfg = sweep.sort_values("texts_per_sec", ascending=False).iloc[0]

# Project onto a real run: ~9.6% of a 254k-row file is own-brand, each offer
# embeds once and each candidate once (top_k=5 -> ~6 texts per offer).
own = int(254_479 * 0.096)
texts = own * 6
print(f"\nProjected full run: {own:,} own-brand offers -> ~{texts:,} texts")
print(f"  at {best_cfg['texts_per_sec']:,.0f} texts/sec "
      f"= {texts / best_cfg['texts_per_sec'] / 60:.1f} min of embedding")


# ============================================ CELL 7: paste-ready config block
print(f"""
Add to config/default.yaml under `embedding:` once you're happy:

embedding:
    enabled: true
    backend: local_sentence_transformer
    model_name: {BEST_FULL}
    model_version: {BEST_FULL.split('/')[-1]}-v1
    batch_size: {int(best_cfg['batch'])}
    device: {'cuda' if DEVICE == 'cuda' else 'cpu'}
    normalize_vectors: true
    similarity_metric: cosine
    local_files_only: false   # set true after the model is cached locally

Notes before integrating:
  * `local_files_only: true` in the current config will make the first load
    FAIL unless the model is already in the HF cache. Pre-download it, or flip
    this false for the first run.
  * The scorer normalizes vectors itself (embedding/scorer.py), so leaving
    normalize_embeddings=False in the backend is correct - don't double-normalize.
  * Dimension only matters for the cache: switching dim invalidates
    embedding_cache.sqlite3, so delete it when you change models.
  * Recall here is a LOWER BOUND - the pipeline feeds a richer offer text.
""")
