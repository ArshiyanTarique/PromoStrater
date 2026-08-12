# gold_data_PERFECT

Human-decided SKU mappings. Every label in this folder was chosen by a person
looking at the offer text and the Product Master — none of it is machine-generated.

This is the answer key the models are measured against. The auto-generated labels
in `data/processed/training_features.parquet` were measured against it and found
to be **~34% wrong**, which is why this folder exists.

## Files

| File | Offers | Role |
|---|---|---|
| `Human_Reviewed_100Rows.xlsx` | 96 usable | first batch, sampled by offer frequency |
| `Human_Reviewed_70Rows.xlsx` | 64 usable | second batch, sampled **randomly** to cover the long tail |
| `Human_Reviewed_160_MERGED.xlsx` | 160 | the two above merged, descriptions resolved to item codes — **this is what training reads** |
| `Human_Reviewed_30Rows.xlsx` | 29 | **HELD OUT. Never train on this.** |
| `sku_reference.xlsx` | 237 SKUs | lookup sheet for filling in new reviews |

An offer may map to up to 4 SKUs: bundle offers advertise several products, and
some offers are ambiguous ("ZING CHICKEN STRIPS 750g" could be the spicy or the
non-spicy SKU, so both are correct).

## The held-out 30

`Human_Reviewed_30Rows.xlsx` is the only unbiased measurement available. It was
sampled at random from offers never reviewed, and no model has trained or tuned
a threshold on it.

Training on it, or repeatedly tuning until it improves, destroys the only honest
read of real-world accuracy. On the current model it reports **Hit@1 0.724**,
against **0.938** on offers the model has seen — that gap is exactly why it is
kept separate.

With n=29 it carries roughly ±16 points of uncertainty, so only treat a change
above ~10 points as real. Growing it to 60–80 offers would make it trustworthy.

## synthesized/

Machine-generated rows. Useful for training, never for measurement.

| File | What it is |
|---|---|
| `propagated_from_160rows.xlsx` | human decisions copied onto near-identical offer texts (≥95% similar, with size and variant guards). The `source` column marks each row `human` or `propagated`. |

Anything generated to cover SKUs with no human example belongs here too.

**Rule: measure on `source=human` only.** A propagated row inherits a judgement
rather than being one, so scoring against it flatters the model.

Note: propagated rows were measured as adding volume but not information — a
model trained on human rows alone matched the one trained on human + propagated.
Their value is coverage of the dump, not model quality.

## Regenerating

```bash
.venv/Scripts/python.exe scripts/resolve_descriptions.py     # descriptions -> item codes
.venv/Scripts/python.exe scripts/propagate_gold_labels.py    # copy to near-identical offers
.venv/Scripts/python.exe scripts/train_final_model.py        # retrain + save
.venv/Scripts/python.exe scripts/evaluate_full_pipeline.py   # compare all models
```
