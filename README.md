<<<<<<< HEAD
# PromoStrater

PromoStrater is an end-to-end SKU mapping and promotional intelligence pipeline designed to automatically match promotional flyer products to a client's master product catalogue.

The project combines traditional text processing, feature engineering, and machine learning to accurately identify product matches while significantly reducing manual review.

---

## Features

- Offer text preprocessing
- Product normalization
- Weight and pack-size extraction
- Protein and product-family detection
- Feature engineering
- LightGBM-based SKU matching
- Confidence scoring
- Competitor identification
- Manual review generation
- Export-ready CSV outputs

---

## Project Structure

```
PromoStrater/
│
├── data/
│   ├── Product_Master.xlsx
│   └── Offer_Dump.csv
│
├── models/
│   └── alkabeer_sku_matcher_v1.joblib
│
├── outputs/
│
├── src/
│
├── sku_mapping_pipeline_ml.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Pipeline

```
Offer Dump
      │
      ▼
Cleaning & Normalization
      │
      ▼
Feature Engineering
      │
      ▼
LightGBM Classifier
      │
      ▼
Best Matching SKU
      │
      ▼
Competitor Discovery
      │
      ▼
CSV Outputs
```

---

## Machine Learning

The ML model predicts whether an offer matches a candidate master SKU using engineered numerical and categorical features.

Example features include:

- Word similarity
- Token similarity
- Character similarity
- Protein match
- Product family match
- Variant match
- Weight match
- Pack-size match
- Mixed protein detection
- Expected match count

---

## Installation

Clone the repository

```bash
git clone https://github.com/ArshiyanTarique/PromoStrater.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

Run the SKU mapping pipeline

```bash
python sku_mapping_pipeline_ml.py
```

---

## Outputs

The pipeline produces CSV files including:

- SKU mappings
- Confidence scores
- Manual review cases
- Competitor mappings

---

## Tech Stack

- Python
- pandas
- NumPy
- RapidFuzz
- LightGBM
- scikit-learn
- openpyxl

---

## Future Improvements

- Embedding-based candidate generation
- Active learning from manual reviews
- Dashboard for business users
- API deployment
- Real-time SKU matching

---

## Author

**Arshiyan Tarique**

Data Analytics Intern  
Salesflo
=======
