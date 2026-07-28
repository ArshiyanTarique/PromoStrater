# 📖 Stage-by-Stage Code Explanation & Logic Guide

This document breaks down the Python code for all 5 stages of the SKU Auto-Mapper pipeline. It is aligned directly with the **`Product Master.xlsx`** schema.

---

## 📌 Table of Contents
* [🟢 Stage 1: Data Standardization & Cleaning](#-stage-1-data-standardization--cleaning)
  * [🎯 Objective](#-objective)
  * [💻 Code Walkthrough](#-code-walkthrough)
  * [🔍 How the Code Works Step-by-Step](#-how-the-code-works-step-by-step)
* [🟢 Stage 2: Variant Splitting & De-aggregation](#-stage-2-variant-splitting--de-aggregation)
  * [🎯 Objective](#-objective-1)
  * [💻 Code Walkthrough](#-code-walkthrough-1)
  * [🔍 How the Code Works Step-by-Step](#-how-the-code-works-step-by-step-1)
* [🟢 Stage 3: Master SKU Matching](#-stage-3-master-sku-matching)
  * [🎯 Objective](#-objective-2)
  * [💻 Code Walkthrough](#-code-walkthrough-2)
  * [🔍 How the Code Works Step-by-Step](#-how-the-code-works-step-by-step-2)
* [🟢 Stage 4: Competitor Discovery](#-stage-4-competitor-discovery)
  * [🎯 Objective](#-objective-3)
  * [💻 Code Walkthrough](#-code-walkthrough-3)
  * [🔍 How the Code Works Step-by-Step](#-how-the-code-works-step-by-step-3)
* [🟢 Stage 5: Output & Audit Matrix](#-stage-5-output--audit-matrix)
  * [🎯 Objective](#-objective-4)
  * [💻 Code Walkthrough](#-code-walkthrough-4)
  * [🔍 How the Code Works Step-by-Step](#-how-the-code-works-step-by-step-4)
* [📊 End-to-End Output Matrix Example](#-end-to-end-output-matrix-example)

---

## 🟢 Stage 1: Data Standardization & Cleaning

### 🎯 Objective
Transform raw, noisy, and abbreviated SKU text into a clean, uniform string so that fuzzy matching engines don't get tripped up by minor typos, uppercase letters, or shorthand abbreviations.

### 💻 Code Walkthrough

```python
replacements = {
    "chk": "chicken", "chkn": "chicken", 
    "nugg": "nuggets", "reg": "regular", 
    "spcy": "spicy", "strp": "strips", 
    "gm": "g", "0.5kg": "500g"
}

clean_skus = []
for item in df["raw_sku_description"]:
    text = str(item).lower()
    words = text.split()
    
    cleaned_words = []
    for word in words:
        if "/" in word:
            sub_words = word.split("/")
            cleaned_subs = [replacements.get(sw, sw) for sw in sub_words]
            cleaned_words.append("/".join(cleaned_subs))
        else:
            cleaned_words.append(replacements.get(word, word))
            
    clean_skus.append(" ".join(cleaned_words))

df["clean_sku_description"] = clean_skus