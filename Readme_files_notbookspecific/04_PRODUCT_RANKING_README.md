# Product Ranking & Sentiment Analysis - README

## Overview

This notebook bridges **Task 2 (Clustering)** and **Task 3 (Summarization)** by analyzing products within each category and identifying the best and worst performers based on sentiment analysis and ratings.

**Purpose:** Rank products to identify top recommendations and products to avoid for AI-powered review summarization.

---

## Input Files

### Required Files:

| File | Source | Description |
|------|--------|-------------|
| `clustered_reviews_export.csv` | Clustering notebook | 59,630 reviews with category labels |
| `sentiment_enriched_reviews.csv` | Sentiment notebook | 59,630 reviews with sentiment predictions |

**Location:** `../datasets/processed/`

---

## What This Notebook Does

### Step-by-Step Workflow:

1. **Load Data** → Import clustered and sentiment-enriched datasets
2. **Validate Data** → Check for mismatches and data quality
3. **Merge Datasets** → Combine using composite key (ASIN + text + rating)
4. **Clean Data** → Remove rows with missing critical values
5. **Calculate Metrics** → Aggregate sentiment and ratings per product
6. **Score Products** → Calculate quality scores combining sentiment + ratings
7. **Rank Products** → Identify top 3 products per category
8. **Find Worst** → Identify worst product per category
9. **Extract Complaints** → Get negative reviews for analysis
10. **Visualize** → Create charts showing rankings
11. **Export** → Save results for LLM summarization
12. **Report** → Generate summary statistics

---

## Quality Score Formula

### Two-Step Calculation:

**Step 1: Sentiment Score** (-1 to +1)
```
sentiment_score = (positive_count × 1.0 + 
                   neutral_count × 0.5 + 
                   negative_count × -1.0) / total_reviews
```

**Step 2: Quality Score** (0 to 1)
```
quality_score = 0.6 × normalized_sentiment + 0.4 × normalized_rating

Where:
  - normalized_sentiment = (sentiment_score + 1) / 2  [converts -1:1 → 0:1]
  - normalized_rating = avg_rating / 5.0              [converts 1:5 → 0:1]
```

### Why This Formula?

- **60% Sentiment Weight:** ML predictions capture nuanced opinions beyond star ratings
- **40% Rating Weight:** Traditional ratings are familiar and reliable
- **Includes Neutral:** Unlike binary positive/negative, considers mixed feelings
- **Score Range 0-1:** Easy to interpret (higher = better)

### Example:

```
Product: Fire HD 8 Tablet
  - 90% positive, 5% neutral, 5% negative
  - Average rating: 4.5 stars

Calculation:
  sentiment_score = (0.90×1 + 0.05×0.5 + 0.05×-1) = 0.875
  quality_score = 0.6 × (0.875+1)/2 + 0.4 × (4.5/5)
                = 0.6 × 0.9375 + 0.4 × 0.90
                = 0.5625 + 0.36
                = 0.923 (Excellent!)
```

---

## Results

### Dataset Statistics:

| Metric | Value |
|--------|-------|
| **Total Products Analyzed** | 114 |
| **Total Categories** | 5 |
| **Total Reviews Processed** | 52,870 |
| **Match Rate** | 99.9% |

### Category Breakdown:

| Category | Products | Reviews | Avg Quality | Avg Rating |
|----------|----------|---------|-------------|------------|
| Computers and electronics | 83 | 19,220 | 0.928 | 4.53 ⭐ |
| Amazon devices & accessories | 19 | 7,423 | 0.945 | 4.59 ⭐ |
| Entertainment Appliances | 3 | 13 | 0.991 | 4.89 ⭐ |
| Kids Toys & kids entertainment | 7 | 14,652 | 0.944 | 4.60 ⭐ |
| Batteries and household essentials | 2 | 11,562 | 0.967 | 4.78 ⭐ |

### Data Processing:

```
Original clustered dataset:  59,630 rows
Original sentiment dataset:  59,630 rows
Merged dataset:              59,634 rows (+4 duplicates)
After cleaning:              52,870 rows (88.7% retained)
Removed:                     6,764 rows (missing names/data)
```

---

## Output Files

### 1. Main Output (for LLM):
**`product_summary_data.json`**
- Format: JSON
- Contents: Structured data for each category
  - Top 3 products with details
  - Worst product with complaints
  - Sample positive/negative reviews
- Use: Direct input to ChatGPT/LLM for summarization

### 2. Full Product Statistics:
**`product_statistics.csv`**
- All 114 products with complete metrics
- Columns: ASIN, name, category, reviews, ratings, sentiment %, quality score
- Use: Analysis, reporting, visualization

### 3. Top Products:
**`top_3_products_per_category.csv`**
- Top 3 products from each category
- Ranked by quality score
- Use: Recommendations, comparisons

### 4. Worst Products:
**`worst_products_per_category.csv`**
- Lowest-scoring product per category
- Includes negative review samples
- Use: Products to avoid, complaint analysis

---

## Technical Details

### Merge Strategy:

**Composite Key:**
```python
composite_key = clean_asin + "|" + clean_name + "|" + clean_text + "|" + clean_rating
```

**Why composite key?**
- More reliable than single ASIN (handles duplicates)
- Ensures exact review matching
- Achieves 99.9% match rate

### Filtering:

**Minimum Review Threshold:** 3 reviews
- Filters out products with insufficient data
- Ensures statistical reliability
- Removed 34 low-sample products

### Data Cleaning:

Rows removed if missing:
- Product ASIN
- Product name
- Category assignment
- Sentiment prediction

---

## Usage

### Prerequisites:

```bash
pip install pandas numpy matplotlib seaborn
```

### Running the Notebook:

1. **Ensure input files exist:**
   ```
   ../datasets/processed/clustered_reviews_export.csv
   ../datasets/processed/sentiment_enriched_reviews.csv
   ```

2. **Run all cells** in order (top to bottom)

3. **Check outputs:**
   ```
   ../datasets/processed/product_summary_data.json
   ../datasets/processed/product_statistics.csv
   ../datasets/processed/top_3_products_per_category.csv
   ../datasets/processed/worst_products_per_category.csv
   ```

### Expected Runtime:

- Data loading: 10 seconds
- Merging: 15 seconds
- Calculations: 30 seconds
- Visualization: 10 seconds
- Export: 5 seconds
- **Total: ~1-2 minutes**

---

## Output Schema

### product_summary_data.json Structure:

```json
[
  {
    "category": "Computers and electronics",
    "top_3_products": [
      {
        "name": "Fire HD 8 Tablet",
        "asin": "B01AHB9CN2",
        "quality_score": 0.938,
        "avg_rating": 4.6,
        "total_reviews": 10966,
        "positive_pct": 94.2,
        "positive_reviews": ["Great value!", "Love it!"],
        "negative_reviews": ["Battery could be better"]
      },
      // ... 2 more products
    ],
    "worst_product": {
      "name": "Kindle Power Adapter",
      "quality_score": 0.464,
      "negative_pct": 60.0,
      "negative_reviews": ["Stopped working", "Poor quality"]
    }
  },
  // ... 4 more categories
]
```

---

## Visualizations Generated

The notebook creates several visualizations:

1. **Quality Score Distribution** - Histogram of all product scores
2. **Top Products by Category** - Bar chart showing rankings
3. **Sentiment Distribution** - Pie/bar charts per category
4. **Rating vs Quality Score** - Scatter plot showing correlation

---

## Known Limitations

### 1. Sample Size Not Weighted
**Issue:** Products with 5 reviews can rank same as 5000 reviews  
**Impact:** Small sample products may rank artificially high/low  
**Mitigation:** Applied minimum 3-review filter  
**Future:** Add Bayesian averaging or confidence intervals

### 2. Missing Product Names (11%)
**Issue:** 6,764 reviews lack product names  
**Cause:** Data quality issue from source  
**Impact:** These reviews excluded from analysis  
**Status:** Acceptable for this project scope

### 3. Duplicate Reviews
**Issue:** 4 duplicate rows after merge  
**Cause:** Composite key matching edge cases  
**Impact:** Minimal (<0.01% of data)  
**Status:** Within acceptable tolerance

### 4. Inf/NaN Values
**Issue:** Some products show inf or NaN in scores  
**Cause:** Division by zero when total_reviews = 0  
**Impact:** These products filtered out  
**Status:** Handled by data cleaning step

---

## Quality Checks

### Validation Steps:

✓ **Merge Quality:** 99.9% match rate  
✓ **Data Completeness:** 88.7% retention  
✓ **Score Range:** All scores in [0, 1]  
✓ **Category Coverage:** All 5 categories have products  
✓ **Top 3 Selection:** All categories (except 2-product category)  
✓ **Worst Product:** Identified for all viable categories  

---

## Project Requirements Met

### Task 2 (Clustering) - Output Usage:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Use clustered reviews | ✅ | Loads `clustered_reviews_export.csv` |
| Categories (4-6) | ✅ | 5 categories identified |
| Product grouping | ✅ | 114 products grouped |

### Task 3 (Summarization) - Data Preparation:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Top 3 products per category | ✅ | `top_3_products_per_category.csv` |
| Key differences | ✅ | Ratings, sentiment %, quality scores |
| Top complaints | ✅ | Negative reviews extracted |
| Worst product | ✅ | Identified per category |
| LLM-ready format | ✅ | `product_summary_data.json` |

---

## File Structure

```
project/
├── datasets/
│   └── processed/
│       ├── clustered_reviews_export.csv          [INPUT]
│       ├── sentiment_enriched_reviews.csv        [INPUT]
│       ├── product_summary_data.json             [OUTPUT - Main!]
│       ├── product_statistics.csv                [OUTPUT]
│       ├── top_3_products_per_category.csv       [OUTPUT]
│       └── worst_products_per_category.csv       [OUTPUT]
├── notebooks/
│   ├── 01_data_prep.ipynb                        [Step 1]
│   ├── 02_sentiment_classification.ipynb         [Step 2]
│   ├── 03_clustering.ipynb                       [Step 3]
│   └── 04_product_ranking_sentiment_analysis.ipynb [THIS ONE - Step 4]
└── visualizations/
    └── (charts generated by notebook)
```

---

## Next Steps

After running this notebook:

✅ **Task 2 Complete:** Product clustering and ranking done  
➡️ **Task 3 Ready:** Use `product_summary_data.json` for summarization  

**Next Notebook:** Summarization using ChatGPT/LLM
- Input: `product_summary_data.json`
- Task: Generate blog-style articles per category
- Output: AI-generated product recommendations

---

## Troubleshooting

### Common Issues:

**1. FileNotFoundError**
```
FileNotFoundError: clustered_reviews_export.csv
```
**Fix:** Run clustering notebook first to generate input files

**2. KeyError: 'composite_key'**
```
KeyError: 'composite_key'
```
**Fix:** Restart kernel and run all cells in order

**3. Memory Error**
```
MemoryError: Unable to allocate array
```
**Fix:** Close other programs; requires ~3GB RAM

**4. Empty Output Files**
```
JSON file is empty or has no categories
```
**Fix:** Check that input files have data; verify merge succeeded

---

## Performance Metrics

### Processing Stats:

- **Merge efficiency:** 99.9% match rate
- **Data retention:** 88.7% of original rows
- **Coverage:** All 5 categories represented
- **Quality:** No inf/NaN in final scores
- **Completeness:** All required outputs generated

### Computation:

- Memory usage: ~300MB
- Processing time: 1-2 minutes
- Output size: ~2MB total

---

## For Your Presentation

### Key Talking Points:

**The Challenge:**
> "I had two datasets—clustered reviews and sentiment predictions—that needed to be merged and analyzed to identify the best products in each category."

**The Solution:**
> "I created a composite key matching system that achieved 99.9% merge accuracy, then developed a quality score combining 60% sentiment analysis with 40% star ratings."

**The Results:**
> "Successfully ranked 114 products across 5 categories, identifying top 3 recommendations and worst products for each. Generated structured JSON data ready for AI summarization."

**The Impact:**
> "This bridges clustering and summarization, enabling automated product recommendations based on 50,000+ customer reviews."

### Metrics to Highlight:

- ✅ 99.9% merge success rate
- ✅ 52,870 reviews analyzed
- ✅ 5 categories, 114 products
- ✅ Quality scores ranging 0.46 - 1.00
- ✅ 4 output files for different use cases

---

## Code Quality Notes

### Strengths:

- ✅ Clear markdown headers for each section
- ✅ Comprehensive data validation
- ✅ Multiple output formats (JSON + CSV)
- ✅ Quality score well-documented
- ✅ Handles edge cases (missing data, duplicates)

### Minor Areas for Improvement:

- ⚠️ Some code duplication in export section
- ⚠️ Could add more visualizations
- ⚠️ Minimum review threshold could be higher (5-10)

**Overall:** Production-ready for bootcamp project ✓

---

## Summary

### Inputs:
- Clustered reviews (59,630)
- Sentiment predictions (59,630)

### Process:
- Merge datasets (99.9% success)
- Calculate quality scores
- Rank products per category

### Outputs:
- Top 3 products per category ✓
- Worst products per category ✓
- LLM-ready JSON ✓
- Complete statistics ✓

### Status:
✅ **Complete and ready for Task 3 (Summarization)**

---

## Quick Reference

| Item | Value |
|------|-------|
| **Input 1** | `clustered_reviews_export.csv` |
| **Input 2** | `sentiment_enriched_reviews.csv` |
| **Main Output** | `product_summary_data.json` |
| **Products Analyzed** | 114 |
| **Categories** | 5 |
| **Reviews Processed** | 52,870 |
| **Runtime** | 1-2 minutes |
| **Status** | ✅ Complete |
| **Next Step** | Summarization (Task 3) |

---

**Created by:** Bootcamp Student  
**Project:** NLP - Amazon Product Reviews  
**Task:** Bridge between Clustering (Task 2) and Summarization (Task 3)  
**Environment:** Python 3.12.5  
**Date:** 2024
