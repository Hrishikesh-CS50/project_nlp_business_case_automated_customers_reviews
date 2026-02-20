# Data Preparation Notebook - README

## Overview

This notebook (`01_data_prep.ipynb`) performs comprehensive data cleaning and preprocessing on Amazon product review datasets. It merges three separate CSV files, removes duplicates, handles missing values, and prepares a clean dataset for sentiment analysis and clustering.

---

## Dataset Information

### Input Files (Raw Data)

Three Amazon product review datasets are merged:

| File | Size | Source |
|------|------|--------|
| `1429_1.csv` | 47 MB | Amazon reviews (21 columns) |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` | 95 MB | Datafiniti dataset (24 columns) |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` | 254 MB | Datafiniti dataset May 2019 (24 columns) |

**Total Raw Data:** ~396 MB (before processing)

### Dataset Source
- Primary: [Datafiniti Amazon Product Reviews (Kaggle)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- Focus: Amazon-branded products (Fire tablets, Kindle, Echo, AmazonBasics)

---

## Data Processing Pipeline

### 1. Data Loading & Merging

```python
# Load three datasets
df1 = pd.read_csv("../datasets/raw/1429_1.csv")
df2 = pd.read_csv("../datasets/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
df3 = pd.read_csv("../datasets/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Align columns (union of all columns) - creates 27 unified columns
all_cols = sorted(set(df1.columns) | set(df2.columns) | set(df3.columns))
df1, df2, df3 = [d.reindex(columns=all_cols) for d in (df1, df2, df3)]

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)
```

**Result:** Combined dataset with 27 columns (sorted alphabetically)

---

### 2. Duplicate Removal

#### Step 1: Remove Exact Duplicates
```python
# Found 95 exact duplicate rows
df = df.drop_duplicates()
```

#### Step 2: Remove Duplicate Reviews
```python
# Found 8,260 duplicate reviews (same id + same review text)
df = df.drop_duplicates(subset=["id", "reviews.text"], keep="first")
```

**Challenge Identified:** Some identical review texts have different ratings across products
- Example: "good quality" appears with both 4-star and 5-star ratings
- These are legitimate reviews for different products (not duplicates)
- Solution: Keep all reviews, as they refer to different products

#### Step 3: Handle Rating Conflicts
```python
# Manually dropped 3 specific conflicting rows after inspection
df = df.drop(index=[37992, 38043, 48173])
```

---

### 3. Final Dataset Statistics

**Output:** `final_dataset_sentiment_analysis.csv`

| Metric | Value |
|--------|-------|
| **Total Reviews** | 59,630 |
| **Total Columns** | 27 |
| **Unique Products (ASINs)** | 94 |
| **Unique Brands** | 8 |
| **Unique Product Names** | 125 |
| **Average Rating** | 4.55 / 5.0 |
| **Rating Range** | 1.0 - 5.0 |

---

## Column Schema

### Essential Columns (Non-Null)

| Column | Description | Non-Null Count |
|--------|-------------|----------------|
| `asins` | Amazon product identifier | 59,628 |
| `name` | Product name | 52,870 |
| `reviews.text` | Review content (TEXT DATA) | **59,630** |
| `reviews.rating` | Star rating (1-5) | 59,597 |
| `reviews.title` | Review title/summary | 59,611 |
| `reviews.date` | Review date | 59,591 |
| `reviews.dateSeen` | Date review was scraped | 59,630 |
| `brand` | Product brand | 59,630 |
| `manufacturer` | Manufacturer name | 59,630 |
| `categories` | Product categories | 59,630 |

### Mostly Empty Columns (Can be Ignored)

| Column | Non-Null Count | Reason |
|--------|----------------|---------|
| `reviews.userCity` | 0 | Empty - privacy/unavailable |
| `reviews.userProvince` | 0 | Empty - privacy/unavailable |
| `reviews.didPurchase` | 10 | Almost empty |
| `reviews.id` | 68 | Almost empty |

### Partially Available Columns

| Column | Non-Null Count | Description |
|--------|----------------|-------------|
| `reviews.doRecommend` | 48,024 | Whether user recommends product |
| `reviews.numHelpful` | 48,112 | Helpful vote count |
| `dateAdded` | 24,971 | Product addition date |
| `dateUpdated` | 24,971 | Product update date |
| `imageURLs` | 24,971 | Product images |
| `primaryCategories` | 24,971 | Main category |

---

## Data Quality Issues Addressed

### ✅ Issues Resolved

1. **Exact Duplicates:** 95 rows removed
2. **Duplicate Reviews:** 8,260 duplicate review texts removed
3. **Mixed Data Types:** Columns aligned across all three datasets
4. **Column Inconsistencies:** Unified schema with 27 columns
5. **Rating Conflicts:** 3 problematic rows manually removed

### ⚠️ Known Limitations

1. **Missing Product Names:** ~6,760 reviews (11.3%) have no product name
2. **Missing Ratings:** 33 reviews (0.06%) have no rating
3. **Empty Columns:** User location data completely unavailable
4. **Partial Data:** Some metadata fields only available for ~42% of reviews

---

## Key Insights from Data

### Product Distribution

- **94 unique products** (Amazon-branded focus)
- **Top product:** Fire HD 8 Tablet (10,966 reviews)
- **8 brands** dominated by Amazon brand

### Review Distribution

- **Rating Distribution:**
  - Mean: 4.55 / 5.0
  - Heavily positive-skewed
  - Most common: 5-star reviews
  
### Categories

- **106 unique category combinations**
- Most common categories:
  - Fire Tablets
  - Kindle eReaders
  - Amazon Echo devices
  - AmazonBasics accessories

---

## Output Files

### 1. Intermediate Output
```
../datasets/processed/merged_dataset_sentiment_analysis.csv
```
- Combined raw data (before duplicate removal)
- Includes all 27 columns

### 2. Final Output (FOR USE IN NEXT STEPS)
```
../datasets/processed/final_dataset_sentiment_analysis.csv
```
- **59,630 reviews**
- **27 columns**
- Duplicates removed
- Ready for sentiment analysis

---

## Usage

### Prerequisites

```bash
pip install pandas numpy
```

### Running the Notebook

1. **Place raw data files in:** `../datasets/raw/`
   - `1429_1.csv`
   - `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
   - `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`

2. **Create output directory:** `../datasets/processed/`

3. **Run all cells** in `01_data_prep.ipynb`

4. **Output location:** `../datasets/processed/final_dataset_sentiment_analysis.csv`

### Expected Runtime

- **Loading:** ~30 seconds
- **Merging:** ~10 seconds
- **Duplicate removal:** ~20 seconds
- **Total:** ~1-2 minutes

---

## Next Steps

After running this notebook, the cleaned data can be used for:

### 1. Sentiment Classification (Task 1)
- Use `reviews.text` and `reviews.rating` columns
- Map ratings to sentiment classes:
  - 1-2 stars → Negative
  - 3 stars → Neutral
  - 4-5 stars → Positive

### 2. Product Category Clustering (Task 2)
- Use `categories`, `name`, `reviews.text` columns
- Group products into 4-6 meta-categories

### 3. Review Summarization (Task 3)
- Aggregate reviews by product/category
- Generate summary articles using LLMs

---

## File Structure

```
project/
├── datasets/
│   ├── raw/
│   │   ├── 1429_1.csv (47 MB)
│   │   ├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv (95 MB)
│   │   └── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv (254 MB)
│   └── processed/
│       ├── merged_dataset_sentiment_analysis.csv (intermediate)
│       └── final_dataset_sentiment_analysis.csv (FINAL - 59,630 reviews)
└── notebooks/
    └── 01_data_prep.ipynb
```

---

## Data Quality Metrics

### Completeness

| Column | Completeness | Status |
|--------|-------------|--------|
| `reviews.text` | 100% | ✅ Perfect |
| `reviews.rating` | 99.94% | ✅ Excellent |
| `asins` | 99.99% | ✅ Excellent |
| `name` | 88.7% | ⚠️ Good |
| `reviews.title` | 99.97% | ✅ Excellent |

### Duplicates Removed

| Type | Count Removed |
|------|---------------|
| Exact duplicates | 95 |
| Duplicate reviews | 8,260 |
| Conflicting reviews | 3 |
| **Total** | **8,358** |

### Final Dataset Quality

- ✅ **No exact duplicates**
- ✅ **No duplicate review texts for same product**
- ✅ **Consistent column schema**
- ✅ **All essential columns populated**
- ✅ **Ready for NLP processing**

---

## Technical Notes

### Memory Usage

- Raw data loading: ~400 MB RAM
- After merging: ~12.3 MB DataFrame
- Final export: ~15 MB CSV file

### Data Types

- **String columns (13):** text, product info, URLs
- **Float columns (5):** ratings, numeric IDs, user location (empty)
- **Object columns (9):** dates, structured data

### Performance Tips

1. **For faster loading:** Use `dtype` parameter to specify column types
2. **For large-scale processing:** Consider chunking or Dask
3. **For memory optimization:** Drop unused columns after loading

---

## Known Issues & Solutions

### Issue 1: Mixed data types warning
```
DtypeWarning: Columns (0,1) have mixed types
```
**Solution:** Specify `dtype` or use `low_memory=False` when loading

### Issue 2: Some product names missing
**Cause:** Original datasets incomplete
**Impact:** 11.3% of reviews affected
**Mitigation:** Use `asins` as primary identifier

### Issue 3: Identical review text for different products
**Cause:** Generic reviews (e.g., "great", "good quality")
**Solution:** Keep all - they're legitimate reviews for different products

---

## Author Notes

- **Created:** Data preparation phase
- **Purpose:** Clean and merge raw Amazon review data
- **Next notebook:** `02_sentiment_classification.ipynb`
- **Environment:** Python 3.12.5, pandas, numpy

---

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- Project README for full context

---

## Quick Reference

**Input:** 3 CSV files (~396 MB raw data)  
**Output:** 1 clean CSV file (59,630 reviews, 27 columns)  
**Processing:** Merge → Deduplicate → Clean → Export  
**Runtime:** ~1-2 minutes  
**Memory:** ~400 MB RAM required  

**Ready for:** Sentiment classification, clustering, summarization
