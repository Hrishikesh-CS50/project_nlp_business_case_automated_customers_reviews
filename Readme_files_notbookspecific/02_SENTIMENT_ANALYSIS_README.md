# Sentiment Classification Notebook - README
## Amazon Product Reviews: Supervised Classification

---

## 📋 Overview

This notebook classifies Amazon product reviews into sentiment categories using machine learning.

**Classification Task:**
- **Negative:** 1-2 star reviews
- **Neutral:** 3 star reviews
- **Positive:** 4-5 star reviews

**Approach:** TF-IDF vectorization + LinearSVC (Support Vector Machine)

---

## 📁 Input File

**File:** `final_dataset_sentiment_analysis.csv`  
**Path:** `../datasets/processed/`  
**Source:** Output from `01_data_prep.ipynb` (data preparation notebook)

**Dataset:**
- 59,630 reviews
- 27 columns
- 94 unique Amazon products

---

## 🔄 Workflow

### Step-by-Step Process:

1. **Load Data** - Import cleaned dataset
2. **Check Ratings** - Visualize star rating distribution (1-5 stars)
3. **Map to Sentiment** - Convert ratings to sentiment labels
4. **Prepare Text** - Combine review title + review text
5. **Clean Text** - Remove URLs, HTML, normalize spacing
6. **Split Data** - 80% train, 10% validation, 10% test
7. **Build Model** - Create TF-IDF + LinearSVC pipeline
8. **Train** - Fit model on training data
9. **Evaluate** - Test on validation and test sets with metrics
10. **Apply** - Predict sentiment for ALL 59,630 reviews
11. **Export** - Save enriched dataset with predictions

---

## 🤖 Model Details

### Pipeline Architecture

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # Use 1-word and 2-word phrases
        min_df=2,              # Ignore words in <2 documents
        max_df=0.95,           # Ignore words in >95% documents
        sublinear_tf=True      # Apply log scaling
    )),
    ("clf", LinearSVC(
        class_weight="balanced",  # Handle imbalanced data
        random_state=42           # Reproducible results
    ))
])
```

### Why This Approach?

**TF-IDF (Term Frequency-Inverse Document Frequency):**
- Converts text to numerical features
- Highlights important words specific to documents
- Reduces weight of common words

**LinearSVC (Linear Support Vector Classifier):**
- Fast and accurate for text classification
- Works well with high-dimensional data (lots of words)
- Industry standard for sentiment analysis
- `class_weight="balanced"` handles imbalanced classes

**No Model Comparison:**
- Went directly with best-known approach (LinearSVC)
- Saves time and computational resources
- Industry practice: use what works

---

## 📊 Results

### Dataset After Cleaning

| Metric | Value |
|--------|-------|
| Total reviews | 59,630 |
| Reviews for training | 46,978 |
| Removed (too short) | 12,652 |

### Data Split

| Set | Count | Percentage |
|-----|-------|------------|
| Training | 37,582 | 80% |
| Validation | 4,698 | 10% |
| Test | 4,698 | 10% |

### Model Performance (Validation Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | **93.85%** |
| **F1 Score (macro)** | **0.663** |
| **F1 Score (weighted)** | **0.934** |
| Precision (macro) | 0.697 |
| Recall (macro) | 0.639 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negative** | 0.70 | 0.65 | 0.67 | 184 |
| **Neutral** | 0.43 | 0.29 | 0.34 | 205 |
| **Positive** | 0.96 | 0.98 | 0.97 | 4,309 |

### Sentiment Distribution (Training Data)

```
Positive:  91.8% (~43,000 reviews)
Neutral:    4.3% (~2,000 reviews)
Negative:   3.9% (~1,800 reviews)
```

---

## ⚠️ Known Limitations

### Class Imbalance

**The Problem:**
- 92% of reviews are positive
- Only 8% are neutral or negative combined
- This is **realistic** for Amazon (most reviews are positive)

**Impact:**
- ✅ Excellent at detecting positive reviews (97% F1)
- ⚠️ Struggles with neutral reviews (34% F1)
- ⚠️ Good but not perfect at negative reviews (67% F1)

**Why It Matters:**
- Some neutral reviews might be mislabeled as positive
- Some negative reviews might be missed
- "Top complaints" analysis may have limited data

**What Was Done:**
- Used `class_weight="balanced"` to give minority classes more importance
- This is a **data characteristic**, not a model failure
- Real Amazon reviews ARE heavily skewed positive

**Is This Acceptable?**
- ✅ **YES** for this project
- Main goal is clustering and summarization (positive products)
- Model still achieves 94% overall accuracy
- Honest limitation to discuss in presentation

---

## 📤 Output File

**File:** `sentiment_enriched_reviews.csv`  
**Path:** `../datasets/processed/`

### Columns Exported:

| Column | Description |
|--------|-------------|
| `reviews.id` | Unique review identifier |
| `asins` | Product ASIN code |
| `name` | Product name |
| `reviews.text` | Review text |
| `reviews.rating` | Original star rating (1-5) |
| `predicted_sentiment_all` | **NEW:** Predicted sentiment (negative/neutral/positive) |

**Stats:**
- Total rows: 59,630
- All reviews have predictions
- Ready for next step: ✅ **Clustering**

---

## 💻 Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### How to Run

1. **Make sure input file exists:**
   ```
   ../datasets/processed/final_dataset_sentiment_analysis.csv
   ```

2. **Run all cells** in order (from top to bottom)

3. **Output automatically saved to:**
   ```
   ../datasets/processed/sentiment_enriched_reviews.csv
   ```

### Runtime

- **Total time:** ~4-5 minutes
- Loading data: 10 seconds
- Training model: 2-3 minutes
- Predictions: 30 seconds
- Export: 5 seconds

---

## 📂 File Structure

```
project/
├── datasets/
│   └── processed/
│       ├── final_dataset_sentiment_analysis.csv  [INPUT]
│       └── sentiment_enriched_reviews.csv        [OUTPUT]
├── notebooks/
│   ├── 01_data_prep.ipynb                        [PREVIOUS]
│   └── 02_Tf_idf_linearsvc_ratings_supervised.ipynb  [THIS ONE]
└── models/
    └── (optional: save trained model)
```

---

## 🔧 Troubleshooting

### Common Errors

**1. FileNotFoundError**
```
FileNotFoundError: ../datasets/processed/final_dataset_sentiment_analysis.csv
```
**Fix:** Run `01_data_prep.ipynb` first

**2. DtypeWarning**
```
DtypeWarning: Columns have mixed types
```
**Fix:** This is just a warning, notebook handles it automatically

**3. Memory Error**
```
MemoryError: Unable to allocate array
```
**Fix:** Close other programs, need ~2GB RAM

**4. Training Takes Forever**
```
Model stuck during training...
```
**Fix:** Normal! Training takes 2-3 minutes. Be patient.

---

## ➡️ Next Steps

After running this notebook, you have:

✅ Trained sentiment classifier (94% accuracy)  
✅ Sentiment predictions for all 59,630 reviews  
✅ Clean dataset ready for clustering  

**Next Notebook:** `03_clustering.ipynb`
- Input: `sentiment_enriched_reviews.csv`
- Task: Group products into 4-6 categories
- Goal: Analyze sentiment patterns per category

---

## 📊 Understanding the Metrics

### Accuracy
**What:** Percentage of correct predictions  
**Your score:** 93.85%  
**Meaning:** Model gets 94 out of 100 reviews right

### Precision
**What:** Of predictions for class X, how many are correct?  
**Example:** Positive precision = 96%  
**Meaning:** When model says "positive", it's right 96% of time

### Recall
**What:** Of all actual X reviews, how many did we find?  
**Example:** Negative recall = 65%  
**Meaning:** Model finds 65 out of 100 negative reviews

### F1 Score
**What:** Balance between precision and recall  
**Formula:** 2 × (precision × recall) / (precision + recall)  
**Use:** Better than accuracy for imbalanced data

### Confusion Matrix
**What:** Shows where model makes mistakes  
**Your patterns:**
- Most positive reviews correctly identified ✓
- Some neutral mislabeled as positive (class imbalance)
- Negative detection decent but could be better

---

## 🎓 For Your Presentation

### Key Talking Points:

**The Problem:**
> "Amazon has 60,000 product reviews across 94 products. Manually reading and categorizing these by sentiment would take weeks. I automated this using machine learning."

**The Solution:**
> "I used TF-IDF vectorization combined with LinearSVC, which is the industry standard for text classification. This approach converts text to numerical features and learns patterns to predict sentiment."

**The Results:**
> "The model achieved 94% accuracy overall, with particularly strong performance on positive reviews (97% F1 score). This is exactly what we need for the next step: identifying top products in each category."

**The Challenge:**
> "The dataset naturally contains 92% positive reviews, which is realistic for Amazon. I addressed this imbalance using class weighting to ensure the model learns from all sentiment types, not just positive."

**The Impact:**
> "This enriched dataset enables automatic product clustering and review summarization, transforming 60,000 reviews into actionable insights in minutes instead of weeks."

### Sample Demo Flow:

1. Show raw review → model predicts "positive"
2. Show another review → model predicts "negative"
3. Show accuracy metrics (94%)
4. Show confusion matrix visualization
5. Show exported dataset with predictions

---

## 📝 Technical Notes

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `ngram_range` | (1, 2) | Captures phrases like "not good" |
| `min_df` | 2 | Removes typos and rare words |
| `max_df` | 0.95 | Removes common words like "the" |
| `sublinear_tf` | True | Prevents long reviews from dominating |
| `class_weight` | 'balanced' | Handles 92% positive imbalance |
| `random_state` | 42 | Makes results reproducible |

### Why These Choices?

**ngram_range=(1,2):**  
Context matters! "not good" has different meaning than "good"

**min_df=2:**  
Removes noise (typos, names, rare words)

**max_df=0.95:**  
Removes words that appear in almost all reviews (uninformative)

**sublinear_tf:**  
Long reviews shouldn't get more weight just for being longer

**class_weight='balanced':**  
Gives equal importance to all classes despite imbalance

---

## 🔍 Code Quality

### What's Good:
- ✅ Clear comments explaining each step
- ✅ Logical flow from data to predictions
- ✅ Proper train/validation/test split
- ✅ Multiple evaluation metrics
- ✅ Visualization of results
- ✅ Clean export for next step

### Minor Issues (Already in Your Notebook):
- Line 63: Triple assignment `CSV_PATH = CSV_PATH = CSV_PATH =`  
  (Works but could be just `CSV_PATH =`)
- Some unused imports (LogisticRegression, MultinomialNB)
- These don't affect functionality!

---

## 📚 What You Learned

### Machine Learning Concepts:
- Supervised classification
- TF-IDF vectorization
- Support Vector Machines (SVM)
- Train/validation/test splits
- Evaluation metrics
- Class imbalance handling

### Practical Skills:
- Text preprocessing
- Scikit-learn pipelines
- Model evaluation
- Data export for downstream tasks

---

## ✅ Project Checklist

- [x] Load and explore data
- [x] Map ratings to sentiment labels
- [x] Clean and preprocess text
- [x] Split into train/val/test
- [x] Build ML pipeline
- [x] Train model
- [x] Evaluate performance
- [x] Predict on full dataset
- [x] Export enriched data
- [ ] Move to clustering (next notebook)

---

## 📖 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explained](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [LinearSVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

---

## 🎯 Quick Reference

| Item | Value |
|------|-------|
| **Input** | `final_dataset_sentiment_analysis.csv` |
| **Output** | `sentiment_enriched_reviews.csv` |
| **Model** | TF-IDF + LinearSVC |
| **Accuracy** | 93.85% |
| **Training Time** | ~3 minutes |
| **Reviews Processed** | 59,630 |
| **Status** | ✅ Complete |
| **Next Step** | Clustering (Task 2) |

---

**Created by:** Bootcamp Student  
**Project:** NLP - Amazon Product Reviews  
**Task:** 1 of 3 (Sentiment Classification)  
**Environment:** Python 3.12.5, nlp_venv  
**Date:** 2024
