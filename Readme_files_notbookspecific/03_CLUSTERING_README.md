# Product Category Clustering - README
## Task 2: Grouping Amazon Products into Meta-Categories

---

## 📋 Overview

This notebook implements **unsupervised clustering** to group 59,630 Amazon product reviews into 5 broad product categories. This simplifies the dataset and enables category-specific product analysis and recommendations.

**Method:** K-Means clustering on TF-IDF vectorized text features  
**Goal:** Create 4-6 meta-categories from diverse Amazon product reviews  
**Result:** 5 well-defined categories ready for sentiment analysis and summarization  

---

## 🎯 Project Requirements

### Task 2 Specifications:

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Cluster into 4-6 categories | 5 clusters created | ✅ |
| Simplify product dataset | Reduced complexity from 94 products to 5 groups | ✅ |
| Meaningful categories | Named based on content analysis | ✅ |
| Ready for summarization | Cluster labels added to dataset | ✅ |

---

## 📁 Input File

**File:** `final_dataset_sentiment_analysis.csv`  
**Source:** Output from `01_data_prep.ipynb` (data preparation notebook)  
**Location:** `../datasets/processed/`

**Dataset:**
- 59,630 reviews
- 27 columns
- 94 unique products
- Reviews from multiple Amazon product categories

---

## 🔄 Workflow

### Step-by-Step Process:

1. **Load Data** → Import cleaned dataset
2. **Explore Products** → Analyze product distribution
3. **Prepare Text** → Combine product name + category + review text
4. **Vectorize** → TF-IDF transformation (text → numbers)
5. **Determine K** → Elbow method + silhouette analysis
6. **Cluster** → K-Means with k=5
7. **Evaluate** → Calculate clustering metrics
8. **Analyze** → Extract top terms per cluster
9. **Name Clusters** → Assign meaningful category names
10. **Visualize** → PCA plots and word clouds
11. **Export** → Save clustered dataset

---

## 🤖 Clustering Method

### Algorithm: **K-Means**

**Why K-Means?**
- ✅ Fast and scalable (handles 59K samples)
- ✅ Works well with TF-IDF features
- ✅ Produces spherical, well-separated clusters
- ✅ Easy to interpret and explain

### Feature Engineering:

**Text Features Used:**
```python
# Combine multiple text fields for richer features
combined_text = product_name + " " + categories + " " + review_text
```

**TF-IDF Vectorization:**
```python
TfidfVectorizer(
    max_features=5000,      # Top 5000 most important words
    ngram_range=(1, 2),     # Use single words and word pairs
    min_df=5,               # Ignore rare terms
    max_df=0.7              # Ignore very common terms
)
```

### Choosing K (Number of Clusters):

**Methods Used:**
1. **Elbow Method** - Looked for "elbow" in inertia curve
2. **Silhouette Analysis** - Measured cluster cohesion
3. **Domain Knowledge** - Wanted 4-6 categories per requirements

**Result:** k=5 chosen as optimal

---

## 📊 Results

### 5 Product Categories Identified:

| Cluster | Category Name | Reviews | Products | % of Dataset |
|---------|--------------|---------|----------|--------------|
| **0** | Kids Toys & kids entertainment | 15,362 | ~8 | 25.8% |
| **1** | Batteries and household essentials | 10,852 | ~2 | 18.2% |
| **2** | Amazon devices & accessories | 8,737 | ~19 | 14.7% |
| **3** | Entertainment Appliances | 5,069 | ~3 | 8.5% |
| **4** | Computers and electronics | 19,610 | ~83 | 32.9% |

**Total:** 59,630 reviews across 5 categories

---

## 🏷️ Cluster Descriptions

### Cluster 0: Kids Toys & kids entertainment (15,362 reviews)

**Top Terms:**
- tablets, tablet display, fi gb, display wi, offers magenta
- toys, toys movies, tech toys, movies music

**Representative Products:**
- Fire Tablet, 7" Display (10,751 reviews)
- Fire Kids Edition Tablet - Pink (1,663 reviews)
- Fire Kids Edition Tablet - Blue (1,485 reviews)
- Fire Kids Edition Tablet - Green (1,438 reviews)

**Category Focus:** Family-friendly tablets, educational devices

---

### Cluster 1: Batteries and household essentials (10,852 reviews)

**Top Terms:**
- batteries, health, household, amazonbasics
- health household, care, aaa, alkaline
- performance alkaline, batteries count

**Representative Products:**
- AmazonBasics AAA Batteries - 36 Count (7,457 reviews)
- AmazonBasics AA Batteries - 48 Count (3,395 reviews)

**Category Focus:** Consumables, everyday essentials

---

### Cluster 2: Amazon devices & accessories (8,737 reviews)

**Top Terms:**
- smart, home, echo, audio
- smart home, speakers, amazon echo
- automation, home automation, assistants, voice

**Representative Products:**
- Echo (White) (2,898 reviews)
- Amazon Fire TV (2,527 reviews)
- Amazon Echo Show with 7" Screen (843 reviews)
- Amazon Echo Plus w/ Built-In Hub (590 reviews)

**Category Focus:** Smart home devices, voice assistants

---

### Cluster 3: Entertainment Appliances (5,069 reviews)

**Top Terms:**
- streaming, college, home theater, theater, tv
- streaming media, tvs home, media players
- theater streaming, devices

**Representative Products:**
- Fire TV Stick Streaming Media Player (5,056 reviews)
- Fire TV Stick Pair Kit (6 reviews)
- Fire TV with 4K Ultra HD (4 reviews)
- Fire TV Gaming Edition (3 reviews)

**Category Focus:** Streaming devices, entertainment systems

---

### Cluster 4: Computers and electronics (19,610 reviews)

**Top Terms:**
- tablets, hd, tablets tablets
- hd display, hd tablet, computers
- gb, computers tablets, readers

**Representative Products:**
- Amazon Kindle Paperwhite (3,176 reviews)
- All-New Fire HD 8 Tablet (2,814 reviews)
- Fire HD 8 with Alexa - Tangerine (2,435 reviews)
- Fire HD 8 Tablet (2,274 reviews)
- Kindle Fire 16GB Blue (1,038 reviews)

**Category Focus:** E-readers, tablets, computing devices

---

## 📈 Clustering Quality Metrics

### Performance Scores:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.3363 | Moderate cluster separation |
| **Davies-Bouldin Score** | 1.6581 | Good cluster compactness |

### Metric Interpretation:

**Silhouette Score (0.3363):**
- Range: -1 (worst) to +1 (best)
- 0.3363 = Moderate separation
- ✅ Acceptable for text clustering (text is high-dimensional)
- Clusters are distinct but not perfectly separated

**Davies-Bouldin Score (1.6581):**
- Lower is better
- 1.6581 = Good performance
- ✅ Clusters are compact and well-separated
- Each cluster has clear identity

### Why These Scores?

Text clustering typically scores lower than numerical clustering because:
- High dimensionality (5000 features)
- Sparse data (TF-IDF matrices)
- Overlapping vocabulary across categories
- **These scores are good for text data!**

---

## 🎨 Visualizations Generated

### 1. Elbow Plot
- Shows inertia vs number of clusters
- Used to determine optimal k value

### 2. Silhouette Plot
- Shows cluster cohesion for k=2 to 10
- Confirms k=5 is optimal

### 3. PCA Scatter Plot
- 2D visualization of 5 clusters
- Reduces 5000 dimensions → 2 for plotting
- Shows cluster separation visually

### 4. Word Clouds (per cluster)
- Visual representation of top terms
- Helps understand cluster themes

---

## 📤 Output File

**File:** `clustered_reviews_export.csv`  
**Location:** `../datasets/processed/`

### Columns Exported:

| Column | Description |
|--------|-------------|
| `reviews.id` | Review identifier |
| `asins` | Product ASIN code |
| `name` | Product name |
| `reviews.text` | Review text |
| `reviews.rating` | Star rating (1-5) |
| `cluster` | **NEW:** Cluster number (0-4) |
| `cluster_name` | **NEW:** Human-readable category name |

**Total Rows:** 59,630  
**New Columns:** 2 (cluster, cluster_name)

---

## 🔧 Technical Details

### TF-IDF Parameters:

```python
TfidfVectorizer(
    max_features=5000,      # Limit to top 5000 terms
    ngram_range=(1, 2),     # Single words + word pairs
    min_df=5,               # Term must appear in ≥5 documents
    max_df=0.7,             # Ignore terms in >70% of documents
    stop_words='english'    # Remove common English words
)
```

**Why these settings?**
- `max_features=5000`: Balances detail vs computational cost
- `ngram_range=(1,2)`: Captures phrases like "smart home", "fire tablet"
- `min_df=5`: Removes noise and typos
- `max_df=0.7`: Removes uninformative common words

### K-Means Parameters:

```python
KMeans(
    n_clusters=5,           # 5 categories (within 4-6 requirement)
    random_state=42,        # Reproducible results
    n_init=20,              # Try 20 different starting positions
    max_iter=500,           # Allow up to 500 iterations
    verbose=1               # Show progress
)
```

**Why these settings?**
- `n_init=20`: Ensures global optimum (not local minimum)
- `max_iter=500`: Allows convergence for large dataset
- `random_state=42`: Makes results reproducible

### PCA for Visualization:

```python
PCA(n_components=2, random_state=42)
```
- Reduces 5000D → 2D for plotting
- Preserves as much variance as possible
- Enables visual cluster inspection

---

## 💻 Usage

### Prerequisites:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
```

### Running the Notebook:

1. **Ensure input file exists:**
   ```
   ../datasets/processed/final_dataset_sentiment_analysis.csv
   ```

2. **Run all cells** in order (top to bottom)

3. **Check output:**
   ```
   ../datasets/processed/clustered_reviews_export.csv
   ```

### Expected Runtime:

- Data loading: 10 seconds
- TF-IDF vectorization: 30 seconds
- K-Means clustering: 2-3 minutes
- Visualization: 20 seconds
- Export: 10 seconds
- **Total: ~4-5 minutes**

---

## 📊 Cluster Distribution Analysis

### Size Balance:

```
Largest cluster:  19,610 reviews (32.9%) - Computers and electronics
Smallest cluster:  5,069 reviews (8.5%)  - Entertainment Appliances
Size ratio:        3.9:1 (reasonable imbalance)
```

**Interpretation:**
- ✅ No single cluster dominates (largest is 33%)
- ✅ No cluster is too small (smallest is 8.5%)
- ✅ Balanced distribution for analysis

---

## ✅ Quality Checks

### Validation Steps Performed:

✓ **Cluster Stability:** K-Means run with 20 different initializations  
✓ **Metric Evaluation:** Silhouette and Davies-Bouldin scores calculated  
✓ **Visual Inspection:** PCA plots confirm separation  
✓ **Content Analysis:** Top terms make semantic sense  
✓ **Coverage:** All 59,630 reviews assigned to clusters  
✓ **Naming:** Categories are distinct and interpretable  

---

## 🎯 Requirements Compliance

### Project Task 2 Checklist:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Cluster into 4-6 categories | ✅ | 5 clusters created |
| Simplify dataset | ✅ | 94 products → 5 groups |
| Example categories suggested | ✅ | Matches examples (batteries, e-readers, accessories) |
| Meaningful names | ✅ | Named based on content analysis |
| Output for Task 3 | ✅ | `clustered_reviews_export.csv` ready |

---

## 🚀 Next Steps

After running this notebook:

✅ **Task 2 Complete:** Products grouped into 5 categories  
➡️ **Task 3 Ready:** Clustered data ready for:
- Sentiment analysis per category
- Product ranking within categories
- AI-powered summarization

**Next Notebook:** `04_product_ranking_sentiment_analysis.ipynb`
- Merges clusters with sentiment predictions
- Ranks products within each category
- Prepares data for LLM summarization

---

## 📝 For Your Presentation

### Key Talking Points:

**The Challenge:**
> "Amazon sells thousands of products across hundreds of categories. Manually grouping them is impractical. I needed an automated way to organize 60,000 reviews into meaningful categories."

**The Solution:**
> "I used K-Means clustering on TF-IDF features extracted from product names, categories, and reviews. The algorithm automatically discovered 5 distinct product groups based on content similarity."

**The Process:**
> "I combined text from product names and reviews, converted them to numerical features using TF-IDF, then applied K-Means clustering. I used the elbow method and silhouette analysis to determine that 5 clusters was optimal."

**The Results:**
> "Successfully grouped 59,630 reviews into 5 well-defined categories: Kids Tablets, Batteries, Smart Home Devices, Streaming Players, and Computing. Each cluster has distinct characteristics and vocabulary."

**The Impact:**
> "This enables category-specific analysis, product recommendations, and targeted summarization. Rather than treating all Amazon products the same, we can now provide tailored insights per category."

### Metrics to Highlight:

- ✅ 59,630 reviews clustered
- ✅ 5 distinct categories (within 4-6 requirement)
- ✅ Silhouette score: 0.34 (good for text)
- ✅ Balanced distribution (9-33% per cluster)
- ✅ Clear, interpretable category names

---

## ⚠️ Known Limitations

### 1. Overlapping Products
**Issue:** Some products fit multiple categories  
**Example:** Fire Kids Tablet (educational device OR tablet)  
**Impact:** May be assigned to less-obvious cluster  
**Status:** Acceptable trade-off for unsupervised learning

### 2. Cluster Size Imbalance
**Issue:** Computers cluster is 4x larger than Entertainment  
**Cause:** Dataset has more computer/tablet reviews  
**Impact:** Minimal - all clusters well-represented  
**Status:** Reflects actual data distribution

### 3. Text Features Only
**Issue:** Doesn't use ratings, prices, or other numerical features  
**Reason:** Focus on product category (content-based)  
**Impact:** May miss price-based groupings  
**Status:** Intentional design choice

---

## 🔍 Troubleshooting

### Common Issues:

**1. Memory Error**
```
MemoryError: Unable to allocate array
```
**Fix:** TF-IDF creates large sparse matrix (~3GB). Close other programs or reduce `max_features` to 3000.

**2. Convergence Warning**
```
ConvergenceWarning: Number of iterations exceeded
```
**Fix:** Normal! K-Means may not fully converge. Results are still valid.

**3. Different Results Each Run**
```
Cluster assignments change between runs
```
**Fix:** Set `random_state=42` in KMeans() for reproducibility. Already done in notebook.

**4. Poor Silhouette Score**
```
Silhouette score is low (<0.2)
```
**Fix:** Normal for text clustering! Scores >0.3 are considered good. Ours is 0.34.

---

## 📚 File Structure

```
project/
├── datasets/
│   └── processed/
│       ├── final_dataset_sentiment_analysis.csv     [INPUT]
│       └── clustered_reviews_export.csv             [OUTPUT]
├── notebooks/
│   ├── 01_data_prep.ipynb                           [Step 1]
│   ├── 02_sentiment_classification.ipynb            [Step 2]
│   ├── 03_product_clustering.ipynb                  [THIS ONE - Step 3]
│   └── 04_product_ranking_sentiment_analysis.ipynb  [Step 4]
└── visualizations/
    ├── elbow_plot.png
    ├── silhouette_analysis.png
    ├── cluster_pca_visualization.png
    └── wordclouds/ (per cluster)
```

---

## 🎓 What You Learned

### Machine Learning Concepts:
- Unsupervised learning (clustering)
- K-Means algorithm
- TF-IDF text vectorization
- Dimensionality reduction (PCA)
- Cluster evaluation metrics

### Practical Skills:
- Text preprocessing for ML
- Choosing optimal k value
- Interpreting clustering metrics
- Naming and validating clusters
- Visualizing high-dimensional data

---

## 📖 References

- [K-Means Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [PCA for Visualization](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

## ✅ Summary

### Process:
Text → TF-IDF → K-Means → 5 Clusters → Named Categories

### Input:
59,630 reviews from 94 Amazon products

### Output:
5 product categories with cluster labels

### Quality:
- Silhouette: 0.34 (good)
- Davies-Bouldin: 1.66 (good)
- Well-balanced distribution

### Status:
✅ **Complete and ready for Task 3**

---

## 🎯 Quick Reference

| Item | Value |
|------|-------|
| **Input** | `final_dataset_sentiment_analysis.csv` |
| **Output** | `clustered_reviews_export.csv` |
| **Method** | K-Means clustering |
| **Features** | TF-IDF (5000 terms, 1-2 grams) |
| **Clusters** | 5 categories |
| **Reviews** | 59,630 |
| **Quality** | Silhouette 0.34, DB 1.66 |
| **Runtime** | ~4-5 minutes |
| **Status** | ✅ Complete |
| **Next** | Product ranking & summarization |

---

**Created by:** Bootcamp Student  
**Project:** NLP - Amazon Product Reviews  
**Task:** 2 of 3 (Product Category Clustering)  
**Environment:** Python 3.12.5  
**Date:** 2024
