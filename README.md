# Amazon Product Review Analysis & Summarization

An end-to-end NLP pipeline that processes 60,000+ Amazon product reviews to generate AI-powered product recommendations and review summaries.

## 🎯 Project Overview

This project uses machine learning and large language models to automatically analyze Amazon product reviews, classify sentiment, cluster products into categories, and generate professional blog-style product recommendations.

**Key Features:**
- ✅ Sentiment classification with 94% accuracy
- ✅ Automated product clustering into 5 categories
- ✅ Quality scoring system combining sentiment + ratings
- ✅ AI-generated review summaries using GPT
- ✅ Identifies top 3 products and worst product per category

## 📊 Dataset

**Source:** [Datafiniti Amazon Product Reviews (Kaggle)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

**Stats:**
- 59,630 product reviews
- 94 unique Amazon products (Fire tablets, Kindle, Echo, AmazonBasics)
- 8 brands
- Rating range: 1-5 stars

## 🔄 Project Workflow

```
Raw Data (3 CSV files)
    ↓
01. Data Preparation
    ↓
02. Sentiment Classification (TF-IDF + LinearSVC)
    ↓
03. Product Clustering (K-Means)
    ↓
04. Product Ranking & Quality Scoring
    ↓
05. AI-Powered Review Summarization (GPT)
    ↓
Final Output (Blog-style articles)
```

## 📁 Repository Structure

```
amazon-review-analysis/
├── notebooks/
│   ├── 01_data_prep.ipynb                          # Data cleaning & merging
│   ├── 02_Tf_idf_linearsvc_ratings_supervised.ipynb # Sentiment classification
│   ├── 03_clustering.ipynb                         # Product clustering
│   ├── 04_product_ranking_sentiment_analysis.ipynb # Quality scoring
│   └── 05_review_summarization_LLM_API.ipynb       # AI summarization
├── datasets/
│   ├── raw/                                        # Original CSV files
│   └── processed/                                  # Cleaned & enriched data
├── generated_summaries/                            # AI-generated articles
├── README_files/                                   # Detailed documentation
│   ├── 01_DATA_PREP_README.md
│   ├── 02_SENTIMENT_ANALYSIS_README.md
│   ├── 03_CLUSTERING_README.md
│   ├── 04_PRODUCT_RANKING_README.md
│   └── 05_Summarization_README.md
├── .env                                            # API keys (not tracked)
└── README.md                                       # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.12.5 or compatible
pip install pandas numpy scikit-learn matplotlib seaborn openai python-dotenv
```

### Running the Pipeline

**Step 1: Data Preparation**
```bash
# Run: 01_data_prep.ipynb
# Input: 3 raw CSV files (~/datasets/raw/)
# Output: final_dataset_sentiment_analysis.csv
```

**Step 2: Sentiment Classification**
```bash
# Run: 02_Tf_idf_linearsvc_ratings_supervised.ipynb
# Input: final_dataset_sentiment_analysis.csv
# Output: sentiment_enriched_reviews.csv
# Model: TF-IDF + LinearSVC (94% accuracy)
```

**Step 3: Product Clustering**
```bash
# Run: 03_clustering.ipynb
# Input: sentiment_enriched_reviews.csv
# Output: clustered_reviews_export.csv
# Method: K-Means clustering (5 categories)
```

**Step 4: Product Ranking**
```bash
# Run: 04_product_ranking_sentiment_analysis.ipynb
# Input: clustered_reviews_export.csv + sentiment_enriched_reviews.csv
# Output: product_summary_data.json
# Generates quality scores and rankings
```

**Step 5: AI Summarization**
```bash
# Run: 05_review_summarization_LLM_API.ipynb
# Input: product_summary_data.json
# Output: Category-specific review articles
# Requires: OpenAI API key in .env file
```

## 📈 Key Results

### Sentiment Classification
- **Accuracy:** 93.85%
- **F1 Score (weighted):** 0.934
- **Model:** TF-IDF + LinearSVC
- **Training Time:** ~3 minutes

### Product Clustering
- **Categories:** 5 product groups
- **Products:** 114 analyzed
- **Reviews:** 52,870 processed

### Product Rankings
| Category | Products | Avg Quality Score |
|----------|----------|-------------------|
| Amazon devices & accessories | 19 | 0.945 |
| Entertainment Appliances | 3 | 0.991 |
| Kids Toys & entertainment | 7 | 0.944 |
| Batteries & essentials | 2 | 0.967 |
| Computers & electronics | 83 | 0.928 |

## 🔧 Technologies Used

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn
- **NLP:** TF-IDF vectorization
- **Clustering:** K-Means
- **Classification:** LinearSVC (Support Vector Machine)
- **AI Generation:** OpenAI GPT-3.5/GPT-4
- **Visualization:** matplotlib, seaborn

## 📊 Quality Score Formula

Combines sentiment analysis with traditional ratings:

```
quality_score = 0.6 × normalized_sentiment + 0.4 × normalized_rating

Where:
  sentiment_score = (positive × 1.0 + neutral × 0.5 + negative × -1.0) / total
  normalized_sentiment = (sentiment_score + 1) / 2
  normalized_rating = avg_rating / 5.0
```

## 📝 Output Examples

### Generated Summary Structure
Each category receives a 600-800 word article including:
- Category overview
- Top 3 product analysis with pros/cons
- Side-by-side comparison
- Product to avoid (with complaints)
- Final recommendations

### Output Formats
- Individual `.txt` files per category
- Combined document with all summaries
- Structured JSON for API integration

## 🎓 Learning Objectives

This project demonstrates:
- ✅ End-to-end NLP pipeline design
- ✅ Supervised machine learning (sentiment classification)
- ✅ Unsupervised learning (clustering)
- ✅ Feature engineering (TF-IDF)
- ✅ Model evaluation and metrics
- ✅ LLM integration and prompt engineering
- ✅ Data cleaning and preprocessing
- ✅ Handling class imbalance

## ⚠️ Known Limitations

1. **Class Imbalance:** 92% positive reviews (realistic for Amazon)
   - Impact: Neutral review detection less accurate (34% F1)
   - Mitigation: Used balanced class weights

2. **Missing Data:** 11% of reviews lack product names
   - Impact: Excluded from final analysis
   - Acceptable for project scope

3. **Sample Size:** Some products have <10 reviews
   - Mitigation: Applied minimum 3-review threshold

## 💰 Cost Considerations

**OpenAI API Usage:**
- GPT-3.5-turbo: ~$0.05 for 5 categories
- GPT-4: ~$0.50 for 5 categories

Actual costs depend on prompt length and number of categories.

## 📚 Detailed Documentation

Each notebook has comprehensive documentation:
- [01_DATA_PREP_README.md](README_files/01_DATA_PREP_README.md) - Data cleaning process
- [02_SENTIMENT_ANALYSIS_README.md](README_files/02_SENTIMENT_ANALYSIS_README.md) - ML model details
- [03_CLUSTERING_README.md](README_files/03_CLUSTERING_README.md) - Clustering methodology
- [04_PRODUCT_RANKING_README.md](README_files/04_PRODUCT_RANKING_README.md) - Ranking system
- [05_Summarization_README.md](README_files/05_Summarization_README.md) - LLM integration

## 🔒 Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

**Note:** Never commit `.env` to version control!

## 🐛 Troubleshooting

### Common Issues

**FileNotFoundError:**
- Ensure you run notebooks in sequence (01 → 02 → 03 → 04 → 05)
- Check file paths match your directory structure

**Memory Error:**
- Close other programs
- Requires ~2-3GB RAM for full pipeline

**API Rate Limit:**
- Add delays between API calls: `time.sleep(1)`
- Consider upgrading OpenAI API tier

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Total reviews processed | 59,630 |
| Sentiment accuracy | 93.85% |
| Merge success rate | 99.9% |
| Categories identified | 5 |
| Products ranked | 114 |
| Total pipeline runtime | ~15 minutes |

## 🤝 Contributing

This is a bootcamp project, but feedback is welcome! Feel free to:
- Report bugs via Issues
- Suggest improvements
- Fork and experiment

## 📄 License

[Specify your license - MIT, Apache 2.0, etc.]

## 👨‍💻 Author

[Hrishikesh Reddy Sanivarapu]  
AI Engineering Bootcamp Project  
[[Link to Linkedin](https://www.linkedin.com/in/hrishikesh-reddy-sanivarapu-046512125/)]

## 🙏 Acknowledgments

- Datafiniti for the Amazon review dataset
- Kaggle for hosting the data
- OpenAI for GPT API access
- 

## 📧 Contact

Questions or suggestions? Open an issue or reach out:
- Email: [s.hrishikeshreddy@gmail.com]]

---

**Last Updated:** February 2026  
**Project Status:** ✅ Complete  
**Python Version:** 3.12.5
