# Review Summarization with LLM API

This Jupyter notebook demonstrates an automated product review summarization system using OpenAI's GPT API. It generates professional, consumer-focused product reviews and comparisons across multiple Amazon product categories.

## Overview

The notebook processes structured product review data and uses large language models to generate comprehensive, blog-style articles comparing top products within each category. It outputs summaries in multiple formats (text files, combined document, and JSON) for easy integration into content management systems or blogs.

## Features

- **Automated Review Summarization**: Generates 600-800 word product review articles using GPT models
- **Multi-Category Support**: Processes multiple product categories in batch
- **Structured Prompts**: Uses carefully crafted prompts to generate consistent, high-quality content
- **Multiple Output Formats**: Saves summaries as individual text files, combined document, and JSON
- **Professional Writing Style**: Generates content in the style of consumer tech blogs (like Wirecutter or The Verge)
- **Data-Driven Insights**: Incorporates actual review statistics, ratings, and sentiment analysis

## Project Structure

```
project_root/
├── notebooks/
│   └── 05_review_summarization_LLM_API.ipynb  # This notebook
├── datasets/
│   └── processed/
│       └── product_summary_data.json          # Input data
├── generated_summaries/                       # Output directory
│   ├── [category_name].txt                   # Individual summaries
│   ├── all_category_summaries.txt            # Combined document
│   └── summaries.json                         # JSON export
└── .env                                       # API credentials (not tracked)
```

## Requirements

### Dependencies

```python
openai>=1.0.0
python-dotenv>=1.0.0
```

### Python Version
- Python 3.12.5 or compatible

### API Access
- OpenAI API key (requires active OpenAI account)

## Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
   ```bash
   pip install openai python-dotenv
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Prepare input data**:
   Ensure `product_summary_data.json` exists in `datasets/processed/`

## Input Data Format

The notebook expects a JSON file with the following structure:

```json
[
  {
    "category": "Category Name",
    "top_3_products": [
      {
        "name": "Product Name",
        "asin": "PRODUCT_ASIN",
        "quality_score": 1.05,
        "avg_rating": 4.67,
        "total_reviews": 15,
        "positive_pct": 100.0,
        "negative_pct": 0.0,
        "neutral_pct": 0.0,
        "positive_reviews": ["Review text 1", "Review text 2"],
        "negative_reviews": ["Complaint text 1"]
      }
    ],
    "worst_product": {
      "name": "Product to Avoid",
      "avg_rating": 3.5,
      "total_reviews": 100,
      "quality_score": 0.75,
      "negative_pct": 25.0,
      "negative_reviews": ["Common complaint 1", "Common complaint 2"]
    },
    "category_metadata": {
      "total_products": 10,
      "top_label": "Top 3"
    }
  }
]
```

## Usage

### Running the Notebook

Execute cells sequentially:

1. **Import Libraries** - Loads required dependencies
2. **Configure API** - Initializes OpenAI client with your API key
3. **Load Data** - Reads product summary data from JSON file
4. **Helper Functions** - Defines formatting and prompt generation utilities
5. **Configure Model** - Sets GPT model and parameters
6. **Generate Summaries** - Creates review articles for each category
7. **Display Results** - Shows generated content in notebook
8. **Save Outputs** - Exports summaries in multiple formats

### Key Configuration

```python
MODEL = "gpt-3.5-turbo"  # or "gpt-4", "gpt-4-turbo"
TEMPERATURE = 0.7         # Controls creativity (0.0-1.0)
```

### Output Files

After execution, the notebook generates:

1. **Individual category files**: `generated_summaries/[category_name].txt`
2. **Combined document**: `generated_summaries/all_category_summaries.txt`
3. **JSON export**: `generated_summaries/summaries.json`

## Generated Content Structure

Each generated article includes:

### Introduction
- Category overview
- Consumer relevance

### Top Products Analysis
For each product:
- Customer favorites (from positive reviews)
- Key features and strengths
- Common concerns (from negative reviews)
- Use case recommendations

### Comparison Section
- Side-by-side feature comparison
- Price-performance analysis
- Best use cases for each product

### Product to Avoid
- Why it's rated poorly
- Common complaints
- Alternative recommendations

### Final Recommendation
- Overall winner
- Budget picks
- Specific use case suggestions

## Workflow

```
┌─────────────────────┐
│  Load JSON Data     │
│  (product_summary_  │
│   data.json)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Format Product     │
│  Information        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create Structured  │
│  Prompts for LLM    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Call OpenAI API    │
│  (GPT-3.5/GPT-4)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Generate           │
│  600-800 Word       │
│  Articles           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Save Outputs       │
│  (TXT/JSON)         │
└─────────────────────┘
```

## Sample Categories

The example dataset includes:
- Computers and electronics
- Amazon devices & accessories
- Entertainment Appliances
- Kids Toys & kids entertainment
- Batteries and household essentials

## Cost Considerations

- **GPT-3.5-turbo**: ~$0.001 per 1K tokens (input) / $0.002 per 1K tokens (output)
- **GPT-4**: ~$0.03 per 1K tokens (input) / $0.06 per 1K tokens (output)

For 5 categories with ~1000 tokens per prompt:
- GPT-3.5: ~$0.05 total
- GPT-4: ~$0.50 total

## Customization

### Modifying the Article Structure

Edit the `create_summarization_prompt()` function to adjust:
- Article length (default: 600-800 words)
- Writing style and tone
- Section structure
- Comparison format

### Changing the Model

```python
MODEL = "gpt-4"  # For higher quality
# or
MODEL = "gpt-4-turbo"  # For faster processing
```

### Adjusting Creativity

```python
TEMPERATURE = 0.9  # More creative/varied
# or
TEMPERATURE = 0.3  # More focused/consistent
```

## Error Handling

The notebook includes error handling for:
- Missing API keys
- File not found errors
- API request failures
- Rate limiting

## Limitations

- Requires active internet connection
- API rate limits apply
- Costs vary based on model and usage
- Generated content should be reviewed for accuracy
- Quality depends on input data completeness

## Best Practices

1. **Review Generated Content**: Always review AI-generated content before publishing
2. **Monitor Costs**: Track API usage to manage expenses
3. **Version Control**: Keep track of prompt changes for consistency
4. **Input Data Quality**: Better input data leads to better summaries
5. **Rate Limiting**: Add delays between API calls if processing many categories

## Troubleshooting

### Common Issues

**API Key Error**:
```
ValueError: OPENAI_API_KEY not found in .env
```
- Solution: Verify `.env` file exists in project root with correct key

**File Not Found**:
```
FileNotFoundError: product_summary_data.json
```
- Solution: Ensure data file is in `../datasets/processed/` relative to notebook

**Rate Limit Error**:
```
RateLimitError: Rate limit exceeded
```
- Solution: Add `time.sleep(1)` between API calls or upgrade API tier

## Future Enhancements

Potential improvements:
- [ ] Add support for multiple LLM providers (Anthropic Claude, etc.)
- [ ] Implement batch processing with progress tracking
- [ ] Add HTML/Markdown output formatting
- [ ] Include automated fact-checking
- [ ] Generate SEO-optimized titles and meta descriptions
- [ ] Add image generation integration
- [ ] Create comparison tables automatically
- [ ] Support for multilingual summaries

## License

[Specify your license here]

## Author

[Your name/organization]

## Acknowledgments

- OpenAI for GPT API
- Product review data sources
- Amazon product review ecosystem

## Support

For issues or questions:
- Check the troubleshooting section
- Review OpenAI API documentation
- Submit an issue in the project repository

---

**Last Updated**: February 2026  
**Notebook Version**: 1.0  
**Compatible with**: OpenAI API v1.0+
