# Semantic Search for Product Queries

This project implements and evaluates different semantic search approaches for product queries using the Amazon ESCI dataset. It explores three main methods: pure semantic search, secondary ranking with TF-IDF, and a hybrid approach combining both.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Approaches](#approaches)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)

## Project Overview

The goal of this project is to create a vector index for a product dataset and assess its quality against search queries. We implement and compare three different search approaches:

1. Semantic Search using FAISS
2. Secondary Ranking with TF-IDF
3. Hybrid Search combining semantic and TF-IDF approaches

We evaluate these methods using metrics such as Hits@N (for N=1, 5, 10) and Mean Reciprocal Rank (MRR).

## Dataset

We use the Amazon ESCI dataset, specifically focusing on the "small version" of the dataset to manage computational resources effectively. The dataset includes:

- Product information (title, description, etc.)
- Search queries
- Relevance labels (exact, substitute, complement, irrelevant)

## Approaches

### 1. Semantic Search using FAISS

- **Method**: We use SentenceTransformers to create embeddings for products and queries, then use FAISS for efficient similarity search.
- **Rationale**: This approach captures semantic meaning, allowing for matches beyond exact keyword matches. FAISS provides fast and efficient similarity search, crucial for large-scale applications.

### 2. Secondary Ranking with TF-IDF

- **Method**: We first perform a semantic search, then re-rank the results using TF-IDF similarity.
- **Rationale**: This combines the benefits of semantic understanding with the precision of keyword matching, potentially improving results for queries with specific terms.

### 3. Hybrid Search

- **Method**: We combine the scores from semantic search and TF-IDF similarity using a weighted average.
- **Rationale**: This approach aims to balance semantic understanding and keyword relevance, potentially providing more robust results across different query types.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/mahhos/semantic-search-project.git
   cd semantic-search-project
   ```

2. Install required packages:
   ```
   pip install pandas numpy matplotlib sentence-transformers scikit-learn faiss-cpu tqdm pyarrow
   ```

## Usage

Run the main script:

```
python semantic_search_application.py
```

This will:
1. Load and preprocess the data
2. Create embeddings and indexes
3. Evaluate all three search approaches
4. Save results to a text file
5. Generate and save a visualization of the results

## Results

The script will generate:
1. A text file with detailed metrics for each approach
2. A PNG image showing a comparison of Hits@N performance
3. Printed MRR scores for each method


### Interpreting the Results

#### Hits@N Comparison Plot

The `hits_at_n_comparison.png` shows a bar chart comparing the Hits@N performance of all three search methods (Semantic Search, Secondary Ranking, and Hybrid Search) for N = 1, 5, and 10. Higher bars indicate better performance.

![Hits@N Comparison](hits_at_n_comparison.png)

#### MRR Comparison Plot

The `mrr_comparison.png` shows a bar chart comparing the Mean Reciprocal Rank (MRR) scores for all three search methods. Higher bars indicate better performance.

![MRR Comparison](mrr_comparison.png)

#### Results Table

The `results_table.csv` file contains a summary of all metrics. Here's a sample of what it might look like:

| Method | Hits@1 | Hits@5 | Hits@10 | MRR |
|--------|--------|--------|---------|-----|
| Semantic Search | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| Secondary Ranking | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| Hybrid Search | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |


### Interpreting the Metrics

- **Hits@N**: The proportion of queries where the correct product was in the top N results. Higher is better.
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the correct products in the search results. Higher is better.

## Conclusion

Based on these results, we can determine which method performs best for our specific use case. Typically, we're looking for:

1. Higher Hits@N scores, especially for lower N values
2. Higher MRR scores





## Future Work

- Experiment with different embedding models
- Implement more advanced indexing techniques
- Incorporate user behavior data for personalized ranking
- Explore other hybrid approaches or ensemble methods
