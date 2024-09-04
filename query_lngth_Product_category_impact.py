import pandas as pd
import numpy as np
from scipy import stats

def categorize_query_length(query):
    words = query.split()
    if len(words) <= 2:
        return 'Short'
    elif len(words) <= 5:
        return 'Medium'
    else:
        return 'Long'

# Assuming df is your dataframe with search results
df['query_length_category'] = df['query'].apply(categorize_query_length)

# Calculate performance metrics for each query length category and search method
for length_category in ['Short', 'Medium', 'Long']:
    for search_method in ['Semantic', 'TF-IDF', 'Hybrid']:
        subset = df[(df['query_length_category'] == length_category) & 
                    (df['search_method'] == search_method)]
        
        hits_at_10 = subset['hits_at_10'].mean()
        mrr = subset['mrr'].mean()
        
        print(f"{length_category} queries, {search_method} search:")
        print(f"Hits@10: {hits_at_10:.4f}, MRR: {mrr:.4f}")

# Similar analysis for product categories
# You'd need to have product categories in your dataset

# Statistical testing
semantic_short = df[(df['query_length_category'] == 'Short') & 
                    (df['search_method'] == 'Semantic')]['mrr']
tfidf_short = df[(df['query_length_category'] == 'Short') & 
                 (df['search_method'] == 'TF-IDF')]['mrr']

t_stat, p_value = stats.ttest_ind(semantic_short, tfidf_short)
print(f"T-test for short queries, Semantic vs TF-IDF: p-value = {p_value:.4f}")