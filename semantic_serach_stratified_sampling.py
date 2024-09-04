import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from sklearn.model_selection import train_test_split

# Load the datasets
print("Loading datasets...")
url_examples = 'https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
url_products = 'https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet'

df_examples = pd.read_parquet(url_examples)
df_products = pd.read_parquet(url_products)

df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale', 'product_id'],
    right_on=['product_locale', 'product_id']
)

# Use only the small version of the dataset
df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
df_train = df_task_1[df_task_1["split"] == "train"]
df_test = df_task_1[df_task_1["split"] == "test"]

print(f"Training data shape: {df_train.shape}")
print(f"Testing data shape: {df_test.shape}")


# Stratified sampling function
def stratified_sample(df, sample_size, random_state=42):
    if len(df) <= sample_size:
        return df

    _, sampled_df = train_test_split(df,
                                     test_size=sample_size,
                                     stratify=df['esci_label'],
                                     random_state=random_state)
    return sampled_df


# Uncomment these lines if you need to further reduce the dataset size
df_train = stratified_sample(df_train, sample_size=50000)
df_test = stratified_sample(df_test, sample_size=50000)

# Embedding function
print("Loading SentenceTransformer model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def create_embeddings(texts):
    return model.encode(texts, show_progress_bar=True)


# Create product embeddings
print("Creating product embeddings...")
product_texts = df_train['product_title'] + ' ' + df_train['product_description'].fillna('')
product_embeddings = create_embeddings(product_texts.tolist())

# Create FAISS index
print("Creating FAISS index...")
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(product_embeddings.astype('float32'))


# Search function using FAISS
def faiss_search(query, k=10):
    query_embedding = create_embeddings([query])
    _, I = index.search(query_embedding.astype('float32'), k)
    return df_train.iloc[I[0]]['product_id'].tolist()


# Evaluation metrics
def calculate_hits_at_n(predictions, actual, n):
    return 1.0 if actual in predictions[:n] else 0.0


def calculate_mrr(predictions, actual):
    try:
        rank = predictions.index(actual) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


# Evaluate search performance
def evaluate_search(df, n_values):
    hits_at_n = {n: [] for n in n_values}
    mrr_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        actual_product_id = row['product_id']
        predictions = faiss_search(query, max(n_values))

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predictions, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predictions, actual_product_id))

    hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
    mrr_avg = np.mean(mrr_scores)

    return hits_at_n_avg, mrr_avg


# Secondary ranking logic using TF-IDF
print("Creating TF-IDF vectorizer...")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(product_texts)


def secondary_ranking(query, initial_results, k=10):
    query_vector = tfidf.transform([query])

    # Filter out any indices that are out of bounds
    valid_indices = [idx for idx in initial_results if idx < tfidf_matrix.shape[0]]

    if not valid_indices:
        print(f"Warning: No valid indices found for query: {query}")
        return []

    similarities = cosine_similarity(query_vector, tfidf_matrix[valid_indices]).flatten()
    reranked_indices = np.argsort(similarities)[::-1][:k]
    return [valid_indices[i] for i in reranked_indices]


# Update the evaluate_search_with_secondary function
def evaluate_search_with_secondary(df, n_values):
    hits_at_n = {n: [] for n in n_values}
    mrr_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        actual_product_id = row['product_id']
        initial_results = faiss_search(query, max(n_values) * 2)
        predictions = secondary_ranking(query, initial_results, max(n_values))

        if not predictions:
            continue  # Skip this query if no valid predictions

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predictions, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predictions, actual_product_id))

    hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
    mrr_avg = np.mean(mrr_scores)

    return hits_at_n_avg, mrr_avg
# def secondary_ranking(query, initial_results, k=10):
#     query_vector = tfidf.transform([query])
#     similarities = cosine_similarity(query_vector, tfidf_matrix[initial_results]).flatten()
#     reranked_indices = np.argsort(similarities)[::-1][:k]
#     return [initial_results[i] for i in reranked_indices]
#
#
# # Evaluate with secondary ranking
# def evaluate_search_with_secondary(df, n_values):
#     hits_at_n = {n: [] for n in n_values}
#     mrr_scores = []
#
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         query = row['query']
#         actual_product_id = row['product_id']
#         initial_results = faiss_search(query, max(n_values) * 2)
#         predictions = secondary_ranking(query, initial_results, max(n_values))
#
#         for n in n_values:
#             hits_at_n[n].append(calculate_hits_at_n(predictions, actual_product_id, n))
#
#         mrr_scores.append(calculate_mrr(predictions, actual_product_id))
#
#     hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
#     mrr_avg = np.mean(mrr_scores)
#
#     return hits_at_n_avg, mrr_avg


# Hybrid search function
def hybrid_search(query, k=10, alpha=0.5):
    query_embedding = create_embeddings([query])
    _, I = index.search(query_embedding.astype('float32'), k * 2)
    semantic_results = I[0]

    query_vector = tfidf.transform([query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix[semantic_results]).flatten()

    combined_scores = alpha * (1 - np.arange(len(semantic_results)) / len(semantic_results)) + (
                1 - alpha) * tfidf_scores

    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    return df_train.iloc[semantic_results[top_k_indices]]['product_id'].tolist()


# Evaluate hybrid search
def evaluate_hybrid_search(df, n_values, alpha=0.5):
    hits_at_n = {n: [] for n in n_values}
    mrr_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        actual_product_id = row['product_id']
        predictions = hybrid_search(query, k=max(n_values), alpha=alpha)

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predictions, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predictions, actual_product_id))

    hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
    mrr_avg = np.mean(mrr_scores)

    return hits_at_n_avg, mrr_avg


# Function to save results to file
def save_results_to_file(filename, results_dict):
    with open(filename, 'w') as f:
        for method, metrics in results_dict.items():
            f.write(f"{method} Results:\n")
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"  {metric}:\n")
                    for n, score in value.items():
                        f.write(f"    {n}: {score:.4f}\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")


# Evaluate all three approaches
print("Evaluating semantic search performance...")
n_values = [5]
hits_at_n_semantic, mrr_semantic = evaluate_search(df_test, n_values)

print("Evaluating secondary ranking performance...")
hits_at_n_secondary, mrr_secondary = evaluate_search_with_secondary(df_test, n_values)

print("Evaluating hybrid search performance...")
hits_at_n_hybrid, mrr_hybrid = evaluate_hybrid_search(df_test, n_values, alpha=0.5)

# Prepare results dictionary
results = {
    "Semantic Search": {"Hits@N": hits_at_n_semantic, "MRR": mrr_semantic},
    "Secondary Ranking": {"Hits@N": hits_at_n_secondary, "MRR": mrr_secondary},
    "Hybrid Search": {"Hits@N": hits_at_n_hybrid, "MRR": mrr_hybrid}
}

# Generate filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"search_results_{timestamp}.txt"

# Save results to file
save_results_to_file(filename, results)
print(f"Results saved to {filename}")

# Visualize results
plt.figure(figsize=(12, 6))
x = list(hits_at_n_semantic.keys())
width = 0.25

plt.bar([i - width for i in x], list(hits_at_n_semantic.values()), width, label='Semantic Search')
plt.bar(x, list(hits_at_n_secondary.values()), width, label='Secondary Ranking')
plt.bar([i + width for i in x], list(hits_at_n_hybrid.values()), width, label='Hybrid Search')

plt.title('Hits@N Performance Comparison')
plt.xlabel('N')
plt.ylabel('Score')
plt.legend()
plt.xticks(x)
plt.savefig(f'performance_comparison_{timestamp}.png')
plt.show()

print("\nMRR Comparison:")
print(f"Semantic Search MRR: {mrr_semantic:.4f}")
print(f"Secondary Ranking MRR: {mrr_secondary:.4f}")
print(f"Hybrid Search MRR: {mrr_hybrid:.4f}")

# Save MRR comparison to file
with open(filename, 'a') as f:
    f.write("\nMRR Comparison:\n")
    f.write(f"Semantic Search MRR: {mrr_semantic:.4f}\n")
    f.write(f"Secondary Ranking MRR: {mrr_secondary:.4f}\n")
    f.write(f"Hybrid Search MRR: {mrr_hybrid:.4f}\n")

print(f"MRR comparison appended to {filename}")