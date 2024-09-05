import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime

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


def handle_missing_data(df):
    # Handle missing product descriptions
    df['processed_description'] = df['product_description'].fillna(df['product_title'])
    
    # Handle missing attributes
    for attr in ['brand', 'color', 'size']:
        df[attr] = df[attr].fillna('Unknown')
    
    # Create flags for missing data
    df['has_description'] = df['product_description'].notna().astype(int)
    df['has_price'] = df['price'].notna().astype(int)
    
    # Handle missing numerical values
    df['price'] = df.groupby('category')['price'].transform(lambda x: x.fillna(x.median()))
    
    return df

# Apply the handling
df_processed = handle_missing_data(df_examples_products)



df_task_1 = df_processed[df_processed["small_version"] == 1]
df_task_1_train = df_task_1[df_task_1["split"] == "train"]
df_task_1_test = df_task_1[df_task_1["split"] == "test"]

# Use only 10% of the data
df_train = df_task_1_train.sample(frac=1, random_state=42)
df_test = df_task_1_test.sample(frac=1, random_state=42)

print(f"Training data shape: {df_train.shape}")
print(f"Testing data shape: {df_test.shape}")

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


# Search function
def search(query, k=10):
    query_embedding = create_embeddings([query])
    D, I = index.search(query_embedding.astype('float32'), k)
    return I[0]


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
        predictions = search(query, max(n_values))
        predicted_product_ids = df_train.iloc[predictions]['product_id'].tolist()

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predicted_product_ids, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predicted_product_ids, actual_product_id))

    hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
    mrr_avg = np.mean(mrr_scores)

    return hits_at_n_avg, mrr_avg


# TF-IDF ranking logic using TF-IDF
print("Creating TF-IDF vectorizer...")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(product_texts)


def secondary_ranking(query, initial_results, k=10):
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix[initial_results]).flatten()
    reranked_indices = np.argsort(similarities)[::-1][:k]
    return [initial_results[i] for i in reranked_indices]


# Evaluate with secondary ranking
def evaluate_search_with_secondary(df, n_values):
    hits_at_n = {n: [] for n in n_values}
    mrr_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        actual_product_id = row['product_id']
        initial_results = search(query, max(n_values) * 2)
        predictions = secondary_ranking(query, initial_results, max(n_values))
        predicted_product_ids = df_train.iloc[predictions]['product_id'].tolist()

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predicted_product_ids, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predicted_product_ids, actual_product_id))

    hits_at_n_avg = {n: np.mean(scores) for n, scores in hits_at_n.items()}
    mrr_avg = np.mean(mrr_scores)

    return hits_at_n_avg, mrr_avg


# Hybrid search function
def hybrid_search(query, k=10, alpha=0.5):
    # Semantic search
    query_embedding = create_embeddings([query])
    D_semantic, I_semantic = index.search(query_embedding.astype('float32'), k * 2)

    # TF-IDF search
    query_tfidf = tfidf.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Combine scores
    combined_scores = alpha * (1 - D_semantic.flatten() / np.max(D_semantic)) + (1 - alpha) * tfidf_scores[
        I_semantic.flatten()]

    # Get top k results
    top_k_indices = np.argsort(combined_scores)[-k:][::-1]

    return I_semantic.flatten()[top_k_indices]


# Evaluate hybrid search
def evaluate_hybrid_search(df, n_values, alpha=0.5):
    hits_at_n = {n: [] for n in n_values}
    mrr_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        actual_product_id = row['product_id']
        predictions = hybrid_search(query, k=max(n_values), alpha=alpha)
        predicted_product_ids = df_train.iloc[predictions]['product_id'].tolist()

        for n in n_values:
            hits_at_n[n].append(calculate_hits_at_n(predicted_product_ids, actual_product_id, n))

        mrr_scores.append(calculate_mrr(predicted_product_ids, actual_product_id))

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


# Evaluate on a very small subset of test data
# sample_size = 1000  # You can adjust this number
# df_test = df_test.sample(n=sample_size, random_state=42)



# Evaluate all three approaches
print("Evaluating semantic search performance...")
n_values = [1, 5, 10]
hits_at_n_semantic, mrr_semantic = evaluate_search(df_test, n_values)

print("Evaluating TF-IDF ranking performance...")
hits_at_n_secondary, mrr_secondary = evaluate_search_with_secondary(df_test, n_values)

print("Evaluating hybrid search performance...")
hits_at_n_hybrid, mrr_hybrid = evaluate_hybrid_search(df_test, n_values, alpha=0.5)

# Prepare results dictionary
results = {
    "Semantic Search": {"Hits@N": hits_at_n_semantic, "MRR": mrr_semantic},
    "TF-IDF Ranking": {"Hits@N": hits_at_n_secondary, "MRR": mrr_secondary},
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
plt.bar(x, list(hits_at_n_secondary.values()), width, label='TF-IDF Ranking')
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
print(f"TD-IDF Ranking MRR: {mrr_secondary:.4f}")
print(f"Hybrid Search MRR: {mrr_hybrid:.4f}")

# Save MRR comparison to file
with open(filename, 'a') as f:
    f.write("\nMRR Comparison:\n")
    f.write(f"Semantic Search MRR: {mrr_semantic:.4f}\n")
    f.write(f"TD-IDF Ranking MRR: {mrr_secondary:.4f}\n")
    f.write(f"Hybrid Search MRR: {mrr_hybrid:.4f}\n")

print(f"MRR comparison appended to {filename}")
