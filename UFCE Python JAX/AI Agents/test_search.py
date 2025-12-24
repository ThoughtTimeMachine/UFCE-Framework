import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
VECTOR_FILE = "knowledge_base/wiki_test_subset.dat"
META_FILE = "knowledge_base/wiki_test_subset_meta.txt"
EMBEDDING_DIM = 384

print("Loading Index...")
# 1. Load the Vectors (Memory Mapped - Instant)
vectors = np.memmap(VECTOR_FILE, dtype='float32', mode='r')
num_vectors = vectors.shape[0] // EMBEDDING_DIM
vectors = vectors.reshape((num_vectors, EMBEDDING_DIM))

# 2. Load the Text (Metadata)
print(f"Loading {num_vectors} text chunks...")
with open(META_FILE, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f]

# 3. Load Model
model = SentenceTransformer('all-MiniLM-L6-v2')

while True:
    query = input("\n🔎 Enter a search query (or 'q' to quit): ")
    if query.lower() == 'q': break
    
    # Encode & Search
    query_vec = model.encode([query])[0]
    scores = np.dot(vectors, query_vec) # Dot product = Cosine Similarity
    top_k_indices = np.argsort(scores)[-3:][::-1] # Get top 3
    
    print("\n--- Results ---")
    for idx in top_k_indices:
        print(f"[Score: {scores[idx]:.4f}] {texts[idx][:200]}...") # Print first 200 chars