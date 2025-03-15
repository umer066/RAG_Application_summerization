import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = "skype 2025"  # Directory containing chat .txt files
VECTOR_DB_PATH = "vector_store.index"  # FAISS index storage path
METADATA_PATH = "metadata.json"  # JSON metadata storage
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded successfully.")

# Function to load and clean chat data
def load_chat_data(data_dir):
    chat_data = []
    print(f"Loading chat data from: {data_dir}")
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        chat_data.append({
                            "text": line,
                            "source": file_name  # Store source file for reference
                        })
    print(f"Total messages loaded: {len(chat_data)}")
    return chat_data

# Load and clean chat data
chat_messages = load_chat_data(DATA_DIR)

# Generate embeddings and store metadata
print("Generating embeddings...")
embeddings = []
metadata = []

for idx, chat in tqdm(enumerate(chat_messages), total=len(chat_messages), desc="Embedding Progress"):
    embedding = model.encode(chat["text"]).tolist()
    embeddings.append(embedding)
    metadata.append({"id": idx, "text": chat["text"], "source": chat["source"]})

# Convert to numpy array
embedding_matrix = np.array(embeddings, dtype=np.float32)

# Debug: Print embedding matrix shape
print("Embedding matrix shape:", embedding_matrix.shape)

# Create FAISS index
d = embedding_matrix.shape[1]  # Ensure correct dimension
print("FAISS index dimension:", d)

index = faiss.IndexFlatL2(d)
print("FAISS index created successfully.")

# Add embeddings to FAISS
index.add(embedding_matrix)
print("Embeddings added to FAISS index.")

# Save FAISS index
faiss.write_index(index, VECTOR_DB_PATH)
print(f"FAISS index saved at: {VECTOR_DB_PATH}")

# Save metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved at: {METADATA_PATH}")

print("\n✅ Data processed, embeddings stored in FAISS.")
