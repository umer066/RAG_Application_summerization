import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration - Set one of these paths
DATA_DIR = "skype 2025"  # Directory containing multiple .txt files
FILE_PATH = None  # Path to a single .txt file (Set this if embedding from a file)

VECTOR_DB_PATH = "vector_store.index"  # FAISS index storage path
METADATA_PATH = "metadata.json"  # JSON metadata storage
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded successfully.")

def load_chat_data():
    """
    Load chat data from a directory containing .txt files or from a single .txt file.
    """
    chat_data = []

    if FILE_PATH and os.path.isfile(FILE_PATH):  # Case 1: Input is a single file
        print(f"Loading chat data from file: {FILE_PATH}")
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    chat_data.append({
                        "text": line,
                        "source": os.path.basename(FILE_PATH)  # Store filename
                    })

    elif DATA_DIR and os.path.isdir(DATA_DIR):  # Case 2: Input is a directory
        print(f"Loading chat data from directory: {DATA_DIR}")
        for file_name in os.listdir(DATA_DIR):
            if file_name.endswith(".txt"):
                file_path = os.path.join(DATA_DIR, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            chat_data.append({
                                "text": line,
                                "source": file_name  # Store source file for reference
                            })

    else:
        print("Error: No valid input source found. Set either DATA_DIR or FILE_PATH.")
        exit(1)

    print(f"Total messages loaded: {len(chat_data)}")
    return chat_data

def main():
    # Load chat data
    chat_messages = load_chat_data()

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

    print("\nData processed, embeddings stored in FAISS.")

if __name__ == "__main__":
    main()
