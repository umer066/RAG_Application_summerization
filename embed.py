import os
import re
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import dateutil.parser

# Configuration - Set one of these paths
DATA_DIR = "skype 2025"  # Folder containing .txt files
FILE_PATH = None  # Set this if processing a single file

VECTOR_DB_PATH = "vector_store.index"  # FAISS index storage path
METADATA_PATH = "metadata.json"  # JSON metadata storage
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
print("Model loaded successfully.")

def is_system_message(text):
    """ Dynamically detect system messages. """
    system_keywords = [
        "added", "removed", "left the conversation",
        "joined the group", "enabled joining", "has renamed",
        "has made the chat history visible", "Copy invite link",
        "Group members:", "changed the group name", "Go to Group Settings"
    ]
    return any(keyword.lower() in text.lower() for keyword in system_keywords)

# def is_date_or_time(text):
#     """ Detect timestamps or dates. """
#     try:
#         dateutil.parser.parse(text, fuzzy=True)
#         return True
#     except ValueError:
#         return False

def is_date_or_time(text):
    """ Detect timestamps or dates while preventing false positives and overflow errors. """
    text = text.strip()

    # Skip if the text is too long to be a date
    if len(text) > 50:  # Most date/time strings are short
        return False

    # Skip if the text contains too many digits (likely not a valid date)
    if sum(c.isdigit() for c in text) > 8:  # More than 8 digits likely not a date
        return False

    try:
        parsed_date = dateutil.parser.parse(text, fuzzy=True)

        # Ensure parsed_date is not too large (prevent overflow)
        if parsed_date.year > 2100 or parsed_date.year < 1900:
            return False

        return True
    except (ValueError, OverflowError):
        return False

def clean_text(lines):
    """ Remove system messages and intelligently associate timestamps with messages. """
    cleaned_data = []
    last_timestamp = None

    for line in lines:
        line = line.strip()
        
        if not line:
            continue

        if is_system_message(line):
            continue

        if is_date_or_time(line):  # Store timestamp for next message
            last_timestamp = line
            continue
        
        # Attach last timestamp to this message if exists
        if last_timestamp:
            line = f"{last_timestamp} - {line}"
            last_timestamp = None  # Reset timestamp after attaching it

        cleaned_data.append(line)

    return cleaned_data

def load_chat_data():
    """ Load chat data from a directory or a single file after cleaning. """
    chat_data = []

    if FILE_PATH and os.path.isfile(FILE_PATH):  # Case 1: Single file
        print(f"Loading chat data from file: {FILE_PATH}")
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            cleaned_lines = clean_text(lines)
            chat_data.extend([{"text": line, "source": os.path.basename(FILE_PATH)} for line in cleaned_lines])

    elif DATA_DIR and os.path.isdir(DATA_DIR):  # Case 2: Directory
        print(f"Loading chat data from directory: {DATA_DIR}")
        for file_name in os.listdir(DATA_DIR):
            if file_name.endswith(".txt"):
                file_path = os.path.join(DATA_DIR, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    cleaned_lines = clean_text(lines)
                    chat_data.extend([{"text": line, "source": file_name} for line in cleaned_lines])

    else:
        print("Error: No valid input source found. Set either DATA_DIR or FILE_PATH.")
        exit(1)

    print(f"Total cleaned messages loaded: {len(chat_data)}")
    return chat_data

def main():
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

    # Ensure FAISS index matches embedding dimensions
    d = embedding_matrix.shape[1]
    print("FAISS index dimension:", d)

    index = faiss.IndexFlatL2(d)
    index.add(embedding_matrix)
    
    # Save FAISS index
    faiss.write_index(index, VECTOR_DB_PATH)
    print(f"FAISS index saved at: {VECTOR_DB_PATH}")

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved at: {METADATA_PATH}")

    print("\n✅ Milestone 2 completed: Correct timestamp association and data stored.")

if __name__ == "__main__":
    main()
