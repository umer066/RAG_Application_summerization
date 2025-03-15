# RAG_Application_summerization

This project processes chat logs, generates sentence embeddings using the `sentence-transformers` library, and stores them in a FAISS vector database for efficient similarity search.

## Features
- Automatically detects whether to embed from a **directory** or a **single file**.
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embedding generation.
- Stores embeddings in a FAISS index.
- Saves metadata (chat text & source) in a JSON file.
- Includes progress tracking using `tqdm`.

## Installation
Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```bash
pip install faiss-cpu sentence-transformers tqdm numpy
```

## Usage
### 1. **Embedding from a Directory**
If you have multiple `.txt` files in a folder, set:

Example:
```python
DATA_DIR = "skype_2025"
FILE_PATH = None
```

### 2. **Embedding from a Single File**
If you want to embed from a single `.txt` file, set:

Example:
```python
DATA_DIR = None
FILE_PATH = "chat_log.txt"
```

### 3. **Run the Script**
Once configured, run the script:

```bash
python embedding.py
```

## Output Files
- **`vector_store.index`** → Stores FAISS embeddings.
- **`metadata.json`** → Stores chat text with corresponding metadata.

## Debugging and Logs
- The script prints debugging information such as:
  - Number of messages processed.
  - Embedding matrix shape.
  - FAISS index dimension.
  - Progress tracking with `tqdm`.

## Example Output
```
Loading embedding model...
Model loaded successfully.
Loading chat data from directory: skype_2025
Total messages loaded: 18503
Generating embeddings...
Embedding Progress: 100%|██████████| 18503/18503 [05:22<00:00, 57.39msg/s]
Embedding matrix shape: (18503, 384)
FAISS index dimension: 384
FAISS index created successfully.
Embeddings added to FAISS index.
FAISS index saved at: vector_store.index
Metadata saved at: metadata.json
Data processed, embeddings stored in FAISS.
```

## Notes
- The script will **exit** if neither `DATA_DIR` nor `FILE_PATH` is set.
- Works with `.txt` files containing chat messages, each message on a new line.
- FAISS index uses L2 (Euclidean) distance for efficient search.
