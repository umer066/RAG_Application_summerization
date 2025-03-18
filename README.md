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

### **1️⃣ Install Python (Skip if already installed)**  

#### **Windows**  
1. Download Python from the official site: [Python.org](https://www.python.org/downloads/windows/)  
2. Run the installer and **check "Add Python to PATH"** before installing.  
3. Verify installation:

```bash
   python --version
```

#### **Mac**
1. Install using Homebrew (recommended):
```bash
brew install python
```
2. Verify installation:
```bash
python3 --version
```

#### **Linux (Debian/Ubuntu)**
1. Install Python via APT:
```bash
sudo apt update && sudo apt install python3 python3-venv python3-pip -y
```
2. Verify installation:
```bash
python3 --version
```


### **Create & Activate a Virtual Environment (Just activate if already created - run second command only)** 

#### Windows (CMD or PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac & Linux (Terminal)
```bash
python3 -m venv venv
source venv/bin/activate
```


###  Install Dependencies
Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```


### ** Cleaning Chat Data for Embedding**  

#### **📌 Overview**  
This script processes and cleans chat data before generating embeddings. It removes system messages, timestamps, and irrelevant text to ensure only meaningful chat messages are embedded.

---

### **⚙️ Cleaning Process**  
✔ **Removes system messages** (e.g., user joins, leaves, invite links, settings changes).  
✔ **Filters out timestamps & dates** (ignores invalid years >2100 or <1900).  
✔ **Strips unnecessary spaces & blank lines**.  


---

### **📌 Output**  
- **Cleaned chat messages** (stored in embeddings)  
- **Metadata file:** `metadata.json`  
- **FAISS index file:** `vector_store.index`  

🎯 **Ready for further analysis & querying!** 🚀

## Usage, set in *embed.py*

### 1. **Embedding from a Directory**
If you have multiple `.txt` files in a folder:

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
python embed.py
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
