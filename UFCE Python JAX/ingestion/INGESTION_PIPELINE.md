# UFCE Ingestion Pipeline (`UFCE_ingestion_pipeline.py`)

## Overview
The `UFCE_ingestion_pipeline.py` is the data refinery of the UFCE framework. It transforms raw text into a **Tiered Memory Reservoir** consisting of a binary vector database (`knowledge_base.dat`) and a text metadata index (`metadata.txt`).

This script implements the **"Zero-Memory" 2-Pass Architecture**, allowing it to process text datasets larger than available system RAM (e.g., 100GB+ on a 16GB laptop).

---

## The 2-Pass Architecture

### Pass 1: The "Dry Run" (Indexing)
* **Goal:** Calculate the exact number of chunks and generate the metadata file *without* loading text arrays into RAM.
* **Mechanism:**
    1.  Streams the input file line-by-line.
    2.  Uses the `AutoTokenizer` (Hugging Face) to generate semantically valid chunks (max 256 tokens).
    3.  Writes each chunk to `metadata.txt` immediately.
    4.  Increments a counter `num_chunks`.
* **Outcome:** We know exactly how large the vector database needs to be.

### Pass 2: The "Vectorization" (Embedding)
* **Goal:** Compute embeddings and write them to the SSD.
* **Mechanism:**
    1.  **Pre-Allocation:** Uses `numpy.memmap` to reserve space on the SSD for `(num_chunks, 384)` float32 matrix. This uses 0 RAM.
    2.  **Streaming:** Re-opens the text file stream.
    3.  **Batching:** Collects small batches (e.g., 64 chunks) in a temporary buffer.
    4.  **GPU Encode:** Sends the batch to the GPU via `SentenceTransformer`.
    5.  **Direct Write:** Flushes the resulting vectors directly to the `memmap` on disk.
    6.  **Garbage Collection:** Clears the batch buffer immediately.

---

## Why It Is Superior

| Feature | Standard Approach | UFCE Pipeline Approach |
| :--- | :--- | :--- |
| **RAM Usage** | Grows with file size ($O(N)$) | Constant / Flat ($O(1)$) |
| **Chunking** | Hard cuts (character count) | **Semantic Tokenizer** (Whole words) |
| **Storage** | `pickle` / `faiss` (RAM heavy) | `numpy.memmap` (SSD / Virtual RAM) |
| **Crash Risk** | High for >10GB files | Near Zero |

## Logical Data Flow


`large_dataset.txt` $\rightarrow$ **Pass 1** $\rightarrow$ `metadata.txt` (Human Readable)
$\downarrow$
**Pass 2** (GPU)
$\downarrow$
`knowledge_base.dat` (Machine Readable / Binary)

This architecture ensures that the **UFCE Agent** always has a 1:1 mapping between row $N$ in the `.dat` file and line $N$ in the `.txt` file, enabling instant lossless retrieval.