# UFCE Ingestion Pipeline (`ufce_ingestion_pipeline_shard.py` & `merge_shards.py`)

## Overview
The UFCE ingestion pipeline has been upgraded to a **sharded, resumable architecture** for handling truly massive datasets (e.g., full Wikipedia dumps, Common Crawl subsets, or multi-domain corpora). It consists of two scripts:

- `ufce_ingestion_pipeline_shard.py`: Processes individual text shards into vector + metadata pairs.
- `merge_shards.py`: Concatenates all shards into the final `knowledge_base_full.dat` and `metadata_full.txt` used by the UFCE agent.

This design enables **safe, parallel, and resumable** ingestion of datasets far larger than system RAM while maintaining the **Zero-Memory** philosophy.

---

## Sharded Workflow

## Preparing Wikipedia Input Shards
The pipeline works best with the full English Wikipedia dump.

1. **Download the Dump**:
   ```bash
   wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

Extract Clean Text Shards (using WikiExtractor):
Install WikiExtractor:bash

pip install wikiextractor

Run extraction to create the wiki_shards/ folder:bash

WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 --output wiki_shards/ -b 1M

This produces multiple text files in wiki_shards/ (one per large article batch), ready for the ingestion pipeline.

### 2. Run Sharded Ingestion (`ufce_ingestion_pipeline_shard.py`)
```bash
python ufce_ingestion_pipeline_shard.py

Key Features:Resumable: Skips already-processed shards.
Semantic Chunking: Uses Hugging Face tokenizer (256 tokens max) for high-quality chunks.
Zero-Memory: Streams input, writes directly to memmap on disk.
Progress Bars: tqdm for both passes.
Output: One .dat + _meta.txt per shard in knowledge_base/.

### 3. Merge Shards (merge_shards.py)
```bash
python merge_shards.py

Key Features:Binary concatenation of .dat files (fast, zero-copy).
Text concatenation of metadata.
Output: Single knowledge_base_full.dat + metadata_full.txt ready for the UFCE agent.

