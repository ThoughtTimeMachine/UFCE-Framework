This is a great idea. A dedicated README.md for the genomic subsystem is crucial because the workflow (Download -> Preprocess -> Ingest -> Merge -> Search) is slightly more complex than just "run this script."

Here is a comprehensive README.md tailored for your knowledge_base_genome/ or genome/ folder. It explicitly documents the "Switchboard" configuration system and how to add custom data.

genome/README.md
Markdown

# 🧬 UFCE Genome: Infinite Context DNA Search

This module implements a hardware-accelerated **Semantic Search Engine for Genomic Data**. Unlike standard BLAST searches that look for exact character matches, UFCE Genome uses vector embeddings to find "semantic" similarities (e.g., similar gene functions, promoter regions, or structural motifs) even if the exact sequence varies.

Powered by **JAX** and **Project VELOCITY**, this system allows you to search through millions of base pairs in milliseconds on consumer hardware.

---

## 🚀 Quick Start (E. coli Demo)

If you just want to run the pre-configured *E. coli* demo:

1.  **Check Configuration:**
    Open `velocity_config.json` in the project root and ensure the active dataset is set:
    ```json
    "active_dataset": "genome"
    ```

2.  **Run the Pipeline (One-Time Setup):**
    ```bash
    # 1. Download & Clean Data
    python genome/preprocessors/preprocess_fasta_ecoli.py

    # 2. Vectorize the Data (JAX/GPU)
    python ufce_ingestion_pipeline_shard.py

    # 3. Merge into Single Database
    python merge_shards.py
    ```

3.  **Launch the Search Agent:**
    ```bash
    python genome/genome_search/ufce_genome_search_demo.py
    ```

---

## ⚙️ The "Switchboard" Architecture

This project uses a **Dataset Agnostic** architecture controlled by `velocity_config.json`. You do not need to edit Python scripts to switch between datasets (e.g., Wikipedia vs. Genome).

### `velocity_config.json`
To switch datasets, simply change the `"active_dataset"` key:

```json
{
    "active_dataset": "genome",  <-- Change this to "wiki" or "custom_data"
    "datasets": {
        "genome": {
            "description": "E. coli Genomic Data",
            "shards_input_dir": "genome/shards/ecoli_shards",
            "vectors_output_dir": "genome/knowledge_base/ecoli_vectors",
            "final_dat_name": "knowledge_base_full.dat",
            "final_meta_name": "metadata_full.txt",
            ...
        }
    }
}
```
The ingestion and merge scripts (`ufce_ingestion_pipeline_shard.py` and `merge_shards.py`) automatically read this config to determine input folders and output filenames.

### 🧪 Adding Custom Genomic Data
Want to search a Human Chromosome, Yeast, or your own synthetic DNA? Follow this standard workflow.

1. **Create a Preprocessor**  
   Create a new script in `genome/preprocessors/` (e.g., `preprocess_human_chr1.py`).  

   **Goal**: Download your raw data (FASTA/FASTQ) and convert it into a simple `.txt` file where each line is a chunk of sequence.  

   **Reference**: See `preprocess_fasta_ecoli.py` for a template that handles downloading, unzipping, and header removal.

2. **Update Configuration**  
   Add a new block to `velocity_config.json`:

```json
"my_custom_genome": {
    "description": "Human Chromosome 1",
    "shards_input_dir": "genome/shards/human_shards",
    "vectors_output_dir": "genome/knowledge_base/knowledge_base_human",
    "final_dat_name": "knowledge_base_full.dat",
    "final_meta_name": "metadata_full.txt",
    "max_tokens": 512,
    "embedding_dim": 384,
    "batch_size": 64,
    "preprocessing": {
        "fasta_url": "[https://url-to-your-data.gz](https://url-to-your-data.gz)",
        "raw_data_dir": "genome/genome_data",
        "raw_filename": "human_chr1.fna.gz",
        "preprocessed_filename": "human_chr1_clean.txt"
    }
}
```
### 3. Run the Pipeline
Set `"active_dataset": "my_custom_genome"` in the `velocity_config.json` file.


1.  **Run your preprocessor:**
```bash
python genome/preprocessors/preprocess_human_chr1.py
```
2.  **Run ingestion:**
```bash
python ufce_ingestion_pipeline_shard.py
```
3.  **Run merge:**
```bash
python merge_shards.py
```
### Features of the Genomic Search Demo

The `ufce_genome_search_demo.py` tool includes advanced visualization features tailored for biology:

- **Smart Windowing**: Automatically expands context (from 600 bp to 1500 bp) for complex queries containing terms like "operon", "cluster", or "sequence".
- **Dynamic Highlighting**: Uses regex to find and **bold/red highlight** matches regardless of case (e.g., "GATTACA" will highlight "gattaca").
- **Position Tracking**: Displays the approximate base-pair (bp) position of the match within the chunk.
- **Semantic Fallback**: If the exact string isn't found, it still shows the most relevant chunk found by vector similarity (marked with ⚠️ Semantic match).

Your custom genomic database is now ready for searching! 🧬