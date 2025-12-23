# Wiki Dump Preparation Script (`prepare_wiki_dump.py`)

## Overview
The `prepare_wiki_dump.py` script is a specialized pre-processing tool designed to convert massive raw Wikipedia XML dumps (e.g., `enwiki-latest-pages-articles.xml`, ~25GB+) into a clean, flat text format (`large_dataset.txt`) suitable for vector ingestion.

Its primary design goal is **Memory Safety**. It avoids loading the entire XML tree into RAM, which is the standard behavior of many XML parsers that causes crashes on consumer hardware.

---

## How It Works

### 1. Streaming XML Parsing (`iterparse`)
Instead of building a Document Object Model (DOM) for the entire 25GB file, the script uses Python's `xml.etree.ElementTree.iterparse`.
* **Event-Driven:** It listens for specific "end" events (like the closing `</page>` tag).
* **Immediate Release:** Once a page is processed and written to disk, `elem.clear()` is called immediately. This frees the memory used by that specific XML node, ensuring RAM usage remains flat (typically <500MB) regardless of file size.

### 2. Heuristic Cleaning
Wikipedia data is messy, filled with `{{Infobox}}` templates, `[[Category:...]]` tags, and `ref` citations. The script applies high-speed Regular Expressions (RegEx) to strip this syntax:
* **Removes:** `{{...}}` (Infoboxes), `<...>` (HTML/XML tags).
* **Formats:** Converts `[[Link|Text]]` to just `Text`.
* **Filters:** Skips `#REDIRECT` pages and "stubs" (articles < 200 characters) to ensure high-density information.

### 3. Structured Output
The output file `large_dataset.txt` is structured for the downstream ingestion pipeline:
* **Headers:** Each article is preceded by `[Wiki: Title]` to provide semantic context for the embedding model.
* **Flat Text:** The body is cleaned, plain text ready for tokenization.

---

## Efficiency Metrics
* **Time Complexity:** $O(N)$ linear scan of the XML file.
* **Space Complexity:** $O(1)$ constant RAM usage (only holds one article at a time).
* **Throughput:** Processes ~2-5 MB/sec (CPU bound by RegEx and disk I/O).

## Usage
```bash
python prepare_wiki_dump.py

Input: enwiki-latest-pages-articles.xml (Must be in root directory) Output: large_dataset.txt