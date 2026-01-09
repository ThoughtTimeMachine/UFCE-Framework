import sys
import bz2
import os
import xml.sax
import re
import time
import html
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from tqdm import tqdm

# --- ULTRA CONFIG V7 (Async Writer Edition) ---
INPUT_FILE = r"wiki\wiki_data\enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = r"wiki\extracted"

# FILE SETTINGS
CHUNK_SIZE_MB = 512
# On Windows, smaller flush sizes often help avoid "System Cache" thrashing
BYTES_PER_FLUSH = 10 * 1024 * 1024 

# PARALLELISM
# Reduced batch size to prevent "Pipe Blocking" on Windows
BATCH_SIZE = 2000           
WORKER_TIMEOUT = 30         # Aggressive timeout: if a batch hangs for 30s, drop it.

# --- COMPILED REGEX PATTERNS (Same as V6) ---
RE_INFOBOX_PARAM = re.compile(r'\|\s*([a-zA-Z0-9_ -]+?)\s*=\s*(.*?)(?=(?:\n\s*\||\}\}))', re.DOTALL)
RE_COMMENT = re.compile(r'', re.DOTALL)
RE_FILES = re.compile(r'\[\[(File|Image|Category):.*?\]\]', re.DOTALL | re.IGNORECASE)
RE_REFS = re.compile(r'<ref.*?>.*?</ref>', re.DOTALL | re.IGNORECASE)
RE_TEMPLATES = re.compile(r'\{\{.*?\}\}', re.DOTALL) 
RE_External_Links = re.compile(r'\[http\S+ (.*?)\]') 
RE_LINKS_PIPED = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]') 
RE_LINKS = re.compile(r'\[\[([^\]]+)\]\]')                    
RE_BOLD_ITALIC = re.compile(r"''+")                            
RE_MULTI_NEWLINES = re.compile(r'\n{3,}')
RE_SPACES = re.compile(r' {2,}')
RE_TABLE_BLOCK = re.compile(r'\{\|.*?\|\}', re.DOTALL)
RE_HEADINGS = re.compile(r'^(={2,6})\s*(.*?)\s*\1', re.MULTILINE)
RE_LIST_UL = re.compile(r'^\*\s+', re.MULTILINE) 
RE_LIST_OL = re.compile(r'^\#\s+', re.MULTILINE) 

FOOTER_HEADERS = [
    "== See also ==", "==See also==", 
    "== References ==", "==References==", 
    "== External links ==", "==External links==", 
    "== Notes ==", "==Notes=="
]

# --- WORKER FUNCTIONS ---
def extract_infobox_data(text):
    if "{{Infobox" not in text: return ""
    extracted_lines = []
    matches = RE_INFOBOX_PARAM.findall(text)
    for key, val in matches:
        key = key.strip().replace("_", " ").title()
        val = val.strip()
        val = RE_LINKS_PIPED.sub(r'\1', val)
        val = RE_LINKS.sub(r'\1', val)
        val = RE_TEMPLATES.sub('', val)
        val = val.replace('\n', ' ')
        if len(val) > 1 and len(val) < 150 and "{" not in val and "<" not in val:
            extracted_lines.append(f"{key}: {val}")
    if extracted_lines:
        return "\n".join(extracted_lines[:15]) + "\n\n" 
    return ""

def convert_tables_to_lists(text):
    def replace_table(match):
        table_content = match.group(0)
        clean_lines = []
        lines = table_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{|') or line.startswith('|-') or line.startswith('|}'): continue
            if line.startswith('|') or line.startswith('!'):
                content = line.lstrip('|!').strip()
                cells = re.split(r'\|\||!!', content)
                for cell in cells:
                    cell = cell.strip()
                    if cell: clean_lines.append(f"* {cell}")
        return "\n".join(clean_lines)
    return RE_TABLE_BLOCK.sub(replace_table, text)

def clean_batch_worker(article_batch):
    results = []
    local_count = 0
    for title, text in article_batch:
        try:
            if text.strip().lower().startswith("#redirect"): continue
            
            earliest_cut = len(text)
            for header in FOOTER_HEADERS:
                idx = text.find(header)
                if idx != -1 and idx < earliest_cut: earliest_cut = idx
            if earliest_cut < len(text): text = text[:earliest_cut]

            text = html.unescape(text)
            infobox_text = extract_infobox_data(text)
            text = convert_tables_to_lists(text)
            text = RE_COMMENT.sub('', text)
            text = RE_FILES.sub('', text)
            text = RE_REFS.sub('', text)
            for _ in range(3): text = RE_TEMPLATES.sub('', text)
            
            def header_repl(match):
                level = len(match.group(1))
                content = match.group(2)
                return f"{'#' * level} {content}"
            text = RE_HEADINGS.sub(header_repl, text)
            text = RE_LIST_UL.sub('- ', text)
            text = RE_LIST_OL.sub('1. ', text)
            text = RE_External_Links.sub(r'\1', text)
            text = RE_LINKS_PIPED.sub(r'\1', text)
            text = RE_LINKS.sub(r'\1', text)
            text = RE_BOLD_ITALIC.sub('', text)
            text = RE_SPACES.sub(' ', text)
            text = RE_MULTI_NEWLINES.sub('\n\n', text)
            clean_text = text.strip()
            
            if infobox_text: clean_text = infobox_text + clean_text

            if len(clean_text) > 200 and not clean_text.startswith("Category:"):
                results.append(f"# {title}\n\n{clean_text}\n\n")
                local_count += 1
        except: continue
    return results, local_count

# --- ASYNC WRITER THREAD ---
class AsyncWriter(threading.Thread):
    def __init__(self, output_dir, chunk_size_mb, pbar):
        super().__init__()
        self.output_dir = output_dir
        self.limit_bytes = chunk_size_mb * 1024 * 1024
        self.pbar = pbar
        self.queue = queue.Queue(maxsize=200) # Buffer up to 200 batches
        self.active = True
        self.daemon = True
        self.total_articles = 0
        
        # File handles
        self.file_index = 0
        self.current_file = None
        self.current_size = 0
        self._ensure_dir()
        self._open_next()

    def _ensure_dir(self):
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def _open_next(self):
        if self.current_file: self.current_file.close()
        fname = os.path.join(self.output_dir, f"wiki_{self.file_index:02d}.txt")
        self.current_file = open(fname, "w", encoding="utf-8")
        self.current_size = 0
        self.file_index += 1
        tqdm.write(f"[IO] Rotation: Opened {fname}")

    def run(self):
        while self.active or not self.queue.empty():
            try:
                # Wait 1s for data, then check 'active' flag again
                data = self.queue.get(timeout=1)
                text_list, count = data
                
                # Write to disk
                big_chunk = "".join(text_list)
                encoded = big_chunk.encode("utf-8")
                self.current_file.write(big_chunk)
                self.current_size += len(encoded)
                
                # Update stats
                self.total_articles += count
                self.pbar.set_description(f"Arts: {self.total_articles:,}")
                
                if self.current_size >= self.limit_bytes: 
                    self._open_next()
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                tqdm.write(f"[WRITER ERR] {e}")

    def submit(self, result):
        # This runs in the Main Thread callback. 
        # It puts data into the queue for the Writer Thread.
        self.queue.put(result)

    def stop(self):
        self.active = False
        if self.current_file: self.current_file.close()

# --- MAIN HANDLER ---
class UltraHandlerV7(xml.sax.ContentHandler):
    def __init__(self, executor, writer):
        self.executor = executor
        self.writer = writer
        self.batch_buffer = []
        self._buffer = []
        self._tag = None
        self._title = None
        self._text = None

    def startElement(self, name, attrs):
        self._tag = name
        if name == "page":
            self._buffer = []
            self._title = None
            self._text = None

    def endElement(self, name):
        if name == "page":
            if self._title and self._text:
                self.batch_buffer.append((self._title, self._text))
                if len(self.batch_buffer) >= BATCH_SIZE:
                    self._submit_batch()
        elif name == "title": self._title = "".join(self._buffer).strip()
        elif name == "text": self._text = "".join(self._buffer)
        self._buffer = []
        self._tag = None

    def characters(self, content):
        if self._tag in ("title", "text"): self._buffer.append(content)

    def _submit_batch(self):
        # Fire and Forget: We don't wait for result here.
        # We attach a callback that handles the result when it's ready.
        future = self.executor.submit(clean_batch_worker, list(self.batch_buffer))
        future.add_done_callback(self._on_batch_complete)
        self.batch_buffer = []

    def _on_batch_complete(self, future):
        try:
            # Check result immediately (or check timeout if we could)
            # Since we can't easily timeout a callback, we trust the worker finished.
            result = future.result()
            self.writer.submit(result)
        except Exception as e:
            # Silently fail on errors to keep pipeline moving
            pass

    def finish(self):
        if self.batch_buffer: self._submit_batch()
        # In V7, we just wait for the writer queue to empty in the main block

if __name__ == "__main__":
    print("--- WIKI ULTRA EXTRACTOR V7 (Async Writer) ---")
    print(f"Input: {INPUT_FILE}")
    
    if os.path.exists(OUTPUT_DIR):
        import shutil
        print("Cleaning old output directory...")
        shutil.rmtree(OUTPUT_DIR)
        time.sleep(1) 
    
    workers = max(1, os.cpu_count()) 
    print(f"Workers: {workers}")

    # Initialize Threaded Writer
    # We pass a dummy pbar for now, will update it later
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        file_size = os.path.getsize(INPUT_FILE)
        
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            
            # Start the Writer Thread
            writer_thread = AsyncWriter(OUTPUT_DIR, CHUNK_SIZE_MB, pbar)
            writer_thread.start()
            
            handler = UltraHandlerV7(executor, writer_thread)
            parser = xml.sax.make_parser()
            parser.setContentHandler(handler)
            parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            
            try:
                with bz2.open(INPUT_FILE, "rt", encoding="utf-8") as f:
                    for line in f:
                        parser.feed(line)
                        pbar.update(len(line.encode('utf-8')))
                
                handler.finish()
                
                # Wait for Writer to clear the queue
                tqdm.write("XML Reading complete. Waiting for writer to finish...")
                writer_thread.queue.join() 
                
            except KeyboardInterrupt:
                print("\nStopped by user.")
            finally:
                writer_thread.stop()
                print(f"\nDONE. Processed {writer_thread.total_articles:,} articles.")
                print(f"Output in {OUTPUT_DIR}")