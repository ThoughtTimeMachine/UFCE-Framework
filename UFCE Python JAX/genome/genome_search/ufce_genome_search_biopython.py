# Copyright (C) 2025 Kyle Killian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import re
import json

# --- TRY IMPORTING BIOPYTHON ---
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.ufce_agent import UFCEAgent

# --- CONFIGURATION LOADER ---
CONFIG_FILE = os.path.join(project_root, "velocity_config.json")

def get_genome_paths():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"❌ Config file not found at: {CONFIG_FILE}")
        
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
        
    # We target 'genome' generically so it works for both E. coli and Human config blocks
    # Note: Ensure your velocity_config.json has a "human_genome" block, or change this key to match yours
    target_key = "human_genome" 
    if target_key not in data.get("datasets", {}):
        # Fallback: check if active_dataset is a genome type
        target_key = data.get("active_dataset", "human_genome")
    
    if target_key not in data["datasets"]:
        raise ValueError(f"❌ Dataset block '{target_key}' missing in config.")
        
    cfg = data["datasets"][target_key]
    kb_dir = os.path.join(project_root, cfg["vectors_output_dir"])
    db_path = os.path.join(kb_dir, cfg["final_dat_name"])
    meta_path = os.path.join(kb_dir, cfg["final_meta_name"])
    
    return db_path, meta_path

BIO_KEYWORDS = ["operon", "array", "gene", "promoter", "sequence", "cluster", "cassette", "island"]

# --- ANSI COLORS ---
RED = "\033[91m"
GREEN = "\033[92m" # Bio Stats
BLUE = "\033[94m"  # Protein
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m" 
YELLOW = "\033[93m"

def run_genome_search_demo():
    print(f"{BOLD}🚀 UFCE Genomic Search Demo (Bio-Enhanced){RESET}")
    
    if not HAS_BIOPYTHON:
        print(f"{YELLOW}⚠️  Warning: Biopython not found. Run 'pip install biopython' for advanced analysis.{RESET}")

    try:
        DB_PATH, META_PATH = get_genome_paths()
        print(f"📂 Targeting DB: {DB_PATH}")
    except Exception as e:
        print(e)
        return

    print("Loading genomic knowledge base...")
    
    try:
        agent = UFCEAgent(db_path=DB_PATH, meta_path=META_PATH, top_k=5)
        print(f"{CYAN}✅ Knowledge base loaded successfully!{RESET}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    print("Ready! Ask about genes, mutations, sequences, etc. (type 'quit' to exit)\n")
    
    while True:
        query = input(f"{BOLD}🧬 Your genomic query: {RESET}").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not query:
            continue
        
        print("\nSearching genome...")
        
        try:
            results, scan_time = agent.search(query)
            
            print(f"\n📊 Top Matches (Scanned in {scan_time:.4f}s):")
            
            if not results:
                print("   No matches found.")
            
            for i, text in enumerate(results, 1):
                # Clean text: remove newlines for processing
                snippet = text.replace('\n', '') 
                
                # --- BIOPYTHON ANALYSIS ---
                strand_info = ""
                protein_preview = ""
                gc_stat = ""
                
                if HAS_BIOPYTHON:
                    seq_obj = Seq(snippet)
                    
                    # 1. GC Content
                    gc = gc_fraction(seq_obj) * 100
                    gc_stat = f"GC: {gc:.1f}%"
                    
                    # 2. Translation (First 60 bases -> 20 AA)
                    try:
                        prot = seq_obj[:60].translate() 
                        protein_preview = f"Protein: {prot}..."
                    except:
                        protein_preview = "Protein: (Non-coding/Partial)"

                    # 3. Reverse Complement Check
                    is_dna_query = all(c in "ATCGNatcgn" for c in query) and len(query) > 3
                    
                    if is_dna_query:
                        query_rc = str(Seq(query).reverse_complement())
                        if query_rc.upper() in snippet.upper():
                            strand_info = f"{RED}[FOUND ON REVERSE STRAND: {query_rc}]{RESET}"

                # --- DYNAMIC WINDOW ---
                window_size = 600
                if len(query) > 20 or any(word in query.lower() for word in BIO_KEYWORDS):
                    window_size = 1500
                
                # --- HIGHLIGHTING & DISPLAY LOGIC ---
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                match = pattern.search(snippet)
                
                display_text = ""
                
                if match:
                    # Case A: Exact Match Found (Forward)
                    print(f"   {CYAN}📍 Match position: ~{match.start()} bp {strand_info}{RESET}")
                    
                    start_idx = max(0, match.start() - (window_size // 4))
                    end_idx = min(len(snippet), match.start() + ((window_size * 3) // 4))
                    display_text = snippet[start_idx:end_idx]
                    
                    if start_idx > 0: display_text = "..." + display_text
                    if end_idx < len(snippet): display_text = display_text + "..."
                    
                    # Apply Highlight
                    display_text = pattern.sub(rf"{RED}{BOLD}\g<0>{RESET}", display_text)

                elif strand_info and HAS_BIOPYTHON:
                     # Case B: Match on Reverse Strand Only
                     print(f"   {CYAN}📍 Match position: (Reverse Strand) {strand_info}{RESET}")
                     
                     # Find where the RC is to center the window
                     rc_pattern = re.compile(re.escape(query_rc), re.IGNORECASE)
                     rc_match = rc_pattern.search(snippet)
                     
                     if rc_match:
                         start_idx = max(0, rc_match.start() - (window_size // 4))
                         end_idx = min(len(snippet), rc_match.start() + ((window_size * 3) // 4))
                         display_text = snippet[start_idx:end_idx]

                         if start_idx > 0: display_text = "..." + display_text
                         if end_idx < len(snippet): display_text = display_text + "..."
                         
                         # Highlight the RC string
                         display_text = rc_pattern.sub(rf"{RED}{BOLD}\g<0>{RESET}", display_text)
                     else:
                         display_text = snippet[:window_size] + "..."

                else:
                    # Case C: Semantic Match (No exact string found)
                    print(f"   {YELLOW}⚠️  Semantic match (exact string not found){RESET}")
                    display_text = snippet[:window_size] + "..."

                # --- PRINT RESULT BLOCK ---
                if HAS_BIOPYTHON:
                    print(f"   {GREEN}🔬 {gc_stat} | {BLUE}{protein_preview}{RESET}")
                
                print(f"{i}. {display_text}\n")
                
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    run_genome_search_demo()