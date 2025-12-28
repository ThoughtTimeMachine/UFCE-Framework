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
# --- PATH SETUP ---
# 1. Get the Project Root (one level up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Add Root to Python Path so we can see the 'ai_agents' folder
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORT ---
# Using the path you confirmed works:
from ai_agents.ufce_agent import UFCEAgent

# --- CONFIGURATION ---
# Points to: Project_Root/knowledge_base_genome/...
DB_PATH = os.path.join(project_root, "knowledge_base_genome", "knowledge_base_full_ecoli.dat")
META_PATH = os.path.join(project_root, "knowledge_base_genome", "metadata_full_ecoli.txt")

# Keywords that trigger a larger context window (1500 chars)
BIO_KEYWORDS = ["operon", "array", "gene", "promoter", "sequence", "cluster", "cassette", "island"]

# --- ANSI COLORS ---
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m" 
YELLOW = "\033[93m"

# --- DEMO FUNCTION ---
def run_genome_search_demo():
    print(f"{BOLD}🚀 UFCE Genomic Search Demo{RESET}")
    print(f"📂 Targeting DB: {DB_PATH}")
    print("Loading genomic knowledge base...")
    
    try:
        agent = UFCEAgent(
            db_path=DB_PATH, 
            meta_path=META_PATH,
            top_k=10
        )
        print(f"{CYAN}✅ Knowledge base loaded successfully!{RESET}\n")
    except Exception as e:
        print(f"\n❌ Error loading knowledge base: {e}")
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
                # Clean text first
                snippet = text.replace('\n', ' ')
                
                # --- DYNAMIC WINDOW CONFIG ---
                window_size = 600 # Default
                
                # Check keywords using the global list
                if len(query) > 20 or any(word in query.lower() for word in BIO_KEYWORDS):
                    window_size = 1500
                
                # --- SMART WINDOWING LOGIC ---
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                match = pattern.search(snippet)
                
                display_text = ""
                
                if match:
                    # Print Metadata about the match location
                    print(f"   {CYAN}📍 Match position: ~{match.start()} bp in chunk{RESET}")
                    
                    # Match found! Calculate dynamic window around it.
                    # 25% Upstream (Promoter region), 75% Downstream (Coding region)
                    upstream_pad = window_size // 4
                    downstream_pad = (window_size * 3) // 4
                    
                    start_idx = max(0, match.start() - upstream_pad)
                    end_idx = min(len(snippet), match.start() + downstream_pad)
                    
                    # Slice the raw text around the match
                    display_text = snippet[start_idx:end_idx]
                    
                    # Add ellipses if we skipped text
                    if start_idx > 0:
                        display_text = "..." + display_text
                    if end_idx < len(snippet):
                        display_text = display_text + "..."
                else:
                    # No exact string match (Semantic Match)
                    print(f"   {YELLOW}⚠️  Semantic match (exact string not found){RESET}")
                    display_text = snippet[:window_size] + "..."

                # --- HIGHLIGHTING ---
                highlighted = pattern.sub(f"{RED}{BOLD}\g<0>{RESET}", display_text)
                
                print(f"{i}. {highlighted}\n")
                
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    run_genome_search_demo()