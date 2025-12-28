import os
import sys

# Add the project root to Python path so we can import from ai_agents
# This works whether running from genome_search/ or project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the agent
from ai_agents.ufce_agent import UFCEAgent  # <-- Correct relative import

# --- CONFIG ---
DB_PATH = "knowledge_base_genome/knowledge_base_full.dat"   # Adjust if needed
META_PATH = "knowledge_base_genome/metadata_full.txt"

# --- DEMO ---
def run_genome_search_demo():
    print("🚀 UFCE Genomic Search Demo")
    print("Loading genomic knowledge base...")
    
    try:
        agent = UFCEAgent(db_path=DB_PATH, meta_path=META_PATH)
        print("✅ Knowledge base loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        print("   Check that the .dat and .txt files exist and paths are correct.")
        return
    
    print("Ready! Ask about genes, mutations, sequences, etc. (type 'quit' to exit)\n")
    
    while True:
        query = input("🧬 Your genomic query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not query:
            continue
        
        print("\nSearching genome...")
        try:
            results = agent.query(query, top_k=10)
            
            print("\n📊 Top 10 Matches:")
            for i, (score, text) in enumerate(results, 1):
                snippet = text[:300].replace('\n', ' ') + ("..." if len(text) > 300 else "")
                print(f"{i}. [Score: {score:.4f}] {snippet}\n")
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    run_genome_search_demo()