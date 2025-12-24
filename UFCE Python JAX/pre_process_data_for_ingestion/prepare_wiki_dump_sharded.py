
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

import xml.etree.ElementTree as ET
import re
import os
import sys
from tqdm import tqdm

# --- CONFIG ---
INPUT_XML = "enwiki-latest-pages-articles.xml"
OUTPUT_DIR = "wiki_shards"      # Directory for shards
SHARD_SIZE_LIMIT = 500 * 1024 * 1024  # 500 MB per file (Adjustable)

# --- TEST MODE CONFIG ---
TEST_ARTICLE_LIMIT = 20000      # Stop after this many articles in Test Mode

def clean_text(text):
    """Basic cleanup to remove Wiki markup and keep text."""
    try:
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\n\s*\n', '\n', text)
    except Exception:
        return ""
    return text.strip()

def find_child_by_tag(parent, local_tag_name):
    for child in parent:
        if child.tag.endswith(local_tag_name):
            return child
    return None

def get_shard_filename(index, is_test_mode=False):
    if is_test_mode:
        return os.path.join(OUTPUT_DIR, "wiki_test_subset.txt")
    return os.path.join(OUTPUT_DIR, f"wiki_shard_{index:03d}.txt")

def process_wiki_xml():
    if not os.path.exists(INPUT_XML):
        print(f"❌ Error: File {INPUT_XML} not found.")
        return

    # --- USER MENU ---
    print("-" * 60)
    print(f"📄 Wiki Dump Processor (Input: {INPUT_XML})")
    print("-" * 60)
    print("1. TEST RUN (Rapid Validation)")
    print("   -> Stops after 20,000 articles. Output: wiki_test_subset.txt")
    print("\n2. 1/4 DATASET RUN (Medium Stress Test)")
    print("   -> Stops after ~11.5GB (approx 23 shards).")
    print("\n3. 1/2 DATASET RUN (Heavy Stress Test)")
    print("   -> Stops after ~23GB (approx 46 shards).")
    print("\n4. FULL PRODUCTION RUN")
    print("   -> Processes entire 46GB file.")
    print("-" * 60)
    
    choice = input("Select Mode [1-4]: ").strip()
    
    is_test_mode = False
    max_shards = float('inf') 

    if choice == '1':
        print("\n🚀 Starting TEST RUN...")
        is_test_mode = True
    elif choice == '2':
        print("\n🏗️  Starting 1/4 DATASET RUN...")
        max_shards = 23
    elif choice == '3':
        print("\n🏭 Starting 1/2 DATASET RUN...")
        max_shards = 46
    elif choice == '4':
        print("\n🌍 Starting FULL PRODUCTION RUN...")
        max_shards = float('inf')
    else:
        print("Invalid choice. Exiting.")
        return

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Context setup
    context = ET.iterparse(INPUT_XML, events=("end",))
    
    count = 0
    current_shard_idx = 1
    current_shard_size = 0
    
    # Open first file
    current_filename = get_shard_filename(current_shard_idx, is_test_mode)
    f_out = open(current_filename, "w", encoding="utf-8")
    print(f"📂 Writing to: {current_filename}")

    try:
        for event, elem in tqdm(context):
            if elem.tag.endswith('page'):
                title_node = find_child_by_tag(elem, 'title')
                revision_node = find_child_by_tag(elem, 'revision')
                
                if title_node is not None and revision_node is not None:
                    text_node = find_child_by_tag(revision_node, 'text')
                    
                    if text_node is not None and text_node.text:
                        raw_text = text_node.text
                        
                        # Filter redirects and stubs
                        if not raw_text.strip().upper().startswith("#REDIRECT") and len(raw_text) > 200:
                            cleaned = clean_text(raw_text)
                            title = title_node.text if title_node.text else "Unknown"
                            
                            if len(cleaned) > 100:
                                header = f"\n\n[Wiki: {title}]\n"
                                data_block = header + cleaned
                                f_out.write(data_block)
                                
                                # Update counters
                                count += 1
                                current_shard_size += len(data_block.encode('utf-8'))

                                # --- CHECK LIMITS ---
                                
                                # 1. Test Mode Limit
                                if is_test_mode and count >= TEST_ARTICLE_LIMIT:
                                    print(f"\n🛑 Test limit reached ({TEST_ARTICLE_LIMIT} articles). Stopping.")
                                    break
                                
                                # 2. Shard Size Limit
                                if not is_test_mode and current_shard_size >= SHARD_SIZE_LIMIT:
                                    f_out.close()
                                    print(f"\n📦 Shard {current_shard_idx} full ({current_shard_size/1024/1024:.2f} MB).")
                                    
                                    # Partial Run Check
                                    if current_shard_idx >= max_shards:
                                        print(f"🛑 Partial Run limit reached ({max_shards} shards). Stopping.")
                                        elem.clear()
                                        break

                                    # Prepare Next Shard
                                    current_shard_idx += 1
                                    current_shard_size = 0
                                    current_filename = get_shard_filename(current_shard_idx)
                                    f_out = open(current_filename, "w", encoding="utf-8")

                # Clear memory
                elem.clear()
                if hasattr(elem, 'getparent') and elem.getparent() is not None:
                     elem.getparent().remove(elem)
    
    # --- SAFETY NETS ---
    except ET.ParseError:
        print("\n⚠️  XML Parse Error or End of File reached. Saving progress...")
    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Closing files safely...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    finally:
        if not f_out.closed:
            f_out.close()
        
    print("-" * 60)
    print(f"✅ DONE. Processed {count} articles.")
    if is_test_mode:
        print(f"👉 Test file: {os.path.join(OUTPUT_DIR, 'wiki_test_subset.txt')}")
    else:
        print(f"👉 Generated {current_shard_idx} shards in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_wiki_xml()