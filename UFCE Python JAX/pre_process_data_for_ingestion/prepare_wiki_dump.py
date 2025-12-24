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
from tqdm import tqdm

# --- CONFIG ---
INPUT_XML = "enwiki-latest-pages-articles.xml"  # Your 25GB download
OUTPUT_TXT = "large_dataset.txt"                # The clean input for UFCE

def clean_text(text):
    """Basic cleanup to remove Wiki markup and keep text."""
    # Remove {{Infobox...}} and other curly brace structures (simplified)
    # Note: A full Wiki parser is complex; this is a 'good enough' 80/20 heuristic
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text) # Remove XML tags
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text) # Clean links [[A|B]] -> B
    return text.strip()

def process_wiki_xml():
    print(f"🔨 Extracting text from {INPUT_XML}...")
    context = ET.iterparse(INPUT_XML, events=("end",))
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
        count = 0
        for event, elem in tqdm(context):
            if elem.tag.endswith('page'):
                # Extract Title and Text
                title_node = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
                revision_node = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
                
                if revision_node is not None:
                    text_node = revision_node.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                    if text_node is not None and text_node.text:
                        raw_text = text_node.text
                        # Skip redirects and tiny stubs
                        if "#REDIRECT" not in raw_text and len(raw_text) > 200:
                            cleaned = clean_text(raw_text)
                            title = title_node.text if title_node is not None else "Unknown"
                            
                            # Write formatted header for your agent
                            header = f"\n\n[Wiki: {title}]\n"
                            f_out.write(header + cleaned)
                            count += 1

                # Clear element from RAM (Critical for massive XML)
                elem.clear()
    
    print(f"✅ Extracted {count} articles to {OUTPUT_TXT}")

if __name__ == "__main__":
    process_wiki_xml()