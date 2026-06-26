import os
import pickle
from collections import Counter

def safe_print(text):
    print(str(text).encode('ascii', 'replace').decode('ascii'))

def inspect_corpus():
    pkl_path = "data/processed/lora_chunks.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return
        
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
        
    total_chunks = len(chunks)
    if total_chunks == 0:
        print("Corpus is empty.")
        return
        
    sections = [c.section for c in chunks]
    section_counts = Counter(sections)
    
    total_length = sum(len(c.text) for c in chunks)
    total_tokens = sum(len(c.text) // 4 for c in chunks)
    
    avg_length = total_length / total_chunks
    avg_tokens = total_tokens / total_chunks
    
    print("=" * 40)
    print("CORPUS INSPECTION STATISTICS")
    print("=" * 40)
    print(f"Total chunks: {total_chunks}")
    print(f"Total sections: {len(section_counts)}")
    print(f"Average chunk length: {avg_length:.1f} chars")
    print(f"Average tokens/chunk: {avg_tokens:.1f} tokens\n")
    
    print("=" * 40)
    print("SECTION DISTRIBUTION")
    print("=" * 40)
    for section, count in section_counts.most_common():
        safe_print(f"{section} : {count}")
    print("\n")
    
    print("=" * 40)
    print("FIRST 5 CHUNKS")
    print("=" * 40)
    for c in chunks[:5]:
        safe_print(f"ID     : {c.chunk_id}")
        safe_print(f"Page   : {c.page}")
        safe_print(f"Section: {c.section}")
        safe_print(f"Source : {c.source}")
        safe_print(f"Text   : {c.text[:500]}...\n")
        
    sorted_by_length = sorted(chunks, key=lambda x: len(x.text))
    
    print("=" * 40)
    print("TOP 10 LONGEST CHUNKS")
    print("=" * 40)
    for c in reversed(sorted_by_length[-10:]):
        safe_print(f"[{len(c.text)} chars] Section: {c.section} | Page: {c.page} | {c.text[:100]}...")
        
    print("\n" + "=" * 40)
    print("TOP 10 SHORTEST CHUNKS")
    print("=" * 40)
    for c in sorted_by_length[:10]:
        safe_print(f"[{len(c.text)} chars] Section: {c.section} | Page: {c.page} | {c.text[:100]}...")

if __name__ == "__main__":
    inspect_corpus()
