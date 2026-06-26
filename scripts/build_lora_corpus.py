import os
import re
import fitz
import pickle
import json
from src.retrieval.chunker import Chunker

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_corpus():
    pdf_path = "data/papers/lora.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    chunker = Chunker(chunk_size=5, overlap=1)
    
    all_chunks = []
    
    current_section = "Abstract"
    section_buffer = []
    
    total_sentences = 0
    total_tokens = 0
    
    def flush_section(section_name, page_no):
        nonlocal total_sentences, total_tokens
        if not section_buffer:
            return
            
        text = " ".join(section_buffer)
        if len(text) < 50:
            section_buffer.clear()
            return
            
        chunks = chunker.chunk_text(text, source="lora.pdf")
        for c in chunks:
            c.section = section_name
            c.page = page_no
            all_chunks.append(c)
            
            sents = list(chunker.nlp(c.text).sents)
            total_sentences += len(sents)
            total_tokens += len(c.text) // 4
            
        section_buffer.clear()

    last_page = 1
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        
        # Sort vertically
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        for b in blocks:
            if b[6] != 0:
                continue
                
            text = b[4].strip()
            if not text:
                continue
                
            # Skip pure numbers
            if text.isdigit() and len(text) < 5:
                continue
                
            lines = text.split('\n')
            first_line = lines[0].strip()
            lower_line = first_line.lower()
            
            if re.match(r'^([0-9]+\.?\s*)?(references|acknowledgements|acknowledgments|appendix)$', lower_line):
                flush_section(current_section, last_page)
                print(f"Hit stop section: {first_line}")
                return finalize_corpus(all_chunks, len(doc), total_sentences, total_tokens)
                
            # Detect new sections
            if len(first_line) < 80 and len(first_line.split()) <= 10:
                if re.match(r'^([0-9]+(\.[0-9]+)*\.?\s+)[A-Z]', first_line) or first_line.isupper() or first_line in ["Abstract", "Introduction", "Conclusion"]:
                    flush_section(current_section, last_page)
                    current_section = first_line

            cleaned = clean_text(text)
            
            # Skip small fragments that are not headers
            if len(cleaned) < 40 and current_section != cleaned:
                continue
                
            section_buffer.append(cleaned)
            last_page = page_num + 1

    flush_section(current_section, last_page)
    return finalize_corpus(all_chunks, len(doc), total_sentences, total_tokens)

def finalize_corpus(all_chunks, total_pages, total_sentences, total_tokens):
    print(f"Pages processed: {total_pages}")
    print(f"Sentences processed: {total_sentences}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total tokens estimated: {total_tokens}")
    
    if len(all_chunks) > 0:
        chunk_lengths = [len(c.text) for c in all_chunks]
        print(f"Average chunk length (chars): {sum(chunk_lengths)/len(chunk_lengths):.1f}")
        print(f"Min chunk length: {min(chunk_lengths)}")
        print(f"Max chunk length: {max(chunk_lengths)}")
    
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/lora_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
        
    json_data = [c.model_dump() for c in all_chunks]
    with open("data/processed/lora_chunks.json", "w") as f:
        json.dump(json_data, f, indent=2)
        
    print("Corpus saved successfully.")

if __name__ == "__main__":
    build_corpus()
