import gradio as gr
import fitz
import re
import json
import time
from src.retrieval.chunker import Chunker
from src.retrieval.retriever import Retriever
from src.pipeline.pipeline import ResearchGuard

# Global instances
current_rg = None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdf(pdf_file):
    global current_rg
    if pdf_file is None:
        return "No PDF uploaded."
    
    # Process PDF into chunks
    doc = fitz.open(pdf_file.name)
    chunker = Chunker(chunk_size=5, overlap=1)
    
    text_buffer = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_buffer.append(page.get_text())
        
    full_text = clean_text(" ".join(text_buffer))
    chunks = chunker.chunk_text(full_text, source=pdf_file.name)
    
    # Update retriever
    retriever = Retriever()
    retriever.build_from_chunks(chunks)
    current_rg = ResearchGuard()
    current_rg.pipeline.retriever = retriever
    
    return f"PDF loaded and indexed successfully. ({len(chunks)} chunks extracted)"

def ask_question(question):
    global current_rg
    if current_rg is None:
        return "Error: Please upload a PDF first.", "", "", "", "", "", ""
        
    if not question.strip():
        return "Error: Empty question.", "", "", "", "", "", ""
        
    retrieved = current_rg.pipeline.retriever.retrieve(question, k=5)
    retrieved_formatted = []
    for c in retrieved:
        chunk_info = f"**Chunk ID**: {c.chunk_id}\n**Page Number**: {c.page if c.page is not None else 'Unknown'}\n**Section**: {c.section if c.section else 'Unknown'}\n**Source**: {c.source}\n\n{c.text}"
        retrieved_formatted.append(chunk_info)
    retrieved_text = "\n\n---\n\n".join(retrieved_formatted)
    
    result = current_rg.run(question)
    
    # Format Results
    answer = result.answer
    claims_list = result.repair_result.claims or []
    claims = "\n".join([f"- {c.text} [{c.claim_type}]" for c in claims_list])
    
    nli_results = ""
    verifications = result.repair_result.verification_results or []
    for r in verifications:
        nli_results += f"Claim: {r.claim}\nLabel: {r.label} (Conf: {r.top_confidence:.2f})\nEvidence: {r.top_evidence}\n\n"
        
    judgment = f"Faithfulness: {result.judgment.faithfulness_score:.2f}\nRepair Needed: {result.judgment.repair_needed}\nReason: {result.judgment.reason}"
    
    repair_attempts = f"Total Attempts: {result.repair_result.attempt}\nHistory: {result.repair_result.repair_history}"
    
    export_json = result.model_dump_json(indent=2)
    
    return retrieved_text, answer, claims, nli_results, judgment, repair_attempts, export_json

def export_to_json(export_str):
    if not export_str:
        return None
    with open("export.json", "w") as f:
        f.write(export_str)
    return "export.json"


with gr.Blocks(title="ResearchGuard") as demo:
    gr.Markdown("# ResearchGuard\n### Self-Healing Hallucination Detection Framework")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Upload PDF")
            pdf_input = gr.File(label="Upload Research Paper", file_types=[".pdf"])
            pdf_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("## 2. Ask Question")
            query_input = gr.Textbox(label="Question", lines=3)
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                clear_btn = gr.Button("Clear")
            export_btn = gr.Button("Export JSON")
            export_file = gr.File(label="Export File")
            
        with gr.Column(scale=2):
            gr.Markdown("## 3. Results")
            answer_output = gr.Textbox(label="Answer", lines=4)
            with gr.Row():
                judgment_output = gr.Textbox(label="Judgment", lines=3)
                repair_output = gr.Textbox(label="Repair Attempts", lines=3)
            with gr.Accordion("Retrieved Chunks", open=False):
                chunks_output = gr.Textbox(label="Top 5 Chunks", lines=10)
            with gr.Accordion("Extracted Claims", open=False):
                claims_output = gr.Textbox(label="Claims", lines=5)
            with gr.Accordion("NLI Verification", open=False):
                nli_output = gr.Textbox(label="NLI Results", lines=10)
                
            export_state = gr.State()

    pdf_input.upload(process_pdf, inputs=[pdf_input], outputs=[pdf_status])
    run_btn.click(ask_question, inputs=[query_input], outputs=[chunks_output, answer_output, claims_output, nli_output, judgment_output, repair_output, export_state])
    clear_btn.click(lambda: ("", "", "", "", "", "", ""), outputs=[query_input, answer_output, judgment_output, repair_output, chunks_output, claims_output, nli_output])
    export_btn.click(export_to_json, inputs=[export_state], outputs=[export_file])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
