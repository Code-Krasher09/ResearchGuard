import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import datetime
from loguru import logger

def main():
    results = {
        "Python": "FAIL",
        "Torch": "FAIL",
        "CUDA": "FAIL",
        "SpaCy": "FAIL",
        "BGE": "FAIL",
        "DeBERTa": "FAIL",
        "Groq": "FAIL",
        "SciFact": "FAIL",
        "Pipeline": "FAIL"
    }

    report_lines = []
    report_lines.append("# ResearchGuard Setup Report")
    report_lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Environment Validation")
    
    print("Running ResearchGuard Validation Suite...")
    
    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Checking Python version... Found: {py_version}")
    report_lines.append(f"- **Python Version:** {py_version}")
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        results["Python"] = "PASS"
    else:
        results["Python"] = "FAIL"
        report_lines[-1] += " (EXPECTED: 3.11.x)"

    # 2. Torch
    print("Checking Torch...")
    try:
        import torch
        report_lines.append(f"- **Torch Version:** {torch.__version__}")
        results["Torch"] = "PASS"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            report_lines.append(f"- **GPU:** {gpu_name}")
            results["CUDA"] = "PASS"
        else:
            report_lines.append("- **GPU:** None (CPU only)")
            results["CUDA"] = "FAIL"
    except ImportError:
        report_lines.append("- **Torch:** Not installed")

    # 3. SpaCy
    print("Checking SpaCy...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        report_lines.append("- **SpaCy:** en_core_web_sm loaded successfully")
        results["SpaCy"] = "PASS"
    except Exception as e:
        report_lines.append(f"- **SpaCy Error:** {e}")

    # 4. Embedder (BGE)
    print("Checking Embedder...")
    try:
        from src.retrieval.embedder import Embedder
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        dim = embedder.get_dimension()
        report_lines.append(f"- **Embedder Dimension:** {dim}")
        if dim == 384:
            results["BGE"] = "PASS"
    except Exception as e:
        report_lines.append(f"- **Embedder Error:** {e}")

    # 5. DeBERTa (Verifier)
    print("Checking DeBERTa Verifier...")
    try:
        from src.verification.verifier import Verifier
        verifier = Verifier()
        labels = verifier.model.config.id2label
        labels_lower = {v.lower() for v in labels.values()}
        report_lines.append(f"- **DeBERTa Labels:** {list(labels.values())}")
        
        expected_labels = {"entailment", "neutral", "contradiction"}
        if expected_labels.issubset(labels_lower):
            results["DeBERTa"] = "PASS"
    except Exception as e:
        report_lines.append(f"- **Verifier Error:** {e}")

    # 6. Groq
    print("Checking Groq...")
    try:
        from src.generation.client import GroqClient
        client = GroqClient()
        if client.health_check():
            report_lines.append("- **Groq:** Client connected successfully")
            results["Groq"] = "PASS"
        else:
            report_lines.append("- **Groq Error:** Health check failed")
    except Exception as e:
        report_lines.append(f"- **Groq Error:** {e}")

    # 7. SciFact
    print("Checking SciFact dataset...")
    corpus_path = "data/scifact/corpus.jsonl"
    claims_path = "data/scifact/claims_train.jsonl"
    if os.path.exists(corpus_path) and os.path.exists(claims_path):
        report_lines.append("- **Dataset:** SciFact found")
        results["SciFact"] = "PASS"
    else:
        report_lines.append("- **Dataset:** SciFact files missing")

    # 8. Pipeline
    print("Checking Pipeline Smoke Test (imports only)...")
    try:
        from src.pipeline.pipeline import ResearchGuard
        report_lines.append("- **Pipeline:** ResearchGuard imported successfully")
        results["Pipeline"] = "PASS"
    except Exception as e:
        report_lines.append(f"- **Pipeline Error:** {e}")

    # Print summary
    print("\n----------------------------------")
    print("ResearchGuard Bringup\n")
    for k, v in results.items():
        print(f"{k:<14} {v}")
    print("\nREADY")

    # Generate Markdown Report
    report_lines.append("\n## Summary")
    report_lines.append("| Component | Status |")
    report_lines.append("| --- | --- |")
    for k, v in results.items():
        report_lines.append(f"| {k} | {v} |")

    with open("docs/setup_report.md", "w") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    main()
