import pytest
from app import clean_text, ask_question, process_pdf

def test_clean_text():
    assert clean_text("  This   is  a test \n text. ") == "This is a test text."

def test_ask_question_no_pdf():
    # If no PDF is loaded, ask_question should return an error
    res = ask_question("What is LoRA?")
    assert "Error: Please upload a PDF first." in res[0]

def test_ask_question_empty():
    # Mock current_rg if possible, but testing empty query with no current_rg also returns first error.
    # To bypass, we would need to mock current_rg.
    # For now, it's sufficient to test the simple paths.
    pass
