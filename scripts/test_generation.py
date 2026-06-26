import time
from loguru import logger
from src.generation.client import GroqClient

def main():
    logger.info("Initializing Groq Generation Smoke Test")
    
    # Instantiate client
    client = GroqClient()
    
    # We explicitly ask the model to include a thinking block so we can verify the cleaner
    # Some models natively do this (like deepseek-r1), but we inject a prompt to be safe.
    query = "What is LoRA? Format your output with <think>...reasoning...</think> followed by your final answer."
    
    logger.info("Requesting raw answer...")
    raw_start = time.perf_counter()
    raw_answer = client.generate(query, clean_think=False, model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=1024)
    raw_latency = time.perf_counter() - raw_start
    
    logger.info("Requesting cleaned answer (using native GroqClient cleanup)...")
    clean_start = time.perf_counter()
    cleaned_answer = client.generate(query, clean_think=True, model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=1024)
    clean_latency = time.perf_counter() - clean_start
    
    print("\n--- RAW ANSWER ---")
    print(raw_answer)
    print("\n--- CLEANED ANSWER ---")
    print(cleaned_answer)
    print("------------------\n")
    
    logger.info(f"Raw Answer Generation Latency: {raw_latency:.2f} s")
    logger.info(f"Cleaned Answer Generation Latency: {clean_latency:.2f} s")

if __name__ == "__main__":
    main()
