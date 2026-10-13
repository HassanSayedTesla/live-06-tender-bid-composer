"""Central HF-only model configuration."""
import os

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Workers: fast + cheap. Manager: strong reasoner.
WORKER_MODEL_ID = os.getenv("WORKER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MANAGER_MODEL_ID = os.getenv("MANAGER_MODEL_ID", "deepseek-ai/DeepSeek-R1")
# Fallback if 72B rate-limited on free tier:
WORKER_FALLBACK_MODEL_ID = os.getenv("WORKER_FALLBACK_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
MAX_STEPS_WORKER = int(os.getenv("MAX_STEPS_WORKER", "8"))
MAX_STEPS_MANAGER = int(os.getenv("MAX_STEPS_MANAGER", "12"))
PLANNING_INTERVAL = int(os.getenv("PLANNING_INTERVAL", "3"))
VAT_RATE = float(os.getenv("VAT_RATE", "0.14"))  # Egypt 14%
