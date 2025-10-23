"""Configuration settings for Azure Cosmos DB for MongoDB vCore.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HR_DOCS_DIR = DATA_DIR / "hr_documents"

# Azure Cosmos DB for MongoDB vCore Configuration
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
COSMOS_DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "hr_knowledge_base")
COSMOS_COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME", "hr_policies")
VECTOR_INDEX_TYPE = os.getenv("VECTOR_INDEX_TYPE", "vector-hnsw")

# Azure OpenAI Configuration
# Embedding Configuration
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
EMBEDDING_MODEL_DEPLOYMENT = os.getenv("EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-ada-002")

# Chat Configuration
AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
CHAT_MODEL_DEPLOYMENT = os.getenv("CHAT_MODEL_DEPLOYMENT", "gpt-4o")

# API Version
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Document Processing Configuration (UNCHANGED)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

def validate_config():
    """Validate required configuration."""
    required_vars = {
        "COSMOS_CONNECTION_STRING": COSMOS_CONNECTION_STRING,
        "AZURE_OPENAI_EMBEDDING_ENDPOINT": AZURE_OPENAI_EMBEDDING_ENDPOINT,
        "AZURE_OPENAI_EMBEDDING_KEY": AZURE_OPENAI_EMBEDDING_KEY,
        "AZURE_OPENAI_CHAT_ENDPOINT": AZURE_OPENAI_CHAT_ENDPOINT,
        "AZURE_OPENAI_CHAT_API_KEY": AZURE_OPENAI_CHAT_API_KEY,
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please check your .env file."
        )

if __name__ != "__main__":
    validate_config()