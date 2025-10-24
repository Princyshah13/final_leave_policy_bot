"""
Document Retrieval Script
==========================
Simple script to retrieve relevant documents from Cosmos DB based on user queries.

This demonstrates the RETRIEVAL part of RAG (Retrieval-Augmented Generation):
- User asks a question
- Question is converted to an embedding
- Similar documents are found using vector search
- Relevant documents are displayed

Author: HR RAG Bot Team
Date: October 2025
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.setting import (
    validate_config,
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_KEY,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_MODEL_DEPLOYMENT,
    COSMOS_CONNECTION_STRING,
    COSMOS_DATABASE_NAME,
    COSMOS_COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    VECTOR_INDEX_TYPE
)

from utils.cosmos_db import CosmosVectorDB
from openai import AzureOpenAI  # Assuming this is a wrapper you've created


class DocumentRetriever:
    """Retrieves relevant documents for user queries using embeddings."""

    def __init__(self):
        print(" Initializing Document Retriever...")

        # Validate configuration
        validate_config()

        # Initialize Azure OpenAI client
        self.embedding_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

        # Initialize Cosmos DB vector store
        self.cosmos_db = CosmosVectorDB(
            connection_string=COSMOS_CONNECTION_STRING,
            database_name=COSMOS_DATABASE_NAME,
            collection_name=COSMOS_COLLECTION_NAME,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            vector_index_type=VECTOR_INDEX_TYPE
        )

        print(" Retriever initialized successfully!\n")

    def generate_query_embedding(self, query: str) -> list:
        """Convert user's question into an embedding vector."""
        response = self.embedding_client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL_DEPLOYMENT
        )
        return response.data[0].embedding

    def retrieve_documents(self, query: str, top_k: int = 5):
        """Find documents relevant to the query."""
        print(f" Processing query: '{query}'")
        print(" Generating query embedding...")
        query_embedding = self.generate_query_embedding(query)

        print(f" Searching for top {top_k} relevant documents...")
        results = self.cosmos_db.vector_search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        return results

    def display_results(self, results: list):
        """Display search results in a readable format without duplicate sources."""
        if not results:
            print("\n No relevant documents found.")
            return

        # ✅ Deduplicate sources
        unique_results = []
        seen_sources = set()
        for doc in results:
            source = doc.get('source', 'Unknown')
            if source not in seen_sources:
                unique_results.append(doc)
                seen_sources.add(source)

        print(f"\n{'='*80}")
        print(f"Found {len(unique_results)} unique relevant documents:")
        print(f"{'='*80}\n")

        for i, doc in enumerate(unique_results, 1):
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            page = doc.get('page', 'N/A')
            score = doc.get('similarity_score', 0.0)

            print(f"{'─'*80}")
            print(f" Result #{i}")
            print(f"{'─'*80}")
            print(f" Source: {source}")
            print(f" Page: {page}")
            print(f" Similarity Score: {score:.4f}")
            print(f"\n Content Preview:")
            print(f"{content[:400]}...")  # Show first 400 characters
            print(f"{'─'*80}\n")


def main():
    """Interactive retrieval demo."""
    print("\n" + "="*80)
    print(" HR DOCUMENT RETRIEVAL SYSTEM")
    print("="*80 + "\n")

    retriever = DocumentRetriever()
    doc_count = retriever.cosmos_db.count_documents()
    print(f" Total documents in database: {doc_count}\n")

    if doc_count == 0:
        print("  No documents found in database!")
        print("   Please run 'embed_documents.py' first to load documents.\n")
        return

    print(" Instructions:")
    print("   - Enter your questions about HR policies")
    print("   - Type 'quit' or 'exit' to stop")
    print("   - Type 'examples' to see sample questions\n")

    while True:
        try:
            query = input(" Your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break

            if query.lower() == 'examples':
                print("\n Example Questions:")
                print("   What is the vacation policy?")
                print("   How many sick days do employees get?")
                print("   What are the working hours?")
                print("   What is the remote work policy?")
                print("   How do I apply for leave?\n")
                continue

            if not query:
                continue

            results = retriever.retrieve_documents(query, top_k=3)
            retriever.display_results(results)

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)