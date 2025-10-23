 
""" 
PDF Processor Module
====================
 
This module handles PDF document processing:
1. Reads PDF files
2. Extracts text from each page
3. Splits text into smaller chunks for better retrieval
4. Generates embeddings for each chunk using Azure OpenAI
Author: HR RAG Bot Team
Date: October 2025
"""
from PyPDF2 import PdfReader
from openai import AzureOpenAI 
from typing import List, Dict

class PDFProcessor: 
    """
    PDFProcessor handles the extraction and embedding of PDF documents.
    Key responsibilities:
    - Extract text from PDF files
    - Split text into manageable chunks
    - Generate embeddings using Azure OpenAI
    """
 
    def __init__(self, azure_endpoint: str, azure_api_key: str,
                 api_version: str, embedding_model: str): 
        """ 
        Initialize the PDF Processor with Azure OpenAI credentials.
        Args:
            azure_endpoint (str): Azure OpenAI endpoint URL
            azure_api_key (str): Azure OpenAI API key
            api_version (str): API version (e.g., "2024-02-01")
            embedding_model (str): Deployment name of embedding model
        """
        # Initialize Azure OpenAI client for generating embeddings
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.embedding_model = embedding_model
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from each page of a PDF file.
        Args:
            pdf_path (str): Path to the PDF file
        Returns:
            List[Dict]: List of dictionaries containing page number and text
        Example:
            [
                {"page": 1, "text": "Content of page 1..."},
                {"page": 2, "text": "Content of page 2..."}
            ]
        """
        print(f" Reading PDF: {pdf_path}")
        # Open and read the PDF file
        reader = PdfReader(pdf_path)
        pages_data = []
        # Extract text from each page
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add pages with content
                pages_data.append({
                    "page": page_num,
                    "text": text
                })
        print(f" Extracted text from {len(pages_data)} pages")
        return pages_data
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000,
                               overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        Why chunking?
        - Makes text more manageable for embeddings
        - Improves retrieval accuracy
        - Prevents exceeding token limits
        Args:
            text (str): Text to split
            chunk_size (int): Maximum characters per chunk (default: 1000)
            overlap (int): Characters to overlap between chunks (default: 200)
        Returns:
            List[str]: List of text chunks
        Example:
            Input: "This is a long document..." (5000 chars)
            Output: ["This is a long...", "...document continues...", ...]
        """
        chunks = []
        start = 0
        text_length = len(text)
        # Split text into overlapping chunks
        while start < text_length:
            # Get chunk from start to start + chunk_size
            end = start + chunk_size
            chunk = text[start:end]
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            # Move start position forward (with overlap)
            start += chunk_size - overlap
        return chunks
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for a piece of text using Azure OpenAI.
        What is an embedding?
        - A numerical representation of text (vector of numbers)
        - Similar texts have similar embeddings
        - Used for semantic search
        Args:
            text (str): Text to embed
        Returns:
            List[float]: Vector embedding (1536 dimensions for ada-002)
        Example:
            Input: "What is the vacation policy?"
            Output: [0.123, -0.456, 0.789, ..., 0.321] (1536 numbers)
        """
        # Call Azure OpenAI to generate embedding
 
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
 
        # Extract the embedding vector from response
        return response.data[0].embedding
 
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*70}")
 
    # Step 1: Extract text from all pages
        pages_data = self.extract_text_from_pdf(pdf_path)
        documents = []
        chunk_counter = 0
 
    # Step 2: Process each page
        for page_data in pages_data:
            page_num = page_data["page"]
            text = page_data["text"]
 
        # Split page text into chunks
            chunks = self.split_text_into_chunks(text, chunk_size, overlap)
            print(f" Page {page_num}: Created {len(chunks)} chunks")
 
        # Step 3: Generate embeddings for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    embedding = self.generate_embedding(chunk)
                # Include page and chunk_index
                    document = {
                        "content": chunk,
                        "embedding": embedding,
                        "source": pdf_path.split("/")[-1],  # filename only
                        "page": page_num,                   # ✅ page number
                        "chunk_index": chunk_counter        # ✅ chunk index
                    }
                    documents.append(document)
                    chunk_counter += 1
                except Exception as e:
                    print(f" Error processing chunk {chunk_counter}: {e}")
                continue
 
        print(f"\n Total documents created: {len(documents)}")
        return documents
 