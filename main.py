import sys
from pathlib import Path 
import utils.setting as config
from utils.setting import validate_config
from utils.pdf_processor import PDFProcessor
from utils.cosmos_db import CosmosVectorDB
 
def main():
    """Main embedding pipeline."""
    try:
        validate_config()

    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
 
    print("\n" + "="*70)
    print("HR DOCUMENT EMBEDDING PIPELINE")
    print("="*70 + "\n")
 
    # Initialize PDF Processor
    print(" Initializing PDF Processor...")
    pdf_processor = PDFProcessor(
        azure_endpoint=config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
        azure_api_key=config.AZURE_OPENAI_EMBEDDING_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        embedding_model=config.EMBEDDING_MODEL_DEPLOYMENT
    )
    # Initialize Cosmos DB

    print(" Initializing Cosmos DB MongoDB vCore Vector Store...")

    cosmos_db = CosmosVectorDB(
        connection_string=config.COSMOS_CONNECTION_STRING,
        database_name=config.COSMOS_DATABASE_NAME,
        collection_name=config.COSMOS_COLLECTION_NAME,
        embedding_dimensions=config.EMBEDDING_DIMENSIONS,
        vector_index_type=config.VECTOR_INDEX_TYPE
    )
    print("\nDeleting old documents from Cosmos DB...")

    deleted_count = cosmos_db.delete_all_documents()
    print( f"Deleted {deleted_count} old documents.\n")
 
    # Ensure data directory exists
    data_dir = Path(config.DATA_DIR)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: The directory {data_dir} does not exist or is not a directory.")
        return
 
    # Find all PDF files
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f" No PDF files found in {data_dir}")
        return
    print(f"\n Found {len(pdf_files)} PDF file(s) to process:\n")

    for pdf_file in pdf_files:
        print(f"   • {pdf_file.name}")
    print()
 
    # Process each PDF

    total_documents = 0
    for pdf_file in pdf_files:
        try:
            print(f" Processing {pdf_file.name}...")
            documents = pdf_processor.process_pdf(
                pdf_path=str(pdf_file),
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )

            if documents:
                print("\n================Sample document structure:")
                print(documents[0])
 
            if documents:
                print(f"  Storing {len(documents)} documents in Cosmos DB...")
                inserted = cosmos_db.insert_documents(documents)
                total_documents += inserted
                print(f"Successfully stored {inserted} documents\n")
            else:
                print(f" No documents created from {pdf_file.name}\n")

        except Exception as e:
            print(f" Error processing {pdf_file.name}: {e}")
            continue
    # Summary

    print("\n" + "="*70)
    print(f" EMBEDDING COMPLETE")
    print(f"   Total documents embedded: {total_documents}")
    print("="*70 + "\n")
 
if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\n Process interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
 