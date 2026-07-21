from pathlib import Path

#Base_file : Path
BASE_DIR=Path(__file__).resolve().parent

#Document : Path
DOC_DIR=BASE_DIR / "Documents"

#Data_base: Path
CHROMA_DB_DIR=DOC_DIR / "chroma_db"

#Chroma_collection
COLLECTION_NAME ="multi_pdf_rag"

#Models:
EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "minimax-m3:cloud"

#Chunking _setting
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
CHUNKING_VERSION = "word_chunker_v1"

#Retrieval_settings
TOP_K=3
DISTANCE_METRIC="cosine"

# Remove database records when a PDF is removed
PRUNE_DELETED_FILES = True

