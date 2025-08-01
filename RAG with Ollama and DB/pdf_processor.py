import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings

load_dotenv()

PDF_FOLDER = "pdfs"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_embeddings")

CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
COLLECTION_NAME = "document_embeddings"

print(f"🔗 Connecting to PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

all_chunks = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        print(f"Loading {file}...")
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

print("Generating embeddings and storing in PostgreSQL...")
try:
    vectorstore = PGVector.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        use_jsonb=True
    )
    print(f"Successfully stored {len(all_chunks)} document chunks in PostgreSQL!")
    print(f"Database: {POSTGRES_DB}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Connection: {POSTGRES_HOST}:{POSTGRES_PORT}")
    
except Exception as e:
    print(f"Error storing embeddings in PostgreSQL: {e}")