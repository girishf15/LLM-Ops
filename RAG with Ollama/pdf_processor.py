import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

load_dotenv()

PDF_FOLDER = "pdfs"
VECTORSTORE_PATH = "vector_db/global_index"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

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

print("Generating embeddings and saving vector store...")
vectorstore = FAISS.from_documents(all_chunks, embeddings)
vectorstore.save_local(VECTORSTORE_PATH)
print(f"Vector store saved in: {VECTORSTORE_PATH}")
