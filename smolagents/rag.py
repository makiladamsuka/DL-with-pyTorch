from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdf"

loader = PyPDFDirectoryLoader(str(PDF_DIRECTORY))
documents = loader.load()
if not documents:
    raise FileNotFoundError(
        f"No PDF files found in {PDF_DIRECTORY}. Add at least one PDF before running rag.py."
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)
if not chunks:
    raise ValueError("The PDF files produced no text chunks.")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = FAISS.from_documents(chunks, embedder)

query = "How do I cook salmon with herbs?"
# Similarity search
relevant_docs = vectorstore.similarity_search(query, k=3)

context = "\n\n".join(doc.page_content for doc in relevant_docs)

print("Context:\n", context)