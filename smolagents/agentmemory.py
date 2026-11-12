import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from smolagents import LogLevel, OpenAIModel, ToolCallingAgent


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

load_dotenv(Path(__file__).with_name(".env"))
token = os.getenv("OPENROUTER_API_KEY")
if not token:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to smolagents/.env or set it "
        "in your PowerShell session before running this script."
    )

model = OpenAIModel(
    model_id="openrouter/free",
    api_base="https://openrouter.ai/api/v1",
    api_key=token,
)

agent = ToolCallingAgent(
    tools=[],
    model=model,
    verbosity_level=LogLevel.OFF,
    instructions=(
        "You are a helpful recipe-book assistant. Use only the recipe context "
        "provided in each task. Remember earlier conversation turns when they "
        "help answer follow-up questions."
    ),
)

while True:
    query = input("User query (or type 'exit' to quit): ")
    if query.strip().lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    if not query.strip():
        continue

    relevant_docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = f"""Answer the user's question using only the recipe context below.
If the context does not contain the answer, say that you could not find it
in the recipe book. Do not invent ingredients or instructions.

Context:
{context}

User question:
{query}
"""

    response = agent.run(prompt, reset=False)
    code = agent.memory.return_full_code()
    print("\nGenerated Code:\n", code, "\n")
    print("=" * 40)
    print("\nAnswer:\n", response, "\n")
    