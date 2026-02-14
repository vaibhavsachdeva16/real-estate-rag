from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------- CONFIG ---------------- #

CHUNK_SIZE = 400
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


# ---------------- INIT ---------------- #

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500
        )

    if vector_store is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )


# ---------------- PROCESS URLs ---------------- #

def process_urls(urls):
    yield "Initializing components..."
    initialize_components()

    yield "Loading data..."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    loader = UnstructuredURLLoader(
        urls=urls,
        headers=headers
    )

    documents = loader.load()

    yield "Splitting into chunks..."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    yield "Storing embeddings..."

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=uuids)

    yield "URLs processed successfully!"


# ---------------- QUESTION ANSWERING ---------------- #

def generate_answer(query):
    initialize_components()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    result = chain.invoke(
        {"question": query},
        return_only_outputs=True
    )

    return result["answer"], result.get("sources", "")
