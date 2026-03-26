
"""
RAG-Based Mini Q&A System
━━━━━━━━━━━━━━━━━━━━━━━━
Framework : LangChain
Embeddings: Ollama (nomic-embed-text)
Vector DB : FAISS
LLM       : Ollama (tinyllama)
"""

import sys
import textwrap
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a document Q&A assistant.

Use ONLY the provided context to answer the question.
If the question is general (like asking for summary), summarize the context.

If the answer is not found, say:
"I couldn't find that information in the document."

Context:
{context}

Question: {question}

Answer:""",
)


# ─────────────────────────────────────────────────────────────
# STEP 1 + 2: Load & Split
# ─────────────────────────────────────────────────────────────

def load_and_split(filepath: str, chunk_size: int = 500, chunk_overlap: int = 100):
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    if ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .txt or .pdf")

    print(f"[RAG] Loading document: {filepath}")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"[RAG] Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3 + 4: Embeddings + FAISS
# ─────────────────────────────────────────────────────────────

def get_embeddings(base_url: str = "http://localhost:11434"):
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url,
    )


def build_vectorstore(chunks, base_url: str = "http://localhost:11434", cache_dir: str = None):
    embeddings = get_embeddings(base_url)

    if cache_dir and Path(cache_dir).exists():
        print(f"[RAG] Loading cached FAISS index from '{cache_dir}/'")
        return FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)

    print(f"[RAG] Embedding {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if cache_dir:
        vectorstore.save_local(cache_dir)
        print(f"[RAG] Saved FAISS index to '{cache_dir}/'")

    return vectorstore


# ─────────────────────────────────────────────────────────────
# STEP 5 + 6: Retrieval + LLM
# ─────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore, base_url="http://localhost:11434", model="tinyllama", top_k=3):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    return chain


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        document_path: str,
        base_url: str = "http://localhost:11434",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
        model: str = "tinyllama",
        use_cache: bool = True,
    ):
        cache_dir = f".rag_cache_{Path(document_path).stem}" if use_cache else None

        # Load or build vectorstore
        if cache_dir and Path(cache_dir).exists():
            embeddings = get_embeddings(base_url)
            vectorstore = FAISS.load_local(
                cache_dir, embeddings, allow_dangerous_deserialization=True
            )
            print(f"[RAG] Loaded cached index")
        else:
            chunks = load_and_split(document_path, chunk_size, chunk_overlap)
            vectorstore = build_vectorstore(chunks, base_url, cache_dir)

        self.vectorstore = vectorstore
        self.chain = build_rag_chain(vectorstore, base_url, model, top_k)

        print(f"[RAG] Ready (model={model})\n")

    # ─────────────────────────────
    # Query (normal RAG)
    # ─────────────────────────────
    def query(self, question: str, verbose: bool = False) -> str:
        result = self.chain.invoke({"query": question})
        answer = result["result"].strip()

        if verbose:
            print("\n--- Retrieved Chunks ---")
            for i, doc in enumerate(result.get("source_documents", []), 1):
                preview = textwrap.shorten(doc.page_content, width=120)
                print(f"[{i}] {preview}")
            print("------------------------\n")

        return answer

    # ─────────────────────────────
    # Summarization (NEW)
    # ─────────────────────────────
    def summarize(self):
        docs = self.vectorstore.docstore._dict.values()
        full_text = "\n".join([doc.page_content for doc in docs])

        # Limit size for LLM
        full_text = full_text[:3000]

        prompt = f"""
Summarize the following document in a clear and concise way:

{full_text}
"""

        response = self.chain.llm.invoke(prompt)
        return response.content
