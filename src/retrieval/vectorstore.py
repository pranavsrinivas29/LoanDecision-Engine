from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config.settings import CHROMA_DIR, TOP_K_RETRIEVAL, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def _get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to project root .env or export it in the shell."
        )
    return OpenAIEmbeddings(api_key=api_key)


def build_vectorstore(split_docs):
    """
    Build and persist a Chroma vector store from documents.
    """
    embeddings = _get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def load_or_build_vectorstore(split_docs=None):
    """
    Load persisted vectorstore if it exists, otherwise build it.
    """
    embeddings = _get_embeddings()

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0

    if count == 0:
        if split_docs is None:
            raise ValueError(
                "No existing vectorstore found and no documents provided to build one."
            )
        vectorstore = build_vectorstore(split_docs)

    return vectorstore


def get_retriever(vectorstore, top_k: int = TOP_K_RETRIEVAL):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve_docs(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)

    raise AttributeError("Retriever has neither 'invoke' nor 'get_relevant_documents'.")