from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

from src.config.settings import PROJECT_ROOT
from src.agent.summarizer import load_llm
from src.inference.pipeline import load_inference_artifacts
from src.retrieval.knowledge_loader import (
    load_knowledge_documents,
    split_knowledge_documents,
)
from src.retrieval.vectorstore import (
    get_retriever,
    load_or_build_vectorstore,
)

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache
def get_preprocessor_and_model():
    preprocessor, model = load_inference_artifacts()
    return preprocessor, model


@lru_cache
def get_retriever_dependency():
    docs = load_knowledge_documents()
    split_docs = split_knowledge_documents(docs)
    vectorstore = load_or_build_vectorstore(split_docs=split_docs)
    retriever = get_retriever(vectorstore)
    return retriever


@lru_cache
def get_llm_dependency():
    return load_llm()