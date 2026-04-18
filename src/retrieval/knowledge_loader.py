from __future__ import annotations

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import KNOWLEDGE_BASE_DIR


def load_knowledge_documents():
    """
    Load markdown knowledge base files.
    """
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    return loader.load()


def split_knowledge_documents(
    docs,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """
    Split documents into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)