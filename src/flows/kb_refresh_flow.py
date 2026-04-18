from __future__ import annotations

from prefect import flow, task, get_run_logger

from src.retrieval.knowledge_loader import (
    load_knowledge_documents,
    split_knowledge_documents,
)
from src.retrieval.vectorstore import build_vectorstore, get_retriever, retrieve_docs


@task
def load_and_split_docs_task():
    docs = load_knowledge_documents()
    split_docs = split_knowledge_documents(docs)
    return split_docs


@task
def rebuild_vectorstore_task(split_docs):
    vectorstore = build_vectorstore(split_docs)
    return vectorstore


@task
def validate_retrieval_task(vectorstore) -> dict:
    retriever = get_retriever(vectorstore)
    query = "What improves loan approval chances?"
    docs = retrieve_docs(retriever, query)

    return {
        "query": query,
        "num_docs_retrieved": len(docs),
        "sources": [doc.metadata.get("source") for doc in docs],
    }


@task
def log_kb_refresh_result_task(result: dict):
    logger = get_run_logger()
    logger.info("Knowledge base refresh completed.")
    logger.info(f"Validation query: {result['query']}")
    logger.info(f"Retrieved docs: {result['num_docs_retrieved']}")
    logger.info(f"Sources: {result['sources']}")


@flow(name="loan-approval-kb-refresh-flow")
def kb_refresh_flow() -> dict:
    """
    Prefect flow for rebuilding the Chroma knowledge base and validating retrieval.
    """
    split_docs = load_and_split_docs_task()
    vectorstore = rebuild_vectorstore_task(split_docs)
    result = validate_retrieval_task(vectorstore)
    log_kb_refresh_result_task(result)
    return result


if __name__ == "__main__":
    result = kb_refresh_flow()
    print(result)