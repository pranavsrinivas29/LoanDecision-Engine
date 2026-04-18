from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agent.prompts import LOAN_SUMMARY_PROMPT
from src.config.settings import OPENAI_MODEL_NAME, TOP_K_RETRIEVAL, PROJECT_ROOT
from src.retrieval.vectorstore import retrieve_docs
from src.utils.formatters import format_dict_pretty, format_retrieved_context

load_dotenv(PROJECT_ROOT / ".env")


def load_llm(model_name: str = OPENAI_MODEL_NAME):
    """
    Load chat model for summarization and Q&A.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to project root .env or export it in the shell."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )


def answer_loan_question(
    question: str,
    input_data: dict,
    prediction_result: dict,
    local_explanation: dict,
    retriever,
    llm=None,
    top_k: int = TOP_K_RETRIEVAL,
) -> dict:
    """
    Generate a grounded answer using:
    - inference inputs
    - prediction result
    - local explanation
    - retrieved markdown knowledge
    """
    if llm is None:
        llm = load_llm()

    retrieved_docs = retrieve_docs(retriever, question)
    retrieved_context = format_retrieved_context(retrieved_docs)

    messages = LOAN_SUMMARY_PROMPT.format_messages(
        input_data=format_dict_pretty(input_data),
        prediction_result=format_dict_pretty(prediction_result),
        local_explanation=format_dict_pretty(local_explanation),
        retrieved_context=retrieved_context,
        question=question,
    )

    response = llm.invoke(messages)

    return {
        "question": question,
        "answer": response.content,
        "retrieved_docs": [
            {
                "source": doc.metadata.get("source"),
                "content": doc.page_content,
            }
            for doc in retrieved_docs[:top_k]
        ],
    }