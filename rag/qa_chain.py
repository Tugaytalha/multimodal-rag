"""QA chain that feeds retrieved chunks to Gemma3 or Gemini vision model."""
from __future__ import annotations

from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.retriever import hybrid_retrieve


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
chain = load_qa_with_sources_chain(llm, chain_type="stuff")


def answer(query: str):
    docs: List[Document] = hybrid_retrieve(query)
    return chain({"question": query, "input_documents": docs}, return_only_outputs=True)
