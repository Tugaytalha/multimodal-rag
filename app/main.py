"""Gradio UI for Multimodal RAG."""
import gradio as gr
import sys, pathlib
# Ensure project root on sys.path so `rag` package is importable when running directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.append(PROJECT_ROOT.as_posix())

from rag.qa_chain import answer


def qa_fn(q):
    result = answer(q)
    return result["output_text"]


demo = gr.Interface(fn=qa_fn, inputs="text", outputs="text", title="Multimodal RAG")

if __name__ == "__main__":
    demo.launch()
