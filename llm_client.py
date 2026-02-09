# llm_client.py
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "llama3:8b"  # or mistral:7b

SYSTEM_PROMPT = """
You are a document question-answering system.

You must answer ONLY using the provided context from the PDF.
Do not use any outside knowledge.
If the answer is not found in the context, reply exactly:
"The document does not contain this information."
"""
class LLMClient:
    @staticmethod
    def ask_llm(question, retrieved_chunks):
        print(f" Retrieved Chunks for LLM: {question}, chunk: {retrieved_chunks}")
        context_text = ""
        for c in retrieved_chunks:
            context_text += f"[Page {c['page']}]\n{c['text']}\n\n"

        prompt = f"""
    {SYSTEM_PROMPT}

    Context:
    {context_text}

    Question:
    {question}
    """

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"]
