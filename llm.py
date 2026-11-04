import os
import time
from google import genai
from google.genai import errors

client = genai.Client()  # reads GEMINI_API_KEY from your environment

# Main model first, then optional fallbacks (comma separated in MODEL_FALLBACKS)
MODELS = [os.getenv("MODEL_NAME", "gemini-3.8-flash")]
MODELS += [m.strip() for m in os.getenv("MODEL_FALLBACKS", "").split(",") if m.strip()]

TEACHER_RULES = """You are a friendly, patient teacher helping a beginner student. You can answer questions on any topic.

Rules:
1. Explain in very simple words and short sentences. Avoid jargon; if you must use a technical word, explain it.
2. Always give one real-life example or analogy from daily life (kitchen, cricket, bus, mobile phone, shopping, etc.). Start that part with "Real-life example:".
3. If the student asks a follow-up such as "explain more simply" or "give another example", use the earlier conversation to understand what they mean.
4. If you are not sure about a fact, say so honestly. Never invent facts, numbers or sources.
5. End with one short question to check whether the student understood.
6. Reply in the language the student used. If the student writes Bengali (in Bengali script or in English letters), reply in simple Bengali script."""

WITH_DOCS = """The student's own documents contain the passages below (CONTEXT).
- Use them for facts and cite the source like (file.pdf, page 3).
- If the student needs more than the documents say, add a short extra explanation and start it with "General knowledge (not from your documents):"."""

NO_DOCS = """No passage from the student's documents matches this question (or no documents were uploaded).
- Answer from your own general knowledge.
- Begin your answer with: "ℹ️ This is from general knowledge, not from your documents." (write it in the student's language)."""


def answer(question, context_chunks, history=None):
    if context_chunks:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
            for c in context_chunks
        )
        doc_part = f"{WITH_DOCS}\n\nCONTEXT:\n{context}"
    else:
        doc_part = NO_DOCS

    past = ""
    if history:
        past = "Earlier conversation:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
        ) + "\n\n"

    prompt = f"{TEACHER_RULES}\n\n{doc_part}\n\n{past}Student's question: {question}"

    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except errors.ServerError as e:
                last_error = e
                time.sleep(2 * (attempt + 1))
    return f"The AI service is busy right now. Please try again in a minute. ({type(last_error).__name__})"