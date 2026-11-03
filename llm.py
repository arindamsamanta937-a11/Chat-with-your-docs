import os
import time
from google import genai
from google.genai import errors

client = genai.Client()  # reads GEMINI_API_KEY from your environment
MODEL = os.getenv("MODEL_NAME", "gemini-3.8-flash")


def answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except errors.ServerError:
            if attempt == 2:
                return "The AI service is busy right now. Please try again in a minute."
            time.sleep(2 * (attempt + 1))