import os
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment
MODEL = os.getenv("MODEL_NAME", "claude-sonnet-5-5")


def answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text