import streamlit as st
from rag import load_pdf, chunk_pages, build_index, retrieve
from llm import answer

st.set_page_config(page_title="AI Teacher", page_icon="🎓")
st.title("🎓 AI Teacher")
st.caption("Ask me anything. Upload PDFs if you want me to teach from your own documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar: optional documents ----------
with st.sidebar:
    st.header("Your documents (optional)")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Index documents only when the uploaded files change ----------
if not files:
    for k in ("chunks", "index", "key"):
        st.session_state.pop(k, None)
else:
    key = tuple(f.name for f in files)
    if st.session_state.get("key") != key:
        with st.spinner("Reading documents..."):
            pages = []
            for f in files:
                pages.extend(load_pdf(f, f.name))
            chunks = chunk_pages(pages)
        if chunks:
            st.session_state.update(key=key, chunks=chunks, index=build_index(chunks))
            st.sidebar.success(f"Ready! {len(chunks)} chunks from {len(pages)} pages.")
        else:
            st.sidebar.error("No readable text found. This may be a scanned PDF. Please upload a text-based PDF.")


def show_sources(sources):
    with st.expander("Sources used"):
        for c in sources:
            st.markdown(f"**{c['source']} — page {c['page']}**")
            st.caption(c["text"])


# ---------- Show the conversation so far ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

# ---------- New question ----------
question = st.chat_input("Ask your teacher anything...")
if question:
    history = list(st.session_state.messages)  # conversation before this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            top = []
            if st.session_state.get("chunks"):
                top = retrieve(question, st.session_state.chunks, st.session_state.index)
            reply = answer(question, top, history)
        st.markdown(reply)
        if top:
            show_sources(top)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": top})