import streamlit as st
from rag import load_pdf, chunk_text, build_index, retrieve
from llm import answer

st.set_page_config(page_title="Chat with your Documents", page_icon="📄")
st.title("📄 Chat with your Documents")

files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if files:
    key = tuple(f.name for f in files)
    if st.session_state.get("key") != key:
        with st.spinner("Reading documents..."):
            text = "\n".join(load_pdf(f) for f in files)
            chunks = chunk_text(text)
            st.session_state.update(key=key, chunks=chunks, index=build_index(chunks))
        st.success(f"Ready! Indexed {len(st.session_state.chunks)} chunks.")

    question = st.text_input("Ask a question about your documents")
    if question:
        with st.spinner("Thinking..."):
            top = retrieve(question, st.session_state.chunks, st.session_state.index)
            st.write(answer(question, top))
        with st.expander("Sources used"):
            for c in top:
                st.caption(c)
else:
    st.info("Upload a PDF to get started.")