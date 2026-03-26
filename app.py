
import streamlit as st
import tempfile
import os
from rag_app import RAGPipeline

st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("📄 RAG-Based Mini Q&A System")
st.write("Upload a document and ask questions based on its content.")

# Upload file
uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    # ✅ Preserve file extension (IMPORTANT FIX)
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.success("✅ Document uploaded successfully!")

    # Initialize RAG only once
    if "rag" not in st.session_state:
        with st.spinner("🔄 Processing document..."):
            st.session_state.rag = RAGPipeline(
                document_path=temp_path,
                chunk_size=500,
                chunk_overlap=100,
                top_k=3
            )
        st.success("✅ RAG system ready!")

    rag = st.session_state.rag

    # User query
    query = st.text_input("💬 Ask a question from the document:")

    if query:
        with st.spinner("🤖 Generating answer..."):

            # ✅ Detect summary-type queries
            if any(word in query.lower() for word in ["summary", "summarize", "overview", "what is this"]):
                answer = rag.summarize()
            else:
                answer = rag.query(query)

        st.markdown("### 📌 Answer:")
        st.write(answer)

        # ⭐ Optional: Show retrieved chunks (for demo/interview)
        if st.checkbox("🔍 Show retrieved chunks"):
            result = rag.chain.invoke({"query": query})
            for i, doc in enumerate(result["source_documents"], 1):
                st.write(f"**Chunk {i}:**")
                st.write(doc.page_content[:300])
                st.write("---")