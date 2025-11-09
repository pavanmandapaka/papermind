import streamlit as st 
from utils.text_processing import extract_text_from_pdf, split_text_into_chunks
from utils.embeddings import get_embedding_model, create_embeddings
from utils.vector_store import create_faiss_index, search_faiss_index
import numpy as np
import textwrap


st.set_page_config(page_title="Chat with Your Notes", page_icon="", layout="wide")
st.title("Chat with Your Notes")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Step 1: Extract text
    with st.spinner(" Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")

    # Clean up line breaks and redundant spaces
    clean_text = " ".join(text.split())

    # Step 2: Preview extracted text
    st.write("###  Preview of Extracted Text")
    preview_text = clean_text[:800] + "..." if len(clean_text) > 800 else clean_text
    st.text(textwrap.fill(preview_text, width=100))

    # Step 3: Split text into chunks
    with st.spinner(" Splitting text into chunks..."):
        chunks = split_text_into_chunks(clean_text)
    st.success(f" Split into {len(chunks)} chunks.")

    # Clean & format example chunk for display
    clean_chunk = chunks[0].replace('\n', ' ').strip()
    formatted_chunk = textwrap.fill(clean_chunk, width=100)
    st.write("### Example Chunk:")
    st.code(formatted_chunk[:500] + "..." if len(formatted_chunk) > 500 else formatted_chunk)

    # Step 4: Generate embeddings
    st.write("###  Generating Embeddings...")
    with st.spinner("Encoding chunks into embeddings..."):
        model = get_embedding_model()
        embeddings = create_embeddings(chunks, model)
        index = create_faiss_index(np.array(embeddings))
    st.success("Embeddings created and stored in FAISS index.")

    # Step 5: User query
    st.write("### Ask something from your notes")
    user_query = st.text_input("Type your question here:")

    if user_query:
        with st.spinner("Searching for the most relevant sections..."):
            q_emb = model.encode([user_query])
            results = search_faiss_index(q_emb, index, chunks, top_k=2)

        st.write("#### Top Relevant Sections:")
        for text, dist in results:
            # Clean and format result text for neat display
            clean_result = " ".join(text.split())
            formatted_result = textwrap.fill(clean_result, width=100)
            st.markdown(f"**Similarity Score:** `{dist:.4f}`")
            st.write(formatted_result)
            st.write("---")
