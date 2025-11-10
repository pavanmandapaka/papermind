import streamlit as st 
from utils.text_processing import extract_text_from_pdf, split_text_into_chunks
from utils.embeddings import get_embedding_model, create_embeddings
from utils.vector_store import create_faiss_index, search_faiss_index
import numpy as np
import textwrap
from llm_service import generate_answer_from_context

st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("Chat with Your Notes")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.model = None
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and not st.session_state.processed:
    # Step 1: Extract text
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")

    # Clean up line breaks and redundant spaces
    clean_text = " ".join(text.split())

    # Step 2: Preview extracted text
    with st.expander("Preview of Extracted Text"):
        preview_text = clean_text[:800] + "..." if len(clean_text) > 800 else clean_text
        st.text(textwrap.fill(preview_text, width=100))

    # Step 3: Split text into chunks
    with st.spinner("Splitting text into chunks..."):
        st.session_state.chunks = split_text_into_chunks(clean_text)
    st.success(f"Split into {len(st.session_state.chunks)} chunks.")

    # Show example chunk
    with st.expander("Example Chunk"):
        clean_chunk = st.session_state.chunks[0].replace('\n', ' ').strip()
        formatted_chunk = textwrap.fill(clean_chunk, width=100)
        st.code(formatted_chunk[:500] + "..." if len(formatted_chunk) > 500 else formatted_chunk)

    # Step 4: Generate embeddings
    with st.spinner("Generating Embeddings..."):
        st.session_state.model = get_embedding_model()
        embeddings = create_embeddings(st.session_state.chunks, st.session_state.model)
        st.session_state.index = create_faiss_index(np.array(embeddings))
    
    st.success("Embeddings created and stored in FAISS index.")
    st.session_state.processed = True
    st.rerun()

# Only show query interface if document is processed
if st.session_state.processed:
    st.write("### 💬 Ask Questions About Your Notes")
    
    # Show conversation context indicator
    if len(st.session_state.chat_history) > 0:
        st.info(f"🧠 **Conversation Memory Active** - I remember the last {min(3, len(st.session_state.chat_history))} exchanges")
    
    # Add a reset button
    if st.button("🔄 Upload New Document"):
        st.session_state.processed = False
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.model = None
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("📚 View Source Chunks"):
                for i, (chunk, score) in enumerate(chat["sources"], 1):
                    st.markdown(f"**Source {i}** (Score: `{score:.4f}`)")
                    st.write(" ".join(chunk.split()))
                    st.divider()
    
    # User query input
    user_query = st.chat_input("Type your question here...")

    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Search and generate answer
        with st.chat_message("assistant"):
            with st.spinner(" Searching and generating answer..."):
                # Search - increased to top 5 chunks for more comprehensive context
                q_emb = st.session_state.model.encode([user_query])
                results = search_faiss_index(q_emb, st.session_state.index, st.session_state.chunks, top_k=5)
                
                # Generate answer WITH conversation history
                answer = generate_answer_from_context(
                    user_query, 
                    results, 
                    max_tokens=768,  # Increased for detailed explanations and solutions
                    use_groq=True,
                    conversation_history=st.session_state.chat_history  # Pass chat history for context
                )
                
                # Display answer
                st.write(answer)
                
                # Show sources in expander
                with st.expander(" View Source Chunks"):
                    for i, (chunk, score) in enumerate(results, 1):
                        st.markdown(f"**Source {i}** (Score: `{score:.4f}`)")
                        st.write(" ".join(chunk.split()))
                        st.divider()
        
        # Save to chat history
        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer,
            "sources": results
        })

else:
    st.info("👆 Please upload a PDF document to get started!")


# Sidebar
with st.sidebar:
    st.header("📊 Document Info")
    
    if st.session_state.processed:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Chunks", len(st.session_state.chunks))
        with col2:
            st.metric("💬 Conversations", len(st.session_state.chat_history))
        
        st.divider()
        
        # Download chat history
        if st.session_state.chat_history:
            chat_text = "\n\n".join([
                f"Q: {chat['question']}\nA: {chat['answer']}"
                for chat in st.session_state.chat_history
            ])
            st.download_button(
                label="💾 Download Chat History",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
    else:
        st.info("Upload a document to see stats")
    
    st.divider()