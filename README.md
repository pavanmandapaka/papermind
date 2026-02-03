# PaperMind

An intelligent PDF chat application that allows you to have conversations with your documents using AI. Upload any PDF and ask questions about its content - PaperMind will provide detailed, context-aware answers based on the document.

## Features

- **PDF Upload & Processing** - Extract and process text from any PDF document
- **Semantic Search** - Find relevant information using advanced embedding-based search
- **Conversational AI** - Ask questions and get detailed, educational answers
- **Conversation Memory** - Maintains context from previous questions (last 3 exchanges)
- **Source Citations** - View the exact document chunks used to generate each answer
- **Chat History Export** - Download your conversation history as a text file
- **Fast & Accurate** - Powered by GROQ's llama-3.1-8b-instant model

## Tech Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **LLM**: GROQ API (llama-3.1-8b-instant) with local FLAN-T5 fallback
- **Text Processing**: LangChain text splitters


## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd papermind
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Get your free GROQ API key from [console.groq.com](https://console.groq.com)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501`

## Usage

1. **Upload a PDF**: Click the file uploader and select your PDF document
2. **Wait for Processing**: The app will extract text, create embeddings, and build a searchable index
3. **Ask Questions**: Type your questions in the chat input
4. **Get Answers**: Receive detailed, context-aware answers with source citations
5. **Continue Conversation**: Ask follow-up questions - the app remembers previous exchanges
6. **Download History**: Export your chat history from the sidebar

## How It Works

1. **Document Processing**
   - Extracts text from uploaded PDF
   - Splits text into manageable chunks (1000 chars with 200 char overlap)
   - Generates embeddings for each chunk using Sentence Transformers

2. **Query Processing**
   - Converts user question into an embedding
   - Searches FAISS index for top 5 most relevant chunks
   - Passes relevant context to LLM along with conversation history

3. **Answer Generation**
   - Uses GROQ's llama-3.1-8b-instant for fast, detailed responses
   - Falls back to local FLAN-T5 model if GROQ is unavailable
   - Provides structured, educational answers with examples and step-by-step explanations

## Project Structure

```
papermind/
├── app.py                      # Main Streamlit application
├── llm_service.py             # LLM integration (GROQ/local)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
└── utils/
    ├── embeddings.py         # Embedding generation
    ├── text_processing.py    # PDF text extraction and chunking
    ├── vector_store.py       # FAISS index operations
    ├── qa.py                 # Question-answering logic
    └── pdf_export.py         # PDF export utilities
```

## Configuration

### LLM Settings

- **Model**: llama-3.1-8b-instant (via GROQ)
- **Max Tokens**: 768 (for detailed explanations)
- **Temperature**: 0.5 (balanced accuracy and creativity)
- **Fallback**: google/flan-t5-small (local)

### Embedding Model

- **Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **Use Case**: Semantic similarity search

### Text Chunking

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Method**: Recursive character splitting

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests


## Acknowledgments

- GROQ for providing fast LLM inference
- Sentence Transformers for embeddings
- FAISS for efficient similarity search
- Streamlit for the awesome web framework

## Contact

For questions or suggestions, please open an issue or reach out!

---

