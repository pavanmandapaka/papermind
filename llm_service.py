# llm_service.py
import os
from dotenv import load_dotenv

# Try to import Groq client; if not available we will fall back later
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# Local fallback imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Tuple

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------
# 1) Groq client initializer
# --------------------------
def get_groq_client():
    if not GROQ_AVAILABLE:
        return None
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

# --------------------------
# 2) Local fallback LLM
# --------------------------
_local_gen = None
_local_tokenizer = None

def get_local_generator(model_name="google/flan-t5-small"):
    global _local_gen, _local_tokenizer
    if _local_gen is None:
        # Use CPU device by default; if CUDA available, pipeline will pick GPU automatically
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        _local_gen = pipeline("text2text-generation", model=model, tokenizer=_local_tokenizer, device=device)
    return _local_gen

# --------------------------
# 3) Build prompt helper with conversation history
# --------------------------
def build_prompt(context: str, question: str, conversation_history: List[dict] = None) -> str:
    """
    Build a clear prompt instructing the LLM to use only the provided context.
    Now includes conversation history for context-aware responses.
    Optimized for providing detailed explanations and solutions.
    """
    prompt = (
        "You are an expert educational assistant specialized in providing detailed explanations and solutions. "
        "Your task is to answer questions using ONLY the information from the provided document context. "
        
        "\n\nYour response style:\n"
        "1. **For Conceptual Questions**: Provide thorough explanations with definitions, key points, and examples\n"
        "2. **For Problem-Solving Questions**: Break down the solution into clear steps with explanations for each step\n"
        "3. **For 'How-to' Questions**: Provide step-by-step methodology with detailed reasoning\n"
        "4. **For 'Why' Questions**: Explain the underlying principles and reasoning in depth\n"
        "5. **For Definition Questions**: Give comprehensive explanations with context, use cases, and examples\n"
        
        "\n\nGuidelines for detailed answers:\n"
        "- Start with a clear, direct answer to the question\n"
        "- Break down complex topics into digestible sections\n"
        "- Explain the 'why' and 'how' behind concepts, not just the 'what'\n"
        "- Use numbered steps for procedures or solutions\n"
        "- Include relevant formulas, theories, or principles from the context\n"
        "- Provide examples or applications when mentioned in the context\n"
        "- If solving a problem, show the logical flow and reasoning\n"
        "- Aim for 5-8 sentences minimum for comprehensive coverage\n"
        "- Use clear structure: Introduction → Explanation → Examples/Steps → Conclusion\n"
        
        "\n\n**Important**: If the query is not related to the document content, say: "
        "'This question is not covered in the provided document. Please ask questions related to the document content.'\n\n"
    )
    
    # Add conversation history if available
    if conversation_history and len(conversation_history) > 0:
        prompt += "**Previous conversation context** (use this to maintain continuity):\n"
        for chat in conversation_history[-3:]:  # Last 3 exchanges for context
            prompt += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"
    
    prompt += f"**Document Content**:\n{context}\n\n"
    prompt += f"**Student's Question**: {question}\n\n"
    prompt += (
        "**Your Detailed Answer** (provide a comprehensive explanation/solution based on the document content above):\n"
    )
    
    return prompt

# --------------------------
# 4) Main generate function with conversation memory
# --------------------------
def generate_answer_from_context(question: str,
                                 retrieved_chunks: List[Tuple[str, float]],
                                 max_tokens: int = 768,  # Increased for very detailed explanations and solutions
                                 use_groq: bool = True,
                                 conversation_history: List[dict] = None) -> str:
    """
    retrieved_chunks: list of tuples (chunk_text, score) ordered by relevance.
    conversation_history: list of dicts with 'question' and 'answer' keys from previous turns.
    Returns: generated answer string.
    """

    # Combine top chunks into context, keep them short (avoid too long prompt)
    # We will join with double newlines to keep chunk separation
    context = "\n\n".join([c for c, _ in retrieved_chunks])

    prompt = build_prompt(context, question, conversation_history)

    # Try Groq if requested and available
    if use_groq and GROQ_AVAILABLE and GROQ_API_KEY:
        client = get_groq_client()
        if client is not None:
            # Use chat completions (the client may vary; adapt if API signature changes)
            # Updated to use llama-3.1-8b-instant (mixtral-8x7b-32768 is deprecated)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,  # Use the passed max_tokens parameter
                temperature=0.5  # Balanced for detailed yet accurate explanations
            )
            # Response parsing depends on Groq client return format
            try:
                return resp.choices[0].message.content
            except Exception:
                # Fallback: try a different path
                try:
                    return resp["choices"][0]["message"]["content"]
                except Exception:
                    return str(resp)

    # Local fallback: use FLAN-T5 (or other model)
    gen = get_local_generator()
    # generate with deterministic settings
    out = gen(prompt, max_length=max_tokens, do_sample=False)
    if isinstance(out, list) and len(out) > 0:
        return out[0].get("generated_text", str(out[0]))
    return str(out)
