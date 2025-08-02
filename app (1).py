import os
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")

from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
import json

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("API_KEY") 

if not API_KEY:
    st.error("API_KEY environment variable not set. Please check your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- SESSION STATE INIT ---
for key in ['download_history', 'chat_history', 'pdf_processed', 'chunks', 'faiss_index', 'pdf_name', 'pdf_summary']:
    if key not in st.session_state:
        st.session_state[key] = [] if 'history' in key else None if 'index' in key else False if key == 'pdf_processed' else ""

# --- EMBEDDING MODEL ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
genai_model = genai.GenerativeModel('gemini-2.5-flash')

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = bytes(pdf_file.read())
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text() + "\n\n"
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

@st.cache_data(show_spinner=False)
def create_faiss_index(chunks):
    try:
        embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        st.error(f"Failed to create embeddings or FAISS index: {e}")
        return None

def search_faiss_index(index, query, chunks, k=5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

def get_gemini_response(context, question):
    instruction = """You are a helpful and meticulous academic assistant named StudyMate.
Your purpose is to answer questions strictly based on the provided text context from a document.
- You must not use any external knowledge or information outside of the provided text.
- If the provided text contains the answer, extract it and present it clearly and concisely.
- If the answer cannot be found in the text, you must explicitly say so."""

    prompt = f"""{instruction}

CONTEXT:
---
{context}
---

QUESTION:
{question}

Please answer the question based only on the context provided above.
"""
    try:
        response = genai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.95,
            )
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def summarize_pdf(text):
    summary_prompt = f"""You are an academic assistant. Summarize the following document in 5-7 bullet points that highlight the most important information.

DOCUMENT:
{text[:6000]}

Please provide a concise summary in bullet points."""
    
    try:
        response = genai_model.generate_content(
            summary_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.95,
            )
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Summary could not be generated due to an error."

def add_to_download_history(question, answer):
    st.session_state.download_history.append({
        "question": question,
        "answer": answer
    })

def get_download_history_txt():
    lines = []
    lines.append("📄 PDF Summary:\n")
    lines.append(st.session_state.pdf_summary or "_No summary available._")
    lines.append("\n\n💬 Chat History:\n")
    for i, entry in enumerate(st.session_state.download_history, 1):
        lines.append(f"Q{i}: {entry['question']}\nA{i}: {entry['answer']}\n")
    return "\n".join(lines)

def get_download_history_json():
    data = {
        "pdf_summary": st.session_state.pdf_summary or "",
        "chat_history": st.session_state.download_history
    }
    return json.dumps(data, indent=2)

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    h1 { color: #f8fafc; font-weight: bold; text-align: center; }
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📚 StudyMate")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_name = uploaded_file.name
            raw_text = extract_text_from_pdf(uploaded_file)
            if raw_text:
                chunks = split_text_into_chunks(raw_text)
                index = create_faiss_index(chunks)
                if index:
                    st.session_state.chunks = chunks
                    st.session_state.faiss_index = index
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = summarize_pdf(raw_text)
                    st.success(f"Processed '{st.session_state.pdf_name}' successfully!")
                else:
                    st.error("Failed to create FAISS index.")

    if st.session_state.pdf_processed:
        st.markdown("---")
        st.info(f"**Current PDF:** `{st.session_state.pdf_name}`")
        if st.button("Upload New PDF"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

# --- MAIN INTERFACE ---
if not st.session_state.pdf_processed:
    st.info("Please upload a PDF in the sidebar to begin.")
else:
    # Summary
    st.markdown("### 📝 PDF Summary")
    st.markdown(st.session_state.get("pdf_summary", "_No summary available._"))

    st.markdown("---")  # Divider before chat

    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("Generating answer..."):
            relevant_chunks = search_faiss_index(
                st.session_state.faiss_index, prompt, st.session_state.chunks
            )
            context = "\n\n".join(relevant_chunks)
            answer = get_gemini_response(context, prompt)

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            add_to_download_history(prompt, answer)

    # Download buttons
    st.markdown("---")
    st.markdown("### 📥 Download Chat History and Summary")

    st.download_button(
        label="Download as .txt",
        data=get_download_history_txt(),
        file_name="chat_history_with_summary.txt",
        mime="text/plain"
    )

    st.download_button(
        label="Download as .json",
        data=get_download_history_json(),
        file_name="chat_history_with_summary.json",
        mime="application/json"
    )
