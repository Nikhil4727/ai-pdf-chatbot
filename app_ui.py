import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

# 🔐 Load .env
load_dotenv()

# Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI PDF Chatbot")
st.title("AI PDF Chatbot")

# 🔥 Cache model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# 🔥 Cache PDF processing
@st.cache_resource
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # ⚠️ empty PDF check
    if not text.strip():
        return None, None

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    model = load_model()
    embeddings = model.encode(chunks)

    # FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks


# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Reset chat when new PDF uploaded
    st.session_state.messages = []

    index, chunks = process_pdf(uploaded_file)

    if index is None:
        st.error(" Could not extract text from PDF")
        st.stop()

    model = load_model()

    st.success("PDF processed successfully!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask something")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        # FAISS search
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), 3)

        context = ""
        for i in indices[0]:
            context += chunks[i] + "\n"

        prompt = f"""
        You are a helpful assistant.
        Answer clearly using only the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # Gemini call with error handling
        with st.spinner("Thinking... "):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                answer = response.text
            except Exception:
                answer = "Error generating response. Try again."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)