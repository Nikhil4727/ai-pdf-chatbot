import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

# 🔐 Load env
load_dotenv()

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI PDF Chatbot")
st.title("📄 AI PDF Chatbot")

# 🔥 Load embeddings (FAST + NO TORCH)
@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
        google_api_key=os.getenv("GEMINI_API_KEY") 
    )

# 🔥 Process PDF
@st.cache_resource
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text.strip():
        return None, None

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings_model = load_embeddings()

    # 🔥 Create embeddings (FAST API-based)
    vectors = embeddings_model.embed_documents(chunks)

    vectors = np.array(vectors).astype("float32")

    # FAISS
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    return index, chunks

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Reset chat
    st.session_state.messages = []

    index, chunks = process_pdf(uploaded_file)

    if index is None:
        st.error("❌ Could not extract text from PDF")
        st.stop()

    embeddings_model = load_embeddings()

    st.success("✅ PDF processed successfully!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask something")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        # 🔍 Embed query
        query_vector = embeddings_model.embed_query(query)
        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = index.search(query_vector, 3)

        context = ""
        for i in indices[0]:
            context += chunks[i] + "\n"

        prompt = f"""
        You are a helpful assistant.
        Answer ONLY from the context below.

        Context:
        {context}

        Question:
        {query}
        """

        with st.spinner("Thinking... 🤔"):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                answer = response.text
            except Exception:
                answer = "❌ Error generating response"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)