# import streamlit as st
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from google import genai
# import os
# from dotenv import load_dotenv
 
# # Load env
# load_dotenv()
 
# # Gemini client
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
 
# st.set_page_config(page_title="AI PDF Chatbot")
# st.title("📄 AI PDF Chatbot")
 
# # Load embeddings locally (no API, no version conflict)
# @st.cache_resource
# def load_embeddings():
#     return SentenceTransformer('all-MiniLM-L6-v2')
 
# # Process PDF
# @st.cache_resource
# def process_pdf(uploaded_file):
#     reader = PdfReader(uploaded_file)
#     text = ""
 
#     for page in reader.pages:
#         content = page.extract_text()
#         if content:
#             text += content
 
#     if not text.strip():
#         return None, None
 
#     # Split
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = splitter.split_text(text)
 
#     embeddings_model = load_embeddings()
 
#     # Create embeddings locally using SentenceTransformer
#     vectors = embeddings_model.encode(chunks)
#     vectors = np.array(vectors).astype("float32")
 
#     # FAISS index
#     index = faiss.IndexFlatL2(vectors.shape[1])
#     index.add(vectors)
 
#     return index, chunks
 
# # Upload PDF
# uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
 
# if uploaded_file is not None:
 
#     # Reset chat
#     st.session_state.messages = []
 
#     index, chunks = process_pdf(uploaded_file)
 
#     if index is None:
#         st.error("❌ Could not extract text from PDF")
#         st.stop()
 
#     embeddings_model = load_embeddings()
 
#     st.success("✅ PDF processed successfully!")
 
#     # Chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
 
#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])
 
#     query = st.chat_input("Ask something")
 
#     if query:
#         st.session_state.messages.append({"role": "user", "content": query})
#         st.chat_message("user").write(query)
 
#         # Embed query locally
#         query_vector = embeddings_model.encode([query])
#         query_vector = np.array(query_vector).astype("float32")
 
#         distances, indices = index.search(query_vector, 3)
 
#         context = ""
#         for i in indices[0]:
#             context += chunks[i] + "\n"
 
#         prompt = f"""
#         You are a helpful assistant.
#         Answer ONLY from the context below.
 
#         Context:
#         {context}
 
#         Question:
#         {query}
#         """
#         with st.spinner("Thinking... 🤔"):
#             try:
#                 response = client.models.generate_content(
#                     model="gemini-2.0-flash",
#                     contents=prompt
#                 )
#                 answer = response.text
#             except Exception as e:
#                 answer = f"❌ Error generating response: {str(e)}"
 
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.chat_message("assistant").write(answer)



# import streamlit as st
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from groq import Groq
# import os
# from dotenv import load_dotenv

# # Load env
# load_dotenv()

# # Groq client
# client = Groq(api_key="")


# st.set_page_config(page_title="AI PDF Chatbot")
# st.title("📄 AI PDF Chatbot")

# # Process PDF
# @st.cache_resource
# def process_pdf(uploaded_file):
#     reader = PdfReader(uploaded_file)
#     text = ""

#     for page in reader.pages:
#         content = page.extract_text()
#         if content:
#             text += content

#     if not text.strip():
#         return None, None, None

#     # Split into chunks
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = splitter.split_text(text)

#     # TF-IDF vectorizer (lightweight, no model download)
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(chunks)

#     return vectors, chunks, vectorizer

# # Upload PDF
# uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# if uploaded_file is not None:

#     # Reset chat
#     st.session_state.messages = []

#     vectors, chunks, vectorizer = process_pdf(uploaded_file)

#     if vectors is None:
#         st.error("❌ Could not extract text from PDF")
#         st.stop()

#     st.success("✅ PDF processed successfully!")

#     # Chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])

#     query = st.chat_input("Ask something")

#     if query:
#         st.session_state.messages.append({"role": "user", "content": query})
#         st.chat_message("user").write(query)

#         # Find relevant chunks using TF-IDF similarity
#         query_vector = vectorizer.transform([query])
#         similarities = cosine_similarity(query_vector, vectors).flatten()
#         top_indices = similarities.argsort()[-3:][::-1]  # top 3 chunks

#         context = ""
#         for i in top_indices:
#             context += chunks[i] + "\n"

#         prompt = f"""
# You are a helpful assistant.
# Answer ONLY from the context below.

# Context:
# {context}

# Question:
# {query}
# """

#         with st.spinner("Thinking... 🤔"):
#             try:
#                 response = client.chat.completions.create(
#                     model="llama-3.3-70b-versatile",
#                     messages=[{"role": "user", "content": prompt}]
#                 )
#                 answer = response.choices[0].message.content
#             except Exception as e:
#                 answer = f"❌ Error generating response: {str(e)}"

#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.chat_message("assistant").write(answer)



import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv


load_dotenv()
# Replace with this:
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")

st.markdown("""
    <style>
        .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
        .stChatInputContainer { position: fixed; bottom: 0; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI PDF Chatbot")

# ── Groq client ───────────────────────────────────────────────────────────────

import os
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ── Process PDF ───────────────────────────────────────────────────────────────
@st.cache_resource
def process_pdf(file_bytes):
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text.strip():
        return None, None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)

    return vectors, chunks, vectorizer

# ── Sidebar: PDF Upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        # Only reprocess if a new file is uploaded
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # reset chat only on new file
            st.session_state.pdf_processed = False

        file_bytes = uploaded_file.read()
        vectors, chunks, vectorizer = process_pdf(file_bytes)

        if vectors is None:
            st.error("❌ Could not extract text from PDF")
        else:
            st.session_state.vectors = vectors
            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.pdf_processed = True
            st.success(f"✅ **{uploaded_file.name}**\nReady to chat!")

    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to start chatting!")
else:
    # Display all chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    query = st.chat_input("Ask something about your PDF...")

    if query:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Find relevant chunks
        query_vector = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, st.session_state.vectors).flatten()
        top_indices = similarities.argsort()[-3:][::-1]

        context = ""
        for i in top_indices:
            context += st.session_state.chunks[i] + "\n"

        # Build conversation history for multi-turn chat
        conversation = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions about a PDF document.
Answer ONLY based on the context provided. If the answer is not in the context, say so.

Context from PDF:
{context}"""
            }
        ]

        # Add last 6 messages for memory
        for msg in st.session_state.messages[-6:]:
            conversation.append({"role": msg["role"], "content": msg["content"]})

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=conversation,
                        temperature=0.3,
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"❌ Error generating response: {str(e)}"

                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})