import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
import os
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")

st.markdown("""
    <style>
        .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
        .stChatInputContainer { position: fixed; bottom: 0; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI PDF Chatbot")

# ── Groq client ───────────────────────────────────────────────────────────────
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
        return None, None  # FIX 1: sirf 2 values (pehle 3 thi)

    # Chunking — natural boundaries pe split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # FIX 2: Indentation sahi ki — chroma_client ab sahi jagah hai
    chroma_client = chromadb.Client()  # in-memory
    collection = chroma_client.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection, chunks  # sirf 2 values

# ── Sidebar: PDF Upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.pdf_processed = False

        file_bytes = uploaded_file.read()
        collection, chunks = process_pdf(file_bytes)

        if collection is None:  # FIX 3: 'vectors' ki jagah 'collection' check
            st.error("❌ Could not extract text from PDF")
        else:
            st.session_state.collection = collection
            st.session_state.chunks = chunks
            st.session_state.pdf_processed = True
            st.success(f"✅ **{uploaded_file.name}**\nReady to chat!")

    if st.session_state.pdf_processed:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("Upload a PDF from the sidebar to start chatting!")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask something about your PDF...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # ChromaDB se relevant chunks fetch karo
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=3
        )
        context = "\n".join(results['documents'][0])  # FIX 4: context variable banaya

        conversation = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions about a PDF document.
Answer ONLY based on the context provided. If the answer is not in the context, say so.

Context from PDF:
{context}"""
            }
        ]

        for msg in st.session_state.messages[-6:]:
            conversation.append({"role": msg["role"], "content": msg["content"]})

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