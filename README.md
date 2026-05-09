AI PDF Chatbot

A RAG-based chatbot where you can upload any PDF and ask questions about it. Built this as a learning project to understand how RAG (Retrieval-Augmented Generation) actually works under the hood.

 What is this?
So basically the idea is simple — you upload a PDF, and then you can chat with it. Like ask "what is this document about?" or "explain chapter 3" and it will answer based on what's actually in the PDF. No hallucinations (hopefully lol).

How it works (the fun part)
This project uses something called **RAG** — Retrieval Augmented Generation. Sounds fancy but it's actually pretty straightforward once you get it.

Step 1 — Read the PDF
Using `pypdf` to extract all the text from whatever PDF you upload.

Step 2 — Chunking
The text is too big to send to an LLM at once, so we break it into smaller pieces (500 characters each with 50 character overlap). Used `RecursiveCharacterTextSplitter` for this — it splits on natural boundaries like paragraphs and sentences instead of just cutting randomly.

Step 3 — Store in ChromaDB
Each chunk gets stored in ChromaDB (a vector database). ChromaDB automatically converts text into embeddings (numbers that represent meaning) and builds an HNSW graph internally for fast search.

Step 4 — Search
When you ask a question, ChromaDB finds the 3 most relevant chunks using semantic search (cosine similarity). So even if you use different words than the PDF, it still finds the right content.

Step 5 — Generate Answer
The relevant chunks + your question get sent to LLaMA 3.3 70B (via Groq API) which then generates the final answer.

```
Your Question
     ↓
ChromaDB finds relevant chunks
     ↓
LLaMA reads chunks + question
     ↓
Answer 
```

---

Tech Stack

| What | Why |
|------|-----|
| Streamlit | For the UI (easiest way to build Python web apps) |
| pypdf | Reading PDF files |
| LangChain | Just using their text splitter utility |
| ChromaDB | Vector database (stores and searches embeddings) |
| Groq + LLaMA 3.3 70B | The actual LLM that generates answers |

Project Structure

```
AI_Project/
├── app_ui.py         # main application file
├── requirements.txt  # all dependencies
├── .env              # API keys (not pushed to github!)
├── .gitignore
└── README.md
```
How to run this locally

1. Clone the repo
```bash
git clone <your-repo-url>
cd AI_Project
```

2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your API key

Create a `.env` file and add:
```
GROQ_API_KEY=your_key_here
```
Get your free API key from [console.groq.com](https://console.groq.com)

5. Run it
```bash
streamlit run app_ui.py
```

What I learned building this
- What RAG actually is and why it's useful (LLMs don't know your private data, RAG fixes that)
- How vector databases work — especially HNSW algorithm which builds a smart graph for fast nearest-neighbor search instead of brute-force comparing every chunk
- Difference between keyword search (TF-IDF) vs semantic search (embeddings) — semantic is way better because it understands meaning not just words
- How ChromaDB stores embeddings + original text + metadata all together in collections
- Streamlit session state (it reruns from scratch on every interaction so you need session state to remember things)
---

Known limitations
- Only works with PDFs that have actual text (scanned PDFs won't work)
- ChromaDB is in-memory so everything resets when you close the app (that's fine for this use case)
- Very large PDFs might be slow to process

