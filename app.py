from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# Gemini setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
 
# 1. Read PDF
reader = PdfReader("data.pdf")
text = ""
 
for page in reader.pages:
    text += page.extract_text()
 
# 2. Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)
 
# 3. Embeddings (local, no API needed)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
 
# 4. FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
 
# USER QUESTION
query = input("Ask something: ")
 
query_embedding = model.encode([query])
 
# Search
k = 3
distances, indices = index.search(np.array(query_embedding), k)
 
# Combine relevant chunks
context = ""
for i in indices[0]:
    context += chunks[i] + "\n"
 
# FINAL PROMPT
prompt = f"""
Answer the question based only on the context below:
 
Context:
{context}
 
Question:
{query}
"""
 
# Gemini call
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
 
print("\n🤖 Answer:\n")
print(response.text)