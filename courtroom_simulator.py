import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import jsonlines
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set your OpenAI API key

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = "gpt-4.1-nano-2025-04-14"  # Use the latest model available

# Initialize embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sample legal documents (you'd replace these with actual case summaries/statutes)


# Load legal documents from sec437_chunks.jsonl
DOCUMENTS = []
with jsonlines.open("sec437_chunks.jsonl") as reader:
    for obj in reader:
        DOCUMENTS.append({"title": obj["id"], "content": obj["text"]})

# Build FAISS index for retrieval
# Filepath for the FAISS index
index_file_path = "faiss_index.bin"

if os.path.exists(index_file_path):
    # Load the FAISS index from the file
    index = faiss.read_index(index_file_path)
else:
    # Build the FAISS index
    doc_embeddings = embedder.encode([doc["content"] for doc in DOCUMENTS])
    index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
    index.add(np.array(doc_embeddings))
    
    # Save the FAISS index to a file
    faiss.write_index(index, index_file_path)

# Utility: retrieve top-k documents
def retrieve_documents(query, k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [DOCUMENTS[i]["content"] for i in I[0]]

# Utility: basic citation score
def score_citations(response, references):
    found = re.findall(r'(Section \d+|[\w\s]+ v\. [\w\s]+)', response)
    matched = sum(1 for c in found if any(c in ref for ref in references))
    return round((matched / max(len(found), 1)) * 100, 2)

# UI
st.title("🧑‍⚖️ AI Courtroom Simulator")

case_summary = st.text_area("🔍 Case Summary", "Client seeks bail for a non-bailable economic offence.")
start_sim = st.button("🎬 Run Simulation")

if start_sim:
    with st.spinner("Retrieving legal context..."):
        context_query = case_summary
        retrieved = retrieve_documents(context_query)

    st.subheader("📚 Retrieved References")
    for ref in retrieved:
        st.markdown(f"> {ref}")

    # Lawyer Turn
    st.subheader("👨‍💼 Lawyer's Argument")
    lawyer_prompt = f"""
    You are a lawyer representing the petitioner in an Indian bail hearing.
    Case Summary: {case_summary}
    Use the following references to frame your legal argument:
    {'\n'.join(retrieved)}
    Respond in under 300 words.
    """
    lawyer_resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a persuasive Indian lawyer."},
        {"role": "user", "content": lawyer_prompt}
    ],
    temperature=0.4
)
    lawyer_text = lawyer_resp.choices[0].message.content
    st.write(lawyer_text)

    # Judge Turn
    st.subheader("👨‍⚖️ Judge's Response")
    judge_prompt = f"""
    You are an Indian High Court judge evaluating the following argument:
    {lawyer_text}
    You may raise concerns, questions, or issue an interim observation. Use the legal references:
    {'\n'.join(retrieved)}
    """
    judge_resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a critical Indian judge."},
        {"role": "user", "content": judge_prompt}
    ],
    temperature=0.3
)
    judge_text = judge_resp.choices[0].message.content
    st.write(judge_text)

    # Score
    st.subheader("📊 Persuasion Index™ Score")
    score = score_citations(lawyer_text, retrieved)
    st.metric(label="Citation Accuracy", value=f"{score}%")

    st.success("Session complete. You can refine the lawyer's argument or run another case.")

