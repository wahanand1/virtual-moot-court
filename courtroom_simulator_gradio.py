import gradio as gr
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import jsonlines
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = "gpt-4.1-nano-2025-04-14"

# Initialize embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load legal documents
DOCUMENTS = []
with jsonlines.open("sec437_chunks.jsonl") as reader:
    for obj in reader:
        DOCUMENTS.append({"title": obj["id"], "content": obj["text"]})

# FAISS index setup
index_file_path = "faiss_index.bin"
if os.path.exists(index_file_path):
    index = faiss.read_index(index_file_path)
else:
    doc_embeddings = embedder.encode([doc["content"] for doc in DOCUMENTS])
    index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
    index.add(np.array(doc_embeddings))
    faiss.write_index(index, index_file_path)

def retrieve_documents(query, k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [DOCUMENTS[i]["content"] for i in I[0]]

def score_citations(response, references):
    found = re.findall(r'(Section \d+|[\w\s]+ v\. [\w\s]+)', response)
    matched = sum(1 for c in found if any(c in ref for ref in references))
    return round((matched / max(len(found), 1)) * 100, 2)

def simulate_court(case_summary):
    # Retrieve legal context
    retrieved = retrieve_documents(case_summary)
    references = "\n\n".join([f"Reference:\n{ref}" for ref in retrieved])
    
    # Lawyer's argument
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
    
    # Judge's response
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
    
    # Calculate score
    score = score_citations(lawyer_text, retrieved)
    
    return references, lawyer_text, judge_text, f"Citation Accuracy: {score}%"

# Gradio Interface
with gr.Blocks(title="AI Courtroom Simulator") as demo:
    gr.Markdown("# 🧑‍⚖️ AI Courtroom Simulator")
    
    with gr.Row():
        case_input = gr.Textbox(
            label="Case Summary",
            placeholder="Client seeks bail for a non-bailable economic offence.",
            lines=3
        )
    
    submit_btn = gr.Button("Run Simulation", variant="primary")
    
    with gr.Row():
        with gr.Column():
            references_output = gr.Textbox(label="📚 Retrieved References", lines=5)
            lawyer_output = gr.Textbox(label="👨‍💼 Lawyer's Argument", lines=8)
            judge_output = gr.Textbox(label="👨‍⚖️ Judge's Response", lines=8)
            score_output = gr.Textbox(label="📊 Persuasion Index™ Score")
    
    submit_btn.click(
        simulate_court,
        inputs=[case_input],
        outputs=[references_output, lawyer_output, judge_output, score_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()