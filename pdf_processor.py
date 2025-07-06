from PyPDF2 import PdfReader
import json

pdf_path = "Section_437_CrPC_Case_Summaries.pdf"
reader = PdfReader(pdf_path)

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"



chunk_size = 500  # characters per chunk
chunks = []
source = "Section_437_CrPC_Case_Summaries.pdf"

for i in range(0, len(full_text), chunk_size):
    chunk = {
        "id": f"sec437_chunk_{i//chunk_size + 1}",
        "text": full_text[i:i+chunk_size].strip(),
        "metadata": {
            "source": source,
            "section": "CrPC 437",
            "page_estimate": (i // 2000) + 1
        }
    }
    chunks.append(chunk)
output_path = "sec437_chunks.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk) + "\n")

print("Saved to", output_path)
