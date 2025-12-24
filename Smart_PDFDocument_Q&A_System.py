# PyPDF2 is a simple way to extract text from PDF file. However the limitation is not able extract text from complex format like in the tables.
#-------------------------------#
# Step 1: Extract Text From PDF
#-------------------------------#
# Install in the terminal
# pip install PyPDF2 sentence-transformers transformers torch faiss-cpu

from PyPDF2 import PdfReader

# Opens the PDF file. And Loads all pages into memory
reader = PdfReader("sample.pdf")

text = ""

# Loops through each page. And Extract All Text Content Only
for page in reader.pages:
    text += page.extract_text()

#-------------------------------#
# Step 2: Clean The Extract Text
#-------------------------------#
import re

# Remove Extra White Space
def clean_text(text):
    text = re.sub(r'\s+', ' ', text) # Normalize Whitespace (eg: Spaces, newlines, tabs)
    text = text.strip()
    return text

# Apply Clean_text function
text = clean_text(text)

#-------------------------------#
# Step 3: Split Text Into Different Small Chunk
# Chunking is necessary because LLMs and embeddings have token limits and perform best with small, meaningful context, 
# making 200–500 word chunks faster and more accurate than large 10,000-word embeddings.
#-------------------------------#

# chunk_text function to split text into different chunk
def chunk_text(text, chunk_size=100, overlap=25):
    words = text.split()
    chunks = []

    # range(start, stop, step)
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

# Apply chunk_text function
chunks = chunk_text(text)

print(f"Total Chunks: {len(chunks)}")
print(chunks[0])

#-------------------------------#
# Step 4: Convert Text To Vector Enbeddings
#-------------------------------#
from sentence_transformers import SentenceTransformer

# Import Embed Model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text into vector enbeddings
embeddings = embed_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True).astype("float32")

#-------------------------------#
# Step 5: Create FAISS (Facebook AI Similarity Search) Index 
#-------------------------------#

import faiss

# What this means
# IndexFlatL2 → exact similarity search
# L2 distance (Euclidean)
# Best for small–medium datasets
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS index
index.add(embeddings)

print(f"Total vectors stored: {index.ntotal}")

#-------------------------------#
# Step 6: Convert Question To Vector Enbeddings
#-------------------------------#
import numpy as np

user_question = "What was the accuracy achieved by the Logistic Regression model in the FYP?"

query_embeddings = embed_model.encode(
    user_question,
    convert_to_numpy=True
).astype("float32")

# If query_embeddings is a 1D vector (e.g., 768 dimensions)
query_embeddings = np.array(query_embeddings).reshape(1, -1)

#-------------------------------#
# Step 7: Use FAISS to find relevant chunks
#-------------------------------#

# Find the relevant chunk indices
k = 3 # number of top relevant chunks to retrive
D, I = index.search(query_embeddings, k)

# Retrieve the actual text chunks
relevant_chunks = [chunks[i] for i in I[0]]

# Combine the top chunks
context = "\n\n".join(relevant_chunks)

# Print context
print("Answer:", context)

#-------------------------------#
# Step 8: Answer The Question/Query
#-------------------------------#
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import LLM Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Write a prompt
prompt = f"Answer the question based on the text:\n{context}\nQuestion: {user_question}"

# Tokenize the prompt
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,   # truncate if too long
    max_length=512
)

# Generate the Answer
outputs = llm_model.generate(
    **inputs,
    max_length=150,
    do_sample=True,        # sample instead of greedy decoding
    top_p=0.9,             # nucleus sampling
    temperature=0.7,       # randomness in output
    num_return_sequences=1
)

# Decode the outputs given by the model
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print answer
print("Answer:", answer)

















