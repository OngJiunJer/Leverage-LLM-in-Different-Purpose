# Leverage-LLM-in-Different-Purpose
-------
# Features
1. PDF Question Answering
   - Extracts text from PDF files
   - Cleans and chunks the text for better processing
   - Converts text chunks into embeddings using SentenceTransformers (all-MiniLM-L6-v2)
   - Stores embeddings in FAISS for similarity search (RAG: retrieve Augmented generator)
   - Uses an LLM (google/flan-t5-small) to answer user questions based on the PDF context

2. SQL Generation & Execution
  - Defines a database schema, rule, and question
  - Generates SQL queries using Google GenAI LLM (gemini-3-flash-preview)
  - Executes SQL queries on a CSV/SQLite database
  - Provides ready-to-use SQL results

---------
# Tech Stack
- Python 3.x
- PyPDF2 – Extract text from PDFs
- SentenceTransformers – Create embeddings
- FAISS – Fast similarity search
- Transformers – LLM for question answering
- Google GenAI – LLM for SQL generation
- SQLite + Pandas – Manage structured data
