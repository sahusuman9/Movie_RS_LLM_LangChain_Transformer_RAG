# 🎬 Movie Recommender System with LangChain + HuggingFace + RAG

## 📌 Overview  
This project implements a **movie recommender system** that combines:  
- **FAISS vector search** for efficient similarity-based retrieval  
- **HuggingFace Transformers** for embeddings & language models  
- **LangChain** to integrate embeddings, retrieval, and LLM-based reasoning  

It allows users to ask natural language queries (e.g., *"Psychological thrillers with twist endings"*) and get personalized movie recommendations with explanations.

---

## 🚀 Features
- **Data Preprocessing**: Cleans and enriches raw movie data with a structured `Metatext` field.  
- **E5 Embedding Wrapper**: Custom class to integrate the multilingual `e5-small` model with LangChain.  
- **FAISS Vector Store**: Stores embeddings for efficient semantic similarity search.  
- **RetrievalQA with T5**: Uses `google/flan-t5-base` for natural language reasoning over retrieved movies.  
- **Custom Prompting**: A carefully crafted prompt template to generate human-like explanations.  

---

## 📂 Dataset
The project uses a **movies.csv** file containing:
- Title, Director, Genre, Overview (Plot), Stars, Year, IMDB Rating  

Example entry:
Title: Inception
Director: Christopher Nolan
Genre: Action, Adventure, Sci-Fi
Plot: A thief who steals corporate secrets through dream-sharing technology...
Stars: Leonardo DiCaprio, Joseph Gordon-Levitt
Year: 2010
Rating: 8.8

---

## ⚙️ Installation
Clone this repo and install dependencies:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```
Ensure you have PyTorch, HuggingFace Transformers, and LangChain installed:
```bash
pip install torch torchvision torchaudio
pip install transformers langchain faiss-cpu
```

🏗️ Project Workflow

1. Preprocessing
  - Combine movie details into Metatext for richer embeddings.
  - Add unique movie_id for indexing.

2. Embeddings
  - Use E5 multilingual small model for document embeddings.
  - Batched embedding for efficiency.

3. Vector Store
  - Store embeddings in FAISS for similarity search.
  - Retrieval + QA

4. RetrievalQA chain combines FAISS retriever + Flan-T5 model.
  - Natural language queries return top-k recommended movies with explanations.

# Query the system
recommender = AdvancedRecommender(vector_store, qa_chain)
recommendations = recommender.recommend(
    "Psychological thrillers with twist endings",
    top_n=3
)

🖥️ Example Usage
# Display results
```bash
for i, rec in enumerate(recommendations):
    print(f"\n{i+1}. {rec['title']} ({rec['year']}) - IMDB {rec['rating']}")
    print(f"Explanation: {rec['explanation']}")
```

Example Output:
1. Shutter Island (2010) - IMDB 8.2
Explanation: A psychological thriller directed by Martin Scorsese...

📊 Tech Stack
- Python
- Pandas / NumPy (data processing)
- FAISS (vector search)
- LangChain (retrieval pipeline)
- HuggingFace Transformers (LLMs & embeddings)
- PyTorch (backend framework)

---
