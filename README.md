# 📚 ChatPDF – Chat with Your PDF Documents

**PDF Genius** is a powerful Streamlit-based web app that allows you to upload PDF files, process their content using advanced text embeddings, and interactively ask questions based on the content — powered by **Gemini 1.5 Flash** and **FAISS** vector search.

---

## 🚀 Features

- Upload multiple PDF documents
- Automatically extract and chunk text
- Generate embeddings using Google Generative AI
- Store vectors using FAISS for fast retrieval
- Chat with your documents using Gemini 1.5 Flash
- Clean, responsive UI with a focus on user experience

---

## 🛠️ Tech Stack

| Tool/Library              | Purpose                                  |
|--------------------------|------------------------------------------|
| `Streamlit`              | Web app frontend                         |
| `PyPDF2`                 | PDF text extraction                      |
| `LangChain`              | Chaining LLM and vector operations       |
| `Google Generative AI`   | Embeddings + Chat (Gemini)               |
| `FAISS`                  | Local vector database                    |
| `dotenv`                 | Manage API keys                          |

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pdf-genius.git
   cd pdf-genius
   ```
2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
4. **Set up environment variables**:

Create a .env file in the root directory:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## 🧪 Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## ⚠️ Notes
Only local FAISS vector storage is used for now
Make sure your Google API key has access to the Generative AI APIs
Embeddings are saved in a folder named faiss_index
