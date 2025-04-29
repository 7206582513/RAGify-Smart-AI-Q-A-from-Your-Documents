# 🔍🧠 RAGify: Smart AI Q&A from Your Documents

**RAGify** is a modern, intelligent web application built with Flask that allows users to upload multiple documents (PDF, DOCX, TXT), automatically generates smart suggested questions using Gemini Pro, and answers any custom queries using a powerful Retrieval-Augmented Generation (RAG) pipeline.

> Ask your documents anything. Get accurate answers. Instantly.

---

## 🌟 Features

- 📂 Upload **multiple documents** (PDF, DOCX, TXT)
- 🔎 Automatically **splits and embeds** content using **Sentence Transformers**
- 💾 **Caches embeddings** for speed (no need to recompute)
- 🔥 Uses **LanceDB** for ultra-fast semantic search
- 🤖 Answers powered by **Google Gemini Pro** LLM
- 💡 Generates **suggested questions** from content using AI
- 🌙 Clean, modern **dark UI** with neon highlights
- 🚀 Optimized for **large files** and blazing-fast performance

---

## 🛠 Tech Stack

| Component        | Tech                         |
|------------------|------------------------------|
| Backend          | Flask (Python)               |
| Frontend         | Bootstrap 5 + Custom CSS     |
| NLP Embeddings   | Sentence Transformers (MiniLM) |
| Vector DB        | LanceDB                      |
| LLM (Q&A + Gen)  | Google Gemini (via `google-generativeai`) |
| File Parsing     | pdfplumber, docx2txt         |
| Chunking         | nltk sentence splitter       |

---

## 🚀 Installation & Setup

### 🔧 Prerequisites
- Python 3.8+
- Google Gemini API key ([get here](https://makersuite.google.com/app))

### 📦 Install Dependencies

pip install -r requirements.txt

📁 Project Structure

├── app.py
├── uploaded_files/         # Stores uploaded documents
├── saved_embeddings/       # Caches embeddings (optional)
├── templates/
│   └── index.html          # Web UI
├── static/                 # (Optional: images, CSS)
├── requirements.txt
└── README.md
🔑 Add Gemini API Key
In app.py, replace this line:

python
Copy
Edit
genai.configure(api_key="YOUR_GEMINI_API_KEY")
With your actual Gemini Pro API key.

▶️ Run the App
bash
Copy
Edit
python app.py
Open your browser at http://127.0.0.1:5000

💡 Example Use Cases
Upload a lease or contract → "What is the termination clause?"

Upload research papers → "What is the main contribution of this work?"

Upload a handbook → "What are the safety protocols for machinery?"

🖼️ Screenshot
<img width="730" alt="image" src="https://github.com/user-attachments/assets/001deed9-fd5a-4673-a691-9ab6267a8689" />
<img width="628" alt="image" src="https://github.com/user-attachments/assets/93ff8de5-d3a9-4eed-a882-ea20f4413b36" />

This project is licensed under the MIT License – feel free to fork and build on it.

🤝 Contributions
Pull requests, issues, and stars are welcome!
If you build on this, tag @Rohit 😄

🙌 Acknowledgements
Google Gemini API

Huggingface Transformers

LanceDB

Flask
