from flask import Flask, render_template, request
import os
import nltk
import re
import pdfplumber
import docx2txt
import torch
import lancedb
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

# Configurations
UPLOAD_FOLDER = 'uploaded_files'
EMBEDDINGS_FOLDER = 'saved_embeddings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Globals
all_text = ""
table = None
suggested_questions = []

# Load basic resources
nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')  # faster embeddings

# Setup Gemini
genai.configure(api_key="Your_GEMINI_API_KEY")  # Replace with your real key
gemini_model = genai.GenerativeModel('gemini-1.5-pro')


# Helpers
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_docx_file(file_path):
    return docx2txt.process(file_path)


def read_pdf_file(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def read_file(file_path):
    if file_path.endswith(".txt"):
        return read_text_file(file_path)
    elif file_path.endswith(".docx"):
        return read_docx_file(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type:", file_path)


def recursive_text_splitter(text, max_chunk_length=500, overlap=50):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_length:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks


def save_embeddings(chunks, embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump((chunks, embeddings), f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        chunks, embeddings = pickle.load(f)
    return chunks, embeddings


def prepare_data(chunks, embeddings):
    data = []
    for chunk, embed in zip(chunks, embeddings):
        temp = {"text": chunk, "vector": embed}
        data.append(temp)
    return data


def create_lancedb(chunks, embeddings):
    db = lancedb.connect("/tmp/lancedb")
    data = prepare_data(chunks, embeddings)
    table = db.create_table("scratch", data=data, mode="overwrite")
    return table


def ask_gemini(contexts, question):
    base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations like [1], [2].

User question: {}
Contexts:
{}
"""
    full_prompt = base_prompt.format(question, "\n".join(contexts))
    response = gemini_model.generate_content(full_prompt)

    try:
        return response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError):
        return "Sorry, unable to generate an answer."


def generate_suggested_questions(text):
    prompt = f"""You are an AI Assistant. Based on the following text, suggest 5 interesting and relevant questions a user might ask.

Text:
{text}

Provide the questions as a bullet-point list."""

    response = gemini_model.generate_content(prompt)

    try:
        suggested = response.candidates[0].content.parts[0].text
        lines = suggested.strip().split('\n')
        questions = [line.lstrip("-•12345. ").strip() for line in lines if line.strip()]
        return questions[:5] if questions else ["No suggestions available."]
    except (IndexError, AttributeError):
        return ["No suggestions available."]


# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    global all_text, table, suggested_questions

    uploaded_files = request.files.getlist("files")
    question = request.form.get("question", "")

    if uploaded_files and uploaded_files[0].filename != '':
        file_paths = []
        combined_text = ""

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

        # Check if embedding already exists
        combined_name = "_".join([secure_filename(file.filename) for file in uploaded_files])[:100]
        embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{combined_name}.pkl")

        if os.path.exists(embedding_path):
            chunks, embeds = load_embeddings(embedding_path)
        else:
            for path in file_paths:
                combined_text += read_file(path) + "\n"

            chunks = recursive_text_splitter(combined_text, max_chunk_length=500, overlap=50)
            embeds = model.encode(chunks, convert_to_numpy=True)
            save_embeddings(chunks, embeds, embedding_path)

        table = create_lancedb(chunks, embeds)
        suggested_questions = generate_suggested_questions(combined_text)

    answer = None
    if question and table:
        query_embedding = model.encode([question])[0]
        results = table.search(query_embedding).limit(5).to_list()
        context = [r["text"] for r in results]
        answer = ask_gemini(context, question)

    return render_template("index.html", answer=answer, suggested_questions=suggested_questions)


# Run
if __name__ == "__main__":
    app.run(debug=True)
