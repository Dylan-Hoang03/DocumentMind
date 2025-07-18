from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import URL
from werkzeug.utils import secure_filename
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext

import faiss
import os
import io
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# === Setup ===
app = Flask(__name__)
app.config["SQLALCHEMY_ECHO"] = True
CORS(app)

# === SQL Server Configuration ===
connection_string = URL.create(
    drivername="mssql+pyodbc",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_SERVER,
    port=int(DB_PORT) if DB_PORT else None,
    database=DB_NAME,
    query={"driver": "ODBC Driver 17 for SQL Server"}
)

app.config["SQLALCHEMY_DATABASE_URI"] = connection_string
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# === Models ===
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    content = db.Column(db.LargeBinary, nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
import re

import re

import re

def format_llm_response(text: str) -> list[str]:
    lines = text.strip().split("\n")
    results = []
    i = 0

    def split_into_sentences(paragraph: str) -> list[str]:
        # Use punctuation-based sentence splitting with a basic regex
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph.strip())

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Handle broken numbered lines like: "1" + "Do this"
        if re.match(r"^\d+\s*$", line) and i + 1 < len(lines):
            number = line.strip()
            next_line = lines[i + 1].strip()
            results.append(f"{number}. {next_line}")
            i += 2

        # Already numbered: "1. Do this"
        elif re.match(r"^\d+\.\s+", line):
            results.append(line)
            i += 1

        # Long unstructured block — split into separate sentences
        else:
            sentences = split_into_sentences(line)
            results.extend(sentences)
            i += 1

    return results







# === Routes ===
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    original_name = file.filename
    filename = secure_filename(original_name)

    # Check for duplicate filename
    existing = Document.query.filter_by(filename=filename).first()
    if existing:
        return jsonify({"error": "File already exists"}), 409

    file_content = file.read()
    doc = Document(filename=filename, original_name=original_name, content=file_content)
    db.session.add(doc)
    db.session.commit()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )

        # Save the index
        base_filename = os.path.splitext(filename)[0]
        faiss_folder = os.path.join("faiss_indexes", base_filename)
        os.makedirs(faiss_folder, exist_ok=True)

        # Save FAISS index separately
        faiss.write_index(faiss_index, os.path.join(faiss_folder, "faiss.index"))

        # Save metadata
        storage_context.persist(persist_dir=faiss_folder)

    finally:
        os.remove(tmp_path)

    return jsonify({"message": "Uploaded and indexed successfully"}), 200

@app.route("/list-pdfs", methods=["GET"])
def list_pdfs():
    docs = Document.query.order_by(Document.id).all()
    return jsonify({"pdfs": [doc.filename for doc in docs]})

@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.get_json()
    filename = data.get("filename")
    question = data.get("question")

    if not filename or not question:
        return jsonify({"error": "Missing filename or question"}), 400

    system_prompt = """You are a helpful assistant answering questions based solely on a specific document provided to you.

Only use the document contents to answer the question. Do not rely on prior knowledge.

If the question is not relevant to the document or if the document does not contain enough information to answer, respond with:
"I'm sorry, I couldn't find relevant information in the document to answer that question."

Provide clear, concise, and factual answers."""

    llm = Ollama(
        temperature=0.1,
        model="mistral",
        base_url="http://localhost:11434",
        request_timeout=600
    )

    base_filename = os.path.splitext(filename)[0]
    faiss_folder = os.path.join("faiss_indexes", base_filename)
    faiss_index_path = os.path.join(faiss_folder, "faiss.index")

    print("📂 Checking FAISS path:", faiss_index_path)
    if not os.path.exists(faiss_index_path):
        return jsonify({"error": "Vector store not found"}), 404

    try:
        faiss_index = faiss.read_index(faiss_index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            persist_dir=faiss_folder,
            vector_store=vector_store
        )

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        index = load_index_from_storage(storage_context, embed_model=embed_model, llm=None)
        query_engine = index.as_query_engine(
            llm=llm,
            system_prompt=system_prompt
        )

        response = query_engine.query(question)
        formatted = format_llm_response(response.response)
        return jsonify({"response": formatted})

    except Exception as e:
        print("❌ Error querying FAISS:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/pdf/<filename>")
def get_pdf(filename):
    doc = Document.query.filter_by(filename=filename).first()
    if not doc:
        return "File not found", 404

    return send_file(
        io.BytesIO(doc.content),
        download_name=doc.original_name,
        mimetype="application/pdf"
    )

# === Init ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)