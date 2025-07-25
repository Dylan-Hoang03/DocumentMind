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
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding

import numpy as np

import faiss
import os
import io
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import re
import fitz  # pip install PyMuPDF
from io import BytesIO

# === Load environment variables ===
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# === Shared OneDrive Folder Path ===
ONEDRIVE_SHARED_FOLDER = r"C:\\Users\\ly.hoangminhdatdylan\\OneDrive - Spartronics\\NguyenNgoc, Lam's files - Dylan Project"

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

# === Index Local PDFs ===
def index_local_pdfs():
    print("🔍 Scanning shared folder for PDFs...")

    for filename in os.listdir(ONEDRIVE_SHARED_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(ONEDRIVE_SHARED_FOLDER, filename)
        base_filename = os.path.splitext(filename)[0]
        faiss_folder = os.path.join("faiss_indexes", base_filename)
        faiss_index_path = os.path.join(faiss_folder, "faiss.index")

        print(f"\n📄 Found PDF: {filename}")

        if os.path.exists(faiss_index_path):
            print(f"✅ FAISS index already exists for {filename}, skipping.")
            continue

        try:
            with open(full_path, "rb") as f:
                file_content = f.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            print(f"📥 Created temporary file for processing: {tmp_path}")

            documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            print(f"📦 Extracted {len(documents)} document chunks from {filename}")

            if not documents:
                print(f"⚠️ No text extracted from {filename}, skipping.")
                continue

            embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio"
            )

            faiss_index = faiss.IndexFlatL2(768)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embed_model
            )

            os.makedirs(faiss_folder, exist_ok=True)
            print(f"💾 Saving FAISS index to: {faiss_folder}")

            try:
                faiss.write_index(faiss_index, os.path.join(faiss_folder, "faiss.index"))
                storage_context.persist(persist_dir=faiss_folder)
                print(f"✅ Indexing completed for {filename}")
            except Exception as e:
                print(f"❌ Failed to write FAISS index for {filename}: {e}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🧹 Temporary file deleted: {tmp_path}")


# === Format LLM Output ===
def format_llm_response(text: str) -> list[str]:
    lines = text.strip().split("\n")
    results = []
    i = 0

    def split_into_sentences(paragraph: str) -> list[str]:
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph.strip())

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if re.match(r"^\d+\s*$", line) and i + 1 < len(lines):
            number = line.strip()
            next_line = lines[i + 1].strip()
            results.append(f"{number}. {next_line}")
            i += 2
        elif re.match(r"^\d+\.\s+", line):
            results.append(line)
            i += 1
        else:
            sentences = split_into_sentences(line)
            results.extend(sentences)
            i += 1

    return results

@app.route("/list-pdfs", methods=["GET"])
def list_pdfs():
    print("📥 /list-pdfs triggered — indexing any new PDFs...")
    pdfs = []

    for filename in os.listdir(ONEDRIVE_SHARED_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(ONEDRIVE_SHARED_FOLDER, filename)
        base_filename = os.path.splitext(filename)[0]
        faiss_folder = os.path.join("faiss_indexes", base_filename)
        faiss_index_path = os.path.join(faiss_folder, "faiss.index")

        pdfs.append(filename)

        # Check if FAISS index already exists
        if os.path.exists(faiss_index_path):
            continue

        print(f"🆕 Indexing {filename} on-demand...")

        try:
            with open(full_path, "rb") as f:
                file_content = f.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            print(f"📄 Extracted {len(documents)} chunks from {filename}")

            if not documents:
                print(f"⚠️ No content found in {filename}, skipping.")
                continue

            embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio"
            )

            faiss_index = faiss.IndexFlatL2(768)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embed_model
            )

            os.makedirs(faiss_folder, exist_ok=True)
            faiss.write_index(faiss_index, os.path.join(faiss_folder, "faiss.index"))
            storage_context.persist(persist_dir=faiss_folder)

            print(f"✅ FAISS index generated for {filename}")

        except Exception as e:
            print(f"❌ Error indexing {filename}: {e}")

        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🧹 Temp file deleted: {tmp_path}")

    return jsonify({"pdfs": pdfs})


# === Query PDFs ===
@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.get_json()
    filenames = data.get("filenames")
    question = data.get("question")
    question+= "answer only according to the documents /no_think"

    if not filenames or not question:
        return jsonify({"error": "Missing filenames or question"}), 400

    system_prompt = """You are a helpful assistant answering questions based solely on specific documents provided to you.

Only use the document contents to answer the question. Do not rely on prior knowledge.

If the question is not relevant to the documents or if they do not contain enough information to answer, respond with:
"I'm sorry, I couldn't find relevant information in the documents to answer that question."

Provide clear, concise, and factual answers. Do not infer any answers. If the answer is not in the document, say so explicitly. /no_think"""

    llm = Ollama(
        temperature=0.1,
        model="mistral",
        base_url="http://localhost:11434",
        request_timeout=600
    )

    all_nodes = []

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_base="http://localhost:1234/v1",
        api_key="lm-studio"
    )

    try:
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            faiss_folder = os.path.join("faiss_indexes", base_filename)
            faiss_index_path = os.path.join(faiss_folder, "faiss.index")

            if not os.path.exists(faiss_index_path):
                print(f"⚠️ FAISS index not found for {filename}")
                continue

            faiss_index = faiss.read_index(faiss_index_path)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                persist_dir=faiss_folder,
                vector_store=vector_store
            )

            docstore = storage_context.docstore
            if not hasattr(docstore, "docs") or not docstore.docs:
                print(f"⚠️ Docstore is empty for {filename}")
                continue

            nodes = list(docstore.docs.values())
            all_nodes.extend(nodes)

        if not all_nodes:
            return jsonify({"error": "No valid document content found in selected PDFs."}), 404

        temp_index = VectorStoreIndex(
            nodes=all_nodes,
            embed_model=embed_model
        )

        similarity_proc = SimilarityPostprocessor(similarity_cutoff=0.2)

        query_engine = temp_index.as_query_engine(
            llm=llm,
            system_prompt=system_prompt,
            node_postprocessors=[similarity_proc]
        )

        response = query_engine.query(question)
        formatted = format_llm_response(response.response)
        return jsonify({"response": formatted})

    except Exception as e:
        print("❌ Error querying documents:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_base="http://localhost:1234/v1",
        api_key="lm-studio"
    )

    try:
        # Embed and normalize the query
        query_embedding = embed_model.get_text_embedding(query)
        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)  # Normalize query for cosine similarity

        results = []

        for filename in os.listdir(ONEDRIVE_SHARED_FOLDER):
            if not filename.lower().endswith(".pdf"):
                continue

            base_filename = os.path.splitext(filename)[0]
            faiss_folder = os.path.join("faiss_indexes_ip", base_filename)
            faiss_index_path = os.path.join(faiss_folder, "faiss.index")
            node_embedding_path = os.path.join(faiss_folder, "node_embeddings.npy")
            node_id_path = os.path.join(faiss_folder, "node_ids.txt")

        
            try:
                # Load the index
                faiss_index = faiss.read_index(faiss_index_path)

                # Safety: confirm it's cosine-compatible
                if not isinstance(faiss_index, faiss.IndexFlatIP):
                    print(f"⚠️ {filename} index is not IP-based. Skipping.")
                    continue

                # Perform the search
                scores, indices = faiss_index.search(query_vector, k=1)
            

                if scores[0][0] >= 0.0:  # You can adjust this threshold
                    results.append({
                        "filename": filename,
                        "score": float(scores[0][0])
                    })

                print(f"✅ {filename} | Score: {scores[0][0]}")

            except Exception as e:
                print(f"❌ Error searching {filename}: {e}")

        results = sorted(results, key=lambda x: -x["score"])
        return jsonify({"matches": results})

    except Exception as e:
        print("❌ Error in semantic search:", e)
        return jsonify({"error": str(e)}), 500





# === Init ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        index_local_pdfs()
    app.run(debug=True) 