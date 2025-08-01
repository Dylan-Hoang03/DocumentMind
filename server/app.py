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
from docx import Document

import re
import fitz  
from io import BytesIO
import docx2txt 

#inport modules


load_dotenv()

#import DB info from .env file


#temp connection to Lam's file on sharepoint

###!!!!! THIS IS THE FOLDER THAT YOU ARE USING AS THE INFORMATION DATABASE!!!###
ONEDRIVE_SHARED_FOLDER = r"\\vtnweb1\\Edoc_Data"

#creaate app
app = Flask(__name__)
CORS(app)


from llama_index.core.schema import Document
#indexxing pdfsfrom docx import Document as DocxDocument  # Import to handle DOCX files

import re

def safe_filename(name):
    # Remove illegal Windows filename characters and strip whitespace
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name.strip().rstrip('.')  # also 

def index_local_pdfs():
    print("Scanning shared folder for PDFs and DOCX files...")

    for root, dirs, files in os.walk(ONEDRIVE_SHARED_FOLDER):
        if "obsoleted" in root:
            continue
        for filename in files:
            if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
                continue

            full_path = os.path.join(root, filename)
            base_filename = safe_filename(os.path.splitext(filename)[0])
            faiss_folder = os.path.join("faiss_indexes", base_filename)
            faiss_index_l2_path = os.path.join(faiss_folder, "faiss_l2.index")
            faiss_index_ip_path = os.path.join(faiss_folder, "faiss_ip.index")

            if os.path.exists(faiss_index_l2_path) and os.path.exists(faiss_index_ip_path):
                print(f"Both FAISS indices already exist for {filename}, skipping.")
                continue

            try:
                documents = None
                tmp_path = None

                if filename.lower().endswith(".pdf"):
                    with open(full_path, "rb") as f:
                        file_content = f.read()
                    if not file_content:
                        print(f"⚠️ Empty PDF file: {filename}, skipping.")
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_content)
                        tmp_path = tmp.name

                    print(f"Created temporary file for processing: {tmp_path}")
                    documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()

                elif filename.lower().endswith(".docx"):
                    text = docx2txt.process(full_path)
                    if not text.strip():
                        print(f"⚠️ Empty DOCX content in {filename}, skipping.")
                        continue
                    documents = [Document(text=text, metadata={"filename": filename})]

                if not documents:
                    print(f"⚠️ No content extracted from {filename}, skipping.")
                    continue

                embed_model = OpenAIEmbedding(
                    model="text-embedding-ada-002",
                    api_base="http://localhost:1234/v1",
                    api_key="lm-studio"
                )

                # L2 index
                faiss_index_l2 = faiss.IndexFlatL2(768)
                vector_store_l2 = FaissVectorStore(faiss_index=faiss_index_l2)
                storage_context_l2 = StorageContext.from_defaults(vector_store=vector_store_l2)

                # IP index
                faiss_index_ip = faiss.IndexFlatIP(768)
                vector_store_ip = FaissVectorStore(faiss_index=faiss_index_ip)
                storage_context_ip = StorageContext.from_defaults(vector_store=vector_store_ip)

                index_l2 = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context_l2, embed_model=embed_model
                )

                index_ip = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context_ip, embed_model=embed_model
                )

                os.makedirs(faiss_folder, exist_ok=True)
                print(f"Saving FAISS indices to: {faiss_folder}")

                try:
                    faiss.write_index(faiss_index_l2, faiss_index_l2_path)
                    faiss.write_index(faiss_index_ip, faiss_index_ip_path)
                    storage_context_l2.persist(persist_dir=faiss_folder)
                    storage_context_ip.persist(persist_dir=faiss_folder)
                    print(f"✅ Indexing completed for {filename}")
                except Exception as e:
                    print(f"❌ Failed to write FAISS indices for {filename}: {e}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"🧹 Temporary file deleted: {tmp_path}")




#formatting list answers and paragraphs
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

#showing pdfs on the sidebar method (outdated due to scope change)

@app.route("/list-pdfs", methods=["GET"])
def list_pdfs():
    print("/list-pdfs triggered — indexing any new PDFs...")
    pdfs = []

    # Use os.walk() to traverse directories and subdirectories
    for root, dirs, files in os.walk(ONEDRIVE_SHARED_FOLDER): 
        for filename in files:
 
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(root, filename)
            base_filename = os.path.splitext(filename)[0]
            faiss_folder = os.path.join("faiss_indexes", base_filename)
            faiss_index_path = os.path.join(faiss_folder, "faiss.index")

            pdfs.append(filename)

            # Check if FAISS index already exists
            if os.path.exists(faiss_index_path):
                continue

            print(f"Indexing {filename} on-demand...")

            try:
                with open(full_path, "rb") as f:
                    file_content = f.read()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name

                documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
                print(f"Extracted {len(documents)} chunks from {filename}")

                if not documents:
                    print(f"No content found in {filename}, skipping.")
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

                print(f"FAISS index generated for {filename}")

            except Exception as e:
                print(f"Error indexing {filename}: {e}")

            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"Temp file deleted: {tmp_path}")

    return jsonify({"pdfs": pdfs})



#ask LLM
@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.get_json()
    question = data.get("question")
    question += " answer only according to the documents /no_think"

    if not question:
        return jsonify({"error": "Missing question"}), 400

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

    all_nodes = []  # We will collect all documents

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_base="http://localhost:1234/v1",
        api_key="lm-studio"
    )

    try:
        # Search through all documents, no need for filenames
        for root, dirs, files in os.walk(ONEDRIVE_SHARED_FOLDER):
            for filename in files:
                
                if not filename.lower().endswith(".pdf"):
                    continue

                # Construct the file path and faiss index path
                full_path = os.path.join(root, filename)
                base_filename = os.path.splitext(filename)[0]
                faiss_folder = os.path.join("faiss_indexes", base_filename)
                faiss_index_path = os.path.join(faiss_folder, "faiss.index")

                if not os.path.exists(faiss_index_path):
                    continue  # Skip if index doesn't exist for this file

                faiss_index = faiss.read_index(faiss_index_path)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(
                    persist_dir=faiss_folder,
                    vector_store=vector_store
                )

                docstore = storage_context.docstore
                if not hasattr(docstore, "docs") or not docstore.docs:
                    continue  # Skip if no documents in the index

                nodes = list(docstore.docs.values())
                all_nodes.extend(nodes)  # Add the nodes (documents) to the list

        if not all_nodes:
            return jsonify({"error": "No valid document content found in all PDFs."}), 404

        # Create a temporary index from all the documents
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
        print("start query")
        response = query_engine.query(question)
        formatted = format_llm_response(response.response)
        return jsonify({"response": formatted})

    except Exception as e:
        print("Error querying documents:", e)
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
        processed_files = set()  # A set to track processed filenames

        # Walk through the directory to find all PDFs and search their FAISS indices
        for root, dirs, files in os.walk(ONEDRIVE_SHARED_FOLDER):
            for filename in files:
                print(f"Processing: {filename}")
                if not filename.lower().endswith(".pdf"):
                    continue
                
                # Check if the document has already been processed
                if filename in processed_files:
                    continue

                base_filename = os.path.splitext(filename)[0]
                faiss_folder = os.path.join("faiss_indexes", base_filename)

                # Paths for both indices (L2 and IP)
                faiss_index_l2_path = os.path.join(faiss_folder, "faiss_l2.index")
                faiss_index_ip_path = os.path.join(faiss_folder, "faiss_ip.index")

                try:
                    # Load FAISS index for IP (Inner Product)
                    if os.path.exists(faiss_index_ip_path):
                        faiss_index_ip = faiss.read_index(faiss_index_ip_path)
                        if not isinstance(faiss_index_ip, faiss.IndexFlatIP):
                            print(f"{filename} index is not IP-based. Skipping.")
                        else:
                            scores, indices = faiss_index_ip.search(query_vector, k=3)
                            if scores[0][0] >= 0.0:
                                results.append({
                                    "filename": filename,
                                    "score": float(scores[0][0]),
                                    "index_type": "IP"
                                })
                    
                    # Load FAISS index for L2 (Euclidean distance)
                    elif os.path.exists(faiss_index_l2_path):
                        faiss_index_l2 = faiss.read_index(faiss_index_l2_path)
                        if not isinstance(faiss_index_l2, faiss.IndexFlatL2):
                            print(f"{filename} index is not L2-based. Skipping.")
                        else:
                            scores, indices = faiss_index_l2.search(query_vector, k=3)
                            if scores[0][0] >= 0.0:
                                results.append({
                                    "filename": filename,
                                    "score": float(scores[0][0]),
                                    "index_type": "L2"
                                })

                except Exception as e:
                    print(f"Error searching {filename}: {e}")

                # Add filename to the processed set to avoid duplicate results
                processed_files.add(filename)

        # If no results are found, return an appropriate message
        if not results:
            return jsonify({"message": "No matching documents found."}), 404

        # Sort the results by score in descending order
        results = sorted(results, key=lambda x: -x["score"])

        return jsonify({"matches": results})

    except Exception as e:
        print("Error in semantic search:", e)
        return jsonify({"error": str(e)}), 500


# === Init ===
if __name__ == "__main__":
    with app.app_context():
        # index_local_pdfs() ##you can comment this out if database remains unchanged
        app.run(debug=True) 