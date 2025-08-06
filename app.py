from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from dotenv import load_dotenv
import os
import faiss

# === Load environment variables ===
load_dotenv()

# === Configuration ===
DATA_DIR = "data/ancient_greece_data"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
SIMILARITY_THRESHOLD = 0.5  # Lowered to allow slightly looser matches

# === Check data directory exists ===
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory {DATA_DIR} not found.")

# === Load text documents from directory ===
documents = SimpleDirectoryReader(DATA_DIR).load_data()

# === Load and set embedding model globally ===
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model

# === Create FAISS vector index ===
faiss_index = faiss.IndexFlatL2(EMBED_DIM)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# === Build the index and create query engine ===
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=5, response_mode="no_text")

# === Initialize Flask app ===
app = Flask(__name__)

# === Route: Homepage ===
@app.route('/')
def home():
    return render_template("index.html")

# === Route: Handle chatbot queries ===
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Perform the query using the index
    response = query_engine.query(question)

    # Filter source nodes based on similarity threshold
    relevant_nodes = [
        node for node in response.source_nodes
        if node.score is None or node.score > SIMILARITY_THRESHOLD
    ]

    # If no relevant nodes found, return fallback response
    if not relevant_nodes:
        return jsonify({
            "answer": "I don’t have enough information to answer that.",
            "sources": []
        })

    # === Build the final answer and list of cited sources ===
    answer_parts = []
    source_files = set()  # Use set to avoid duplicates

    for node in relevant_nodes:
        answer_parts.append(node.node.text)

        # Extract filename or ID from metadata
        source_path = node.node.metadata.get('file_path', node.node.node_id)
        file_name = os.path.basename(source_path)
        source_files.add(file_name)

    # Combine all parts of answer
    final_answer = "\n\n".join(answer_parts)

    # Create citation string
    citation_text = "This answer is based on information from: " + ", ".join(source_files) + "."

    # Append citation at the end
    final_answer += f"\n\n📁 {citation_text}"

    return jsonify({
        "answer": final_answer,
        "sources": list(source_files)  # Send filenames to frontend (optional)
    })

# === Run the Flask server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
