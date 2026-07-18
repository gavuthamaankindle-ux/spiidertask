import os
import fitz
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
import numpy as np

app = Flask(__name__)
CORS(app) # Allows your frontend to communicate with this server seamlessly

# Initialize models globally
vectorstore = None
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}  
)
confidence_evaluator = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

# --- DATA INGESTION ENGINE ---
def readpdf(address):
    global vectorstore
    if not os.path.exists(address):
        print(f" Error: File not found at '{address}'")
        return

    with fitz.open(address) as data:
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""], 
            chunk_size=1000,      
            chunk_overlap=150,    
            is_separator_regex=False
        )
        finalchunks = []
        for i in range(len(data)):
            page = data.load_page(i)
            text = page.get_text()
            if not text.strip():
                continue
            raw_chunks = recursive_splitter.split_text(text)
            for chunk in raw_chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"source": os.path.basename(address), "page": i + 1}
                )
                finalchunks.append(doc)
        
        if finalchunks:
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=finalchunks,
                    embedding=embeddings_model,
                    persist_directory="./chromadb"
                )
            else:
                vectorstore.add_documents(documents=finalchunks)
print("started")
# Ingest existing data directory on startup
# data_folder = 'data'
# if os.path.exists(data_folder):
#     pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
#     if pdf_files:
#         print(f"Found {len(pdf_files)} PDF(s). Processing...")
        
#         for pdf_path in pdf_files:
            
#             readpdf(pdf_path)
#             break
#         print("Ingestion complete.")

print("completed")

# 1. Initialize the identical embedding model used during setup
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}  
)

persist_dir = "./chromadb"

# 2. Safely connect and load the persistent vector store database
if os.path.exists(persist_dir):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings_model
    )
    print(f" Successfully loaded existing collection vector store from '{persist_dir}'")
    print(f"Total items indexed: {len(vectorstore.get()['ids'])}")
else:
    print(f" Error: The directory '{persist_dir}' was not found. Run ingestion first.")
    vectorstore = None


# --- RAG UTILITIES ---
def confidence_score(query: str, retrieved_docs: list, generated_answer: str) -> dict:
    if isinstance(generated_answer, list):
        generated_answer = " ".join([str(item) for item in generated_answer])
    else:
        generated_answer = str(generated_answer)
        
    if not retrieved_docs or not generated_answer.strip():
        return {"score": 0.0, "level": "Very Low (No Data Source)"}
    
    insufficient_keywords = ["insufficient information", "cannot answer", "not mentioned", "i do not know"]
    if any(kw in generated_answer.lower() for kw in insufficient_keywords):
        return {"score": 0.0, "level": "Low (Context Deficient)"}
        
    full_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    retrieval_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    retrieval_scores = confidence_evaluator.predict(retrieval_pairs)
    normalized_retrieval = 1 / (1 + np.exp(-np.array(retrieval_scores)))
    avg_retrieval_score = float(np.mean(normalized_retrieval))

    groundedness_score = confidence_evaluator.predict([full_context, generated_answer])
    normalized_groundedness = float(1 / (1 + np.exp(-groundedness_score)))

    total_score = (avg_retrieval_score * 0.4) + (normalized_groundedness * 0.6)
    total_score = max(0.0, min(1.0, total_score))
    
    level = "High" if total_score >= 0.75 else ("Medium" if total_score >= 0.45 else "Low")
    return {"score": round(total_score, 2), "level": f"{level} ({int(total_score*100)}% verified)"}

# --- CORE API ROUTE ---
@app.route('/query', methods=['POST'])
def handle_query():
    global vectorstore
    
    # Extract structural elements from user FormData payload
    user_query = request.form.get('user_query', '').strip()
    threshold = float(request.form.get('threshold', 0.35))
    
    if not user_query:
        return jsonify({"answer": "Empty query received.", "confidence": None, "citations": []}), 400

    if vectorstore is None:
        return jsonify({
            "answer": "Vectorstore initialization check failed. Please ensure reference PDFs are processed correctly.", 
            "confidence": "None", 
            "citations": []
        })

    # Guardrail checks for safety critical scenarios
    emergency_keywords = ["heart attack", "stroke", "dying", "emergency", "suicide", "overdose"]
    if any(w in user_query.lower() for w in emergency_keywords):
        return jsonify({
            "answer": "CRITICAL SAFETY NOTICE: If you are experiencing a medical emergency, please contact your local emergency services (like 911, 102, or 112) immediately.",
            "confidence": "Critical Guardrail Triggered",
            "citations": ["System Safety Override Engine"]
        })

    # Execute Vector space similarity lookup
    rawdoc = vectorstore.similarity_search_with_relevance_scores(user_query, k=5)
    finaldoc = [doc for doc, score in rawdoc if score > threshold]

    if not finaldoc:
        return jsonify({
            "answer": "Based on the retrieved evidence, there is insufficient information to answer this query. No source documents crossed the required strictness threshold criteria.",
            "confidence": "Low (Threshold Drop)",
            "citations": []
        })

    context_str = ""
    citations = []
    for idx, doc in enumerate(finaldoc):
        src = doc.metadata.get("source", "Unknown Document")
        pg = doc.metadata.get("page", "?")
        ref_tag = f"[Source {idx+1}]"
        context_str += f"{ref_tag} (Page: {pg}):\n{doc.page_content}\n\n"
        citations.append(f"{ref_tag} {src}, Page {pg}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0.0,
        max_output_tokens=512,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    system_instruction = (
        "You are a strict, objective Medical AI Assistant. Your task is to answer the user's question "
        "using ONLY the provided context blocks below. Follow these core guidelines:\n"
        "1. GROUNDEDNESS: Rely exclusively on the clear facts directly mentioned in the context.\n"
        "2. CITATIONS: Include reference tags (e.g., [Source 1]) exactly adjacent to target claims.\n"
        "3. INSUFFICIENT EVIDENCE: Return a context deficiency statement if info is missing."
    )

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=f"--- CONTEXT ---\n{context_str}\n--- USER QUERY ---\n{user_query}")
    ]
    
    response = llm.invoke(messages)
    raw_response = response.content

    # Calculate hybrid evaluation metric structures
    confidence_data = confidence_score(user_query, finaldoc, raw_response)

    return jsonify({
        "answer": raw_response,
        "confidence": confidence_data['level'],
        "citations": citations
    })

# Fallback index container point if hosted together
@app.route('/')
def index():
    return "Medical AI Engine Server running on port 5000."

if __name__ == '__main__':
    # Ensure your GOOGLE_API_KEY environment variable is configured before running
    app.run(host='0.0.0.0', port=5000, debug=True)