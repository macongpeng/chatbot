#!/usr/bin/env python3
"""
Hybrid Search LlamaIndex Implementation for MediRecords Chatbot
Combines BM25 (keyword matching) + Vector Search (semantic similarity)
Based on successful medibotbeflask_hybrid.py approach
"""

import os
import base64 
import binascii
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.core.schema import MetadataMode

# Hybrid search components
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.prompts import PromptTemplate

import json
import hashlib

from flask import Flask, request, Response
from waitress import serve
from flask_cors import CORS

# HYBRID OPTIMIZATION: Better models configuration
llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0,  # More deterministic  
    max_tokens=2000,
    region_name="ap-southeast-2"
)

embed_model = BedrockEmbedding(
    model_name="cohere.embed-english-v3",
    region_name="ap-southeast-2"
)

Settings.llm = llm
Settings.embed_model = embed_model

# HYBRID OPTIMIZATION: Better chunking (from successful hybrid research)
Settings.chunk_size = 800  # Larger chunks for complete instructions
Settings.chunk_overlap = 100  # More overlap for context continuity

def decodefilename(filepath):
    """Decode base64 encoded filenames"""
    name = os.path.basename(filepath)
    try:
        filenamebase64_bytes = name.encode("ascii") 
        filename_bytes = base64.b64decode(filenamebase64_bytes) 
        decodedfilename = filename_bytes.decode("ascii")
        return decodedfilename
    except (binascii.Error, UnicodeDecodeError) as error:
        return name

# HYBRID OPTIMIZATION: Enhanced document loading with better metadata
print("📚 Loading documents for hybrid search...")
filename_fn = lambda filename: {"file_name": decodefilename(filename)}
documents = SimpleDirectoryReader(
    "htmlpages/knowledge/",
    file_metadata=filename_fn,
    recursive=True
).load_data()

print(f"✅ Loaded {len(documents)} documents")

# HYBRID OPTIMIZATION: Better node splitting  
splitter = SentenceSplitter(
    chunk_size=800,  # Match Settings
    chunk_overlap=100,
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"📄 Created {len(nodes)} nodes for hybrid search")

# HYBRID COMPONENT 1: Create Vector Index (semantic search)
print("🔍 Building vector index...")
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=6)  # More candidates

# HYBRID COMPONENT 2: Create BM25 Retriever (keyword search) 
print("📝 Building BM25 index...")
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=6,  # More candidates
)

# HYBRID COMPONENT 3: Query Fusion Retriever (combining both approaches)
print("🔄 Creating hybrid retriever...")
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=4,  # Final top_k after fusion
    num_queries=1,  # Don't generate multiple query variations
    mode="reciprocal_rerank",  # Use reciprocal rank fusion (RRF)
    use_async=False,
    verbose=False  # Set to True for debugging
)

# HYBRID COMPONENT 4: Query preprocessing (from successful approach)
def preprocess_medical_query(query: str) -> str:
    """Preprocess queries to improve hybrid search performance"""
    query_lower = query.lower()
    
    if "appointment" in query_lower and "create" in query_lower:
        return f"MediRecords {query} step-by-step tutorial instructions"
    elif "medicare" in query_lower and ("billing" in query_lower or "claim" in query_lower):
        return f"MediRecords {query} process workflow steps"
    elif "patient" in query_lower and ("add" in query_lower or "new" in query_lower):
        return f"MediRecords {query} registration procedure steps"
    elif "sms" in query_lower and "reminder" in query_lower:
        return f"MediRecords {query} configuration setup steps"
    else:
        return f"MediRecords {query} how-to guide"

print("✅ Hybrid search system ready!")

# Enhanced prompt for medical precision
template = (
    "We have provided context information below from MediRecords EMR system documentation.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Instructions:\n"
    "- Focus on specific MediRecords functionality and features\n" 
    "- Provide step-by-step procedures when available\n"
    "- Use exact terminology from the system\n"
    "- Don't give an answer unless it is supported by the context above\n"
    "- Answer truthfully using only the provided context\n"
    "- If the answer is not contained within the context, reply 'I don't know' directly\n"
    "- Skip preamble text and reasoning, give just the answer\n"
    "- Do not mention what context states, just answer the question\n"
    "- Do not provide a summary of the context and question\n"
    "- Provide file_names of the context as references where the answer is found\n"
    "\nGiven the above information, please answer the question: {query_str}\n"
)

qa_template = PromptTemplate(template)

def getQueryResult(query: str) -> str:
    """Enhanced query function with hybrid retrieval and preprocessing"""
    
    # HYBRID ENHANCEMENT: Preprocess query for better results
    processed_query = preprocess_medical_query(query)
    
    # HYBRID RETRIEVAL: Use fusion of BM25 + Vector search
    nodes = hybrid_retriever.retrieve(processed_query)
    
    # Build context from hybrid results
    context_str = ""
    source_files = []
    
    for node in nodes:
        context_str += node.node.text + "\n\n"
        # Extract source file information
        file_name = node.node.metadata.get('file_name', 'Unknown')
        if file_name not in source_files:
            source_files.append(file_name)
    
    # Format context with source information
    context_str += f"\nSource files: {', '.join(source_files)}"
    
    # Generate response using enhanced template
    formatted_prompt = qa_template.format(context_str=context_str, query_str=query)
    response = llm.complete(formatted_prompt)
    
    return response.text

app = Flask(__name__)
CORS(app)

cache = {}

@app.post('/medibot/chat')
def on_get_chat():
    key = 'get_chat'
    req = request.get_json()
    usermessage = req.get('usermessage', '')
    if not usermessage:
        return Response(json.dumps({"error": "Usermessage not provided"}), status=400, mimetype='application/json')

    args_string = key + "{'usermessage': '" + usermessage + "}"
    key_hash = hashlib.md5(args_string.encode('utf-8')).hexdigest()
    data = cache.get(key_hash)
    
    if data is None:
        response = getQueryResult(usermessage)
        data = str(response)
        cache[key_hash] = data
        
    res = {"data": data}
    res_json = json.dumps(res)
    return Response(response=res_json, status=201, mimetype='application/json', headers={'Access-Control-Allow-Origin': '*'})

@app.get('/health/liveness')
def on_get_liveness():
    return Response(response='OK', status=200)

@app.get('/health/readiness') 
def on_get_readiness():
    return Response(response='OK', status=200)

@app.get('/debug/search/<query>')
def debug_hybrid_search(query):
    """Debug endpoint to see hybrid search results"""
    try:
        processed_query = preprocess_medical_query(query)
        nodes = hybrid_retriever.retrieve(processed_query)
        
        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank': i + 1,
                'content': node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                'metadata': node.node.metadata,
                'score': getattr(node, 'score', 'N/A')
            })
        
        return Response(
            response=json.dumps({
                'original_query': query,
                'processed_query': processed_query,
                'hybrid_results': results
            }, indent=2),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return Response(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )

if __name__ == '__main__':
    print("🚀 Starting MediRecords LlamaIndex Hybrid Search Chatbot")
    print("🔍 Using BM25 + Vector Search for improved precision")
    print("🌐 Visit http://localhost:8080/debug/search/<your_query> to test search")
    serve(app, host="0.0.0.0", port=8080)