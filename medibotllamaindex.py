import os
import base64 
import binascii
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.core.schema import MetadataMode

#from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptTemplate


from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank


import json
import hashlib

from flask import Flask, request, Response

from waitress import serve
from flask_cors import CORS

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=5,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Build llm models
llm = Bedrock(
    model="mistral.mixtral-8x7b-instruct-v0:1",
    temperature=0,
    timeout=60,
    context_size=10000,
    max_tokens=2000,
)

#batch size=1000 takes 5 minutes 30 seconds to start
embed_model = BedrockEmbedding(
    #model = "amazon.titan-embed-text-v1",
    model = "cohere.embed-english-v3",
    )
Settings.llm =llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=250, chunk_overlap=10)

def decodefilename(filepath):
    name = os.path.basename(filepath)
    #print(name)
    try:
        filenamebase64_bytes = name.encode("ascii") 
    
        filename_bytes = base64.b64decode(filenamebase64_bytes) 
        decodedfilename = filename_bytes.decode("ascii")
        return decodedfilename
    except (binascii.Error, UnicodeDecodeError) as error:
        return name

filename_fn = lambda filename: {"file_name": decodefilename(filename)}
documents = SimpleDirectoryReader(
    "htmlpages/knowledge/",
    file_metadata=filename_fn,
    recursive=True).load_data()

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3, verbose=True)

reranker = RankGPTRerank(
    top_n = 3,
    llm = llm,
)

# Build a prompt template to only provide answers based on the loaded documents 
template = (
"We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "Don't give an answer unless it is supported by the context above.\n"
    "Answer the question as truthfully as possible strictly using only the provided context.\n"
    "If the answer is not contained within the context and chat history, just reply 'I don\'t know' directly.\n"
    "Skip any preamble text and reasoning and give just the answer.\n"
    "Do not mention what context states, just answer the question.\n"
    "Do not proviide a summary of the context and the question.\n"
    "Provide all urls of the context where the answer is found.\n"
    "The file_name is base64 encoded, decode it by using base64 before return as the references.\n"
)

qa_template = PromptTemplate(template)

def getQueryResult(query):
    # display source for debug purpose
    nodes = retriever.retrieve(query)
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle = query_bundle)
    # Retrieve the context from the model
    context_list = [n.get_content(metadata_mode=MetadataMode.ALL) for n in ranked_nodes]
    prompt = qa_template.format(context_str="\n\n".join(context_list), query_str=query)

    # Generate the response 
    response = llm.complete(prompt)
    return str(response)

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

    args_string = key+"{'usermessage': '" + usermessage + "}"
    # Use hashlib to create a hash of the query string for a unique and consistent cache key
    key_hash = hashlib.md5(args_string.encode('utf-8')).hexdigest()
    data = cache.get(key_hash)
    if (data is None):
        response = getQueryResult(usermessage)
        data=str(response)
        cache[key_hash] = data
    res = {}
    res["data"] = data
    res_json = json.dumps(res)
    return Response(response=res_json, status=201, mimetype='application/json', headers={'Access-Control-Allow-Origin': '*'})

@app.get('/health/liveness')
def on_get_liveness():
    return Response(response='OK', status=200)

@app.get('/health/readiness')
def on_get_rediness():
    return Response(response='OK', status=200)

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)