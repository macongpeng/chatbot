from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


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
    model = "amazon.titan-embed-image-v1",
    )
Settings.llm =llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("htmlpages/knowledge/", recursive=True).load_data()

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
#nodes = splitter.get_nodes_from_documents(documents)

nodes = node_parser.get_nodes_from_documents(documents)
base_nodes = splitter.get_nodes_from_documents(documents)

sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

query_engine = sentence_index.as_query_engine(
    similarity_top_k=4,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.
Provide all urls of the context where the answer is found.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

# list of `ChatMessage` objects
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hello assistant, I am a user of Medirecord EMR system. I have a few questions regarding the system to clarify.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okay, I am here to help."),
]

#query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=False,
)


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
        response = chat_engine.chat(usermessage)
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