from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import json
import hashlib

from flask import Flask, request, Response
from waitress import serve
from flask_cors import CORS

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0",
                  model_kwargs={"temperature": 0.1},)

embeddings = BedrockEmbeddings(
    #model_id="amazon.titan-embed-text-v1",
    model_id="cohere.embed-english-v3",
)

loader = DirectoryLoader(
    'htmlpages/knowledge/', 
    glob="**/*.json", 
    loader_cls=TextLoader, 
    recursive=True, 
    silent_errors=True, 
    use_multithreading=True)

docs = loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 100,)
splits = spliter.split_documents(docs)

db = Chroma.from_documents(documents=splits, embedding=embeddings)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert on Medirecords system. Answer the question based on the following context from Medirecords knowledge base@https://support.medirecords.com/hc/en-us 
            and the chat history:
            <context>
            {context}
            Answer the question as truthfully as possible strictly using only the provided context, and if the answer is not contained within the context and chat history, just reply "I don't know" directly.
            Skip any preamble text and reasoning and give just the answer.
            Do not mention what context states, just answer the question.
            Do not proviide a summary of the context and the question.""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

store = {}

session_id = "medibot_chat_session_123"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

runnable = prompt | llm

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    context_messages_key="context",
)

def getQuery(query_text, session_id):
    embedded_query = embeddings.embed_query(query_text)
    results = db.similarity_search_by_vector(embedding=embedded_query, k=8)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    #print(context_text)
    msg = with_message_history.invoke(
        { "input": query_text,
         "context": context_text},
        config={"configurable": {"session_id": session_id}},
        )
    return msg.content

app = Flask(__name__)
CORS(app)

cache = {}

@app.post('/medibot/chat')
def on_get_chat():
    key = 'get_chat'
    req = request.get_json()
    usermessage = req.get('usermessage', '')
    session_id = req.get('session_id', 'medibot_chat_session_123')
    if not usermessage:
        return Response(json.dumps({"error": "Usermessage not provided"}), status=400, mimetype='application/json')

    args_string = key+"{'usermessage': '" + usermessage + "}"
    # Use hashlib to create a hash of the query string for a unique and consistent cache key
    key_hash = hashlib.md5(args_string.encode('utf-8')).hexdigest()
    data = cache.get(key_hash)
    if (data is None):
        response = getQuery(usermessage, session_id)
        data=response
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