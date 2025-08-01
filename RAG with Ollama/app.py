from flask import Flask, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re
import os

load_dotenv()

VECTORSTORE_PATH = "vector_db/global_index"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:latest")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL)
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

PROMPT_TEMPLATE = """
You are an expert Cloud Solution Architect in AWS Cloud Services. Use the provided context to answer the query.
Use short memory if helpful. If unsure, say "I don't know." Be concise (max 3 sentences).

Short Memory:
{chat_memory}

Query: {user_query}
Context: {document_context}
Answer:
"""

@app.route("/query", methods=["POST"])
def query_docs():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400


    history = session.get("chat_history", [])
    memory_str = ""
    for turn in history[-5:]:
        memory_str += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"


    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])


    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    raw_answer = chain.invoke({
        "user_query": query,
        "document_context": context,
        "chat_memory": memory_str
    })

    answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    history.append({"question": query, "answer": answer})
    session["chat_history"] = history

    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset_memory():
    session["chat_history"] = []
    return jsonify({"status": "Memory cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
