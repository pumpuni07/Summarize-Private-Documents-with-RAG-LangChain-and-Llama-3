from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

DB_PATH = "vectorstore"

app = FastAPI(title="RAG Private Documents API")

class Query(BaseModel):
    question: str


def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.2, "max_length": 2048},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=False,
        get_chat_history=lambda h: h,
    )
    return chain


chain = get_chain()


@app.post("/ask")
def ask(query: Query):
    result = chain({"question": query.question})
    return {"answer": result["answer"]}
