import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

DB_PATH = "vectorstore"

st.set_page_config(page_title="RAG Private Documents", page_icon="🔍", layout="wide")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_chain():
    db = load_vectorstore()
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

st.title("🔍 RAG Chatbot for Private Documents")
st.write("Ask questions about your private documents using RAG and conversational memory.")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Your question", key="user_query")

if st.button("Ask") and user_query.strip():
    chain = load_chain()
    result = chain({"question": user_query})
    answer = result["answer"]
    st.session_state.history.append((user_query, answer))
    st.markdown(f"**Answer:** {answer}")

if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
