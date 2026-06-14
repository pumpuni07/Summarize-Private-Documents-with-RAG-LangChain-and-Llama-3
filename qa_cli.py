from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

DB_PATH = "vectorstore"


def qa():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=HuggingFaceHub(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"temperature": 0.2, "max_length": 2048},
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=False,
    )

    history = []
    while True:
        query = input("Question: ")

        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = qa_chain({"question": query})

        history.append((query, result["answer"]))

        print("Answer:", result["answer"])


if __name__ == "__main__":
    qa()
