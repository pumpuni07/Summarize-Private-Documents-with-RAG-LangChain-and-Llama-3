from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "data"
DB_PATH = "vectorstore"


def build_vectorstore():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.*", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)


if __name__ == "__main__":
    build_vectorstore()
    print("Vector store saved to", DB_PATH)
