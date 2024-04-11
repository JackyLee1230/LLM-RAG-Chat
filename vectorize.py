from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


docs = []

loaders = [PyPDFLoader('./pdf/MCCheungTinFu.pdf')]

for file in loaders:
    docs.extend(file.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})  # change to GPU if a CUDA device is available

vectorstore = Chroma.from_documents(
    docs, embedding_function, persist_directory="./chroma_db_ps")

print(vectorstore._collection.count())
