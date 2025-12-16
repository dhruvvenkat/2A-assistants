from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

texts = []
for file in Path("text").glob("*.txt"):
    texts.append(file.read_text())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

documents = splitter.create_documents(texts)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="vectordb"
)

print(f"Indexed {len(documents)} chunks.")
