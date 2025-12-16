from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Embeddings + DB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="vectordb", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 8})

# --- LLM ---
llm = Ollama(model="gemma2:2b", temperature=0.1)

# --- Proper streaming callback ---
class PrintCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.buffer = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.buffer.append(token)

callback_handler = PrintCallbackHandler()

print("Course agent ready. Type 'exit' to quit.")

while True:
    query = input("\n> ")
    if query.lower() in {"exit", "quit"}:
        break

    # Retrieve chunks
    docs_and_scores = db.similarity_search_with_score(query, k=8)
    context_text = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])

    prompt_text = f"""
You are a university TA assistant helping with course material.
Use only the context below to answer the question.
If the answer is not in the context, say you don’t know.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    # Generate with proper callback
    llm.generate([prompt_text], callbacks=[callback_handler])
    print("\n")  # final newline after completion
