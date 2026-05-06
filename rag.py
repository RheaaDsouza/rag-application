import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# persist_dir = "./chroma_langchain_db"
persist_dir = str(Path(__file__).parent / "chroma_langchain_db")

# load documents
def load_document(file_path):
    if Path(file_path).suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif Path(file_path).suffix.lower() in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type.")
    return loader.load()

# Split documents into chunks
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    

    embeddings =  GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=api_key,
        task_type="retrieval_document",
    )
    return embeddings

def vector_db(chunks=None):
    embeddings = get_embeddings()
    if chunks is None:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

def embed_and_store(docs):
    store = vector_db()
    if store._collection.count() > 0:
        return store.add_documents(documents=docs)
    store = vector_db(docs)
    return store.get()["ids"] or []

def ingest_docs(file_path, source_name=None):
    document = load_document(file_path)
    if source_name:
        for doc in document:
            doc.metadata["source"] = source_name
    texts = chunk_documents(document)
    embed_and_store(texts)
    return {
        "file": source_name or Path(file_path).name,
        "pages": len(document),
        "chunks": len(texts),
    }

def retrieve_docs():
    vector_store = vector_db()
    if vector_store._collection.count() == 0:
        return []
    all_docs = vector_store._collection.get()
    if not all_docs or not all_docs.get("metadatas"):
        return []
    seen = set()
    files = []
    for meta in all_docs["metadatas"]:
        src = meta.get("source", "")
        src_path = Path(src)
        # Ignore entries created from temporary upload paths.
        if src_path.is_absolute() and ("tmp" in src_path.parts or "var" in src_path.parts):
            continue
        name = Path(src).name
        if name and name not in seen:
            seen.add(name)
            files.append(name)
    return sorted(files)


def rag_chain():
    vector_store = vector_db()
    if vector_store._collection.count() == 0:
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        max_retries=5, 
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    contextualize_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "only reformulate it if needed, otherwise return it as is."
        "IMPORTANT: Keep the reformulated question"
        "under 50 words and remove any special characters or URLs."
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_system_prompt = (
        "Use the following retrieved context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain =  create_retrieval_chain(history_aware_retriever, qa_chain)
    return chain

def ask(chain, question, chat_history=None):
    history = []
    if chat_history:
        for role, content, _ in chat_history:
            if role == "user":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))

    response = chain.invoke({
        "input": question,
        "chat_history": history,
    })

    sources = list({
        Path(doc.metadata.get("source", "unknown")).name
        for doc in response.get("context", [])
    })

    return {
        "answer":  response["answer"],
        "sources": sources,
    }


   
if __name__ == "__main__":

    demo_path = ""
    out = ingest_docs(demo_path)
    print(out)
    print("Files in store:", retrieve_docs())
