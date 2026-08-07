import os
import hashlib
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PERSIST_BASE_DIR = "chroma_db"

# -------------------------------------------------------------------
# 1. Resource Caching
# -------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatDeepSeek(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_API_BASE"),
        temperature=0.3,
        streaming=True  # Enabled streaming
    )

# -------------------------------------------------------------------
# 2. Ingestion Engine
# -------------------------------------------------------------------
def process_uploaded_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    collection_name = f"doc_{file_hash}"
    collection_path = os.path.join(PERSIST_BASE_DIR, collection_name)

    if os.path.exists(collection_path):
        return collection_name, True, 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = get_embeddings()
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=collection_path,
            collection_name=collection_name
        )
        return collection_name, False, len(chunks)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# -------------------------------------------------------------------
# 3. Streaming Query Execution Engine
# -------------------------------------------------------------------
def get_response_stream(user_query: str, collection_name: str = None):
    """
    Retrieves document chunks and returns a stream generator alongside retrieved sources.
    """
    retrieved_docs = []
    context_text = "No document is currently selected."

    if collection_name:
        collection_path = os.path.join(PERSIST_BASE_DIR, collection_name)
        if os.path.exists(collection_path):
            embeddings = get_embeddings()
            vector_db = Chroma(
                persist_directory=collection_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([f"--- Chunk ---\n{doc.page_content}" for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
        
        Guidelines:
        1. If the user's message is a casual greeting, small talk, or general question (e.g., "hi", "how are you", "who are you"), respond naturally and warmly.
        2. If the user asks a question about the document or uploaded content, answer accurately using ONLY the context provided below.
        3. If a document question cannot be answered from the provided context, politely state that the information isn't present in the document.

        Document Context:
        {context}"""),
        ("user", "{input}")
    ])

    llm = get_llm()
    # Adding StrOutputParser ensures the stream yields string tokens directly
    chain = prompt_template | llm | StrOutputParser()
    
    # Returns the generator object for streaming, and retrieved sources
    stream_generator = chain.stream({"context": context_text, "input": user_query})
    
    return stream_generator, retrieved_docs