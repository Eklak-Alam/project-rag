import streamlit as st
from dotenv import load_dotenv
from rag_engine import process_uploaded_pdf, get_response_stream 123456

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a PDF document in the sidebar to get started, or ask me any question!"}
    ]

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

if "active_filename" not in st.session_state:
    st.session_state.active_filename = None

# Sidebar: Document Ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.active_filename != uploaded_file.name:
            with st.spinner("Analyzing document..."):
                collection_name, is_cached, chunk_count = process_uploaded_pdf(uploaded_file)
                
                st.session_state.active_collection = collection_name
                st.session_state.active_filename = uploaded_file.name

                if is_cached:
                    st.success(f"⚡ Loaded **{uploaded_file.name}** instantly from cache!")
                else:
                    st.success(f"🎉 Processed **{uploaded_file.name}** into {chunk_count} chunks!")

    if st.session_state.active_filename:
        st.info(f"**Active File:** {st.session_state.active_filename}")

# Main Canvas: Chat Interface
st.title("📚 RAG Document Assistant")

# Render prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handler
if user_query := st.chat_input("Ask a question or say hello..."):
    # Append user query to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate streaming assistant response
    with st.chat_message("assistant"):
        stream_generator, sources = get_response_stream(
            user_query=user_query,
            collection_name=st.session_state.active_collection
        )
        
        # Real-time token streaming to the screen
        full_response = st.write_stream(stream_generator)

        # Render retrieved sources expander
        if sources:
            with st.expander("🔍 View Retrieved Document Sources"):
                for i, doc in enumerate(sources):
                    st.write(f"**Source Chunk {i+1}**")
                    st.caption(doc.page_content)
                    st.divider()

    # Save final response string to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
