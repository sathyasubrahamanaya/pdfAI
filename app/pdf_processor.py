import os
import uuid
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

def load_documents(file_path):
    loader = UnstructuredPDFLoader(
        file_path, 
        poppler_path="/usr/bin/pdftoppm", 
        tesseract_path="/usr/bin/tesseract"
    )
    return loader.load()

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=500,
        chunk_overlap=50
    )
    doc_chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(doc_chunks, embeddings)

def create_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="gemma2-9b-it",
        temperature=0.7
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

def pdf_chat_page():
    st.set_page_config(
        page_title="Pdf-AI",
        page_icon="📑",
        layout="wide"
    )


    st.title("📄 Chat with your PDF 🔓")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    if not st.session_state.pdf_uploaded:
        st.subheader("Upload your PDF (Only 1 PDF per session)")
        uploaded_file = st.file_uploader(
            "Upload your PDF",
            type="pdf",
            disabled=st.session_state.pdf_uploaded,
            key="pdf_uploader"
        )
        if uploaded_file:
            unique_filename = f"{uuid.uuid4()}.pdf"
            file_path = os.path.join(working_dir, unique_filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing your PDF, please wait..."):
                try:
                    documents = load_documents(file_path)
                    st.session_state.vectorstore = setup_vectorstore(documents)
                    st.session_state.chain = create_chain(st.session_state.vectorstore)
                    st.session_state.pdf_uploaded = True
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
    else:
        st.warning("You have reached the PDF limit. Please restart the session or contact support.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input and st.session_state.pdf_uploaded:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.chain({"question": user_input})
                    assistant_response = response["answer"]
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")