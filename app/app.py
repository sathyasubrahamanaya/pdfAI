from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import nltk
import uuid
from sqlmodel import SQLModel, Field, create_engine, Session
import bcrypt
from datetime import datetime

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Configure database
sqlite_url = "sqlite:///.pdfai.db"
engine = create_engine(sqlite_url)
SQLModel.metadata.create_all(engine)

class User(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}  # Add this line
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True
    )
    email: str = Field(unique=True, index=True)
    password_hash: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())

def init_session():
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "show_register" not in st.session_state:
        st.session_state.show_register = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chain" not in st.session_state:
        st.session_state.chain = None

def get_user(db: Session, email: str) -> User:
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str) -> User:
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = User(email=email, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def login_page():
    st.title("🔒 Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            with Session(engine) as db:
                user = get_user(db, email)
                if user and user.verify_password(password):
                    st.session_state.user_authenticated = True
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    st.form("register_link_form", clear_on_submit=True)
    st.button("Create New Account", on_click=lambda: setattr(st.session_state, "show_register", True))
    st.rerun()

def register_page():
    st.title("✍️ Register")
    with st.form("register_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            else:
                with Session(engine) as db:
                    existing_user = get_user(db, email)
                    if existing_user:
                        st.error("Email already registered")
                    else:
                        create_user(db, email, password)
                        st.session_state.user_authenticated = True
                        st.session_state.show_register = False
                        st.rerun()

    st.form("back_to_login_form", clear_on_submit=True)
    st.button("Back to Login", on_click=lambda: setattr(st.session_state, "show_register", False))
    st.rerun()

def pdf_chat_page():
    st.set_page_config(
        page_title="Pdf-AI",
        page_icon="📑",
        layout="wide"
    )
    st.title("📄 Chat with your PDF 🔓")

    working_dir = os.path.dirname(os.path.abspath(__file__))

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

    if not st.session_state.pdf_uploaded:
        st.subheader(".Upload your PDF (Only 1 PDF per session)")
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf", disabled=st.session_state.pdf_uploaded, key="pdf_uploader")
        if uploaded_file:
            unique_filename = f"{uuid.uuid4()}.pdf"
            file_path = os.path.join(working_dir, unique_filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("🔮 Processing your PDF, please wait..."):
                try:
                    documents = load_documents(file_path)
                    st.session_state.vectorstore = setup_vectorstore(documents)
                    st.session_state.chain = create_chain(st.session_state.vectorstore)
                    st.session_state.pdf_uploaded = True
                except Exception as e:
                    st.error(f"⚠️ Error processing PDF: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
    else:
        st.warning("🔒 You have reached the PDF limit. Please restart the session or contact support.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input and st.session_state.pdf_uploaded:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("🌟 Generating response..."):
                try:
                    response = st.session_state.chain({"question": user_input})
                    assistant_response = response["answer"]
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"🚀 Error generating response: {str(e)}")

def main():
    load_dotenv()
    init_session()
    with Session(engine) as db:
        if st.session_state.user_authenticated:
            pdf_chat_page()
        elif st.session_state.show_register:
            register_page()
        else:
            login_page()

if __name__ == "__main__":
    main()