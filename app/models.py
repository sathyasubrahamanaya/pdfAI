from sqlmodel import SQLModel, Field, create_engine, Session
import uuid
from datetime import datetime
import bcrypt
import streamlit as st

# Database setup
sqlite_url = "sqlite:///.pdfai.db"
engine = create_engine(sqlite_url)
SQLModel.metadata.create_all(engine)

class User(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
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