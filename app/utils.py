import streamlit as st

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