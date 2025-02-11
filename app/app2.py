from dotenv import load_dotenv
import streamlit as st
from auth import login_page, register_page
from pdf_processor import pdf_chat_page
from models import init_session

load_dotenv()

def main():
    init_session()
    if st.session_state.user_authenticated:
        pdf_chat_page()
    elif st.session_state.show_register:
        register_page()
    else:
        login_page()

if __name__ == "__main__":
    main()