import streamlit as st
from models import get_user, create_user, engine, Session

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

    st.button("Back to Login", on_click=lambda: setattr(st.session_state, "show_register", False))
    st.rerun()