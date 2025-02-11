import streamlit as st
from auth import get_user, create_user

def login_page():
    st.title("🔒 Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            with st.spinner("Logging in..."):
                user = get_user(email)
                if user and user.verify_password(password):
                    st.session_state.user_authenticated = True
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    # Registration link
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
                with st.spinner("Registering..."):
                    existing_user = get_user(email)
                    if existing_user:
                        st.error("Email already registered")
                    else:
                        create_user(email, password)
                        st.session_state.user_authenticated = True
                        st.session_state.show_register = False
                        st.rerun()

    # Back to login
    st.form("back_to_login_form", clear_on_submit=True)
    st.button("Back to Login", on_click=lambda: setattr(st.session_state, "show_register", False))
    st.rerun()