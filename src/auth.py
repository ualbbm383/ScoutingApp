import streamlit as st

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Acceso privado")
    st.write("Introduce las credenciales para acceder a la aplicación.")

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    valid_user = st.secrets.get("APP_USERNAME", "admin")
    valid_password = st.secrets.get("APP_PASSWORD", "admin")

    if st.button("Entrar", width="stretch"):
        if username == valid_user and password == valid_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")

    return False