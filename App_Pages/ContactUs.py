

import streamlit as st

import requests
from streamlit_lottie import st_lottie

def load_lottie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def app():
    col1, col2 = st.columns([0.4,0.6])
    with col1:
        lottie_picture=load_lottie("https://assets5.lottiefiles.com/packages/lf20_ucns0iaz.json")
        st_lottie(lottie_picture,height=345, key="I")
    with col2:
        st.write("")
        st.subheader(":mailbox: Get Connected with us")
        contact_form = f"""
            <form action = "https://formsubmit.co/Markrjmarte9@gmail.com" method = "POST">
                <input type= "hidden" name="_captcha" value="false">
                <input type="hidden" name="_template" value="table">
                <input type="text" name="name" placeholder="Your email" required>
                <input type = "email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder ="Your message here"></textarea>
                <button type = "submit">Send</button>
           </form>
            """
        st.markdown(contact_form,unsafe_allow_html=True)
        local_css("App_Pages/style.css")
