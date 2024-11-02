import streamlit as st
import base64
from PIL import Image

def app():
    # Load and display the background image as a full-screen image
    img = Image.open("App_Images/dashboard.png")

    # Display the image at the top of the app
    st.image(img, use_column_width=True)

    # Custom header bar
    st.markdown("""
        <style>
            .topnav {
                background-color: #0ed145;
                overflow: hidden;
            }
            .topnav a {
                display: block;
                color: white;
                text-align: center;
                padding: 14px 16px;
                font-size: 22px;
            }
        </style>
        <div class="topnav"></div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    app()
