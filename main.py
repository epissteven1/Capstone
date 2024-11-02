import streamlit as st
import base64
from App_Pages import Home, AppDescription, Record, ContactUs
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Filipino-to-Baybayin-Voice-Recognition-System",
    page_icon="App_Images/iconb.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("App_Images/irlak.jpg")

# Custom CSS for page styling
st.markdown(f"""
    <style>
        body {{
            text-align: center;
            background-color: white;
        }}

        .sidebar-content {{
            background-color: #242525;
            color: white;
        }}

        footer {{
            visibility: visible;
        }}
        footer:before {{
            content: 'Capstone2 @ 2024-2025: FBVRS';
            color: #FF4B4B;
            display: block;
            position: relative;
            padding: 2px;
            top: 3px;
        }}

        [data-testid="stSidebarContent"] {{
            background-image: url("data:image/png;base64,{img}");
            background-size: cover;
            background-position: left;
            background-color: #242525;
            color: white;
        }}

        /* Mobile adjustments */
        @media only screen and (max-width: 600px) {{
            [data-testid="stSidebarContent"] {{
                object-fit: fill;
            }}
        }}
    </style>
""", unsafe_allow_html=True)

def app():
    menu_list = ["Home", "Transcribe", "Description", "Contact Us"]
    with st.sidebar:
        option = option_menu("MENU",
                             menu_list,
                             icons=['house', 'play', 'sliders', 'telephone'],
                             menu_icon="app-indicator",
                             default_index=0,
                             styles={
                                 "container": {"padding": "5!important"},
                                 "icon": {"color": "#b77b82", "font-size": "20px"},
                                 "nav-link": {"font-size": "12px", "text-align": "left", "margin": "0px",
                                              "--hover-color": "#F6E1D3"},
                                 "nav-link-selected": {"background-color": "#00008B"},
                             })

    # Render selected page
    if option == menu_list[0]:
        Home.app()
    if option == menu_list[1]:
        Record.app()
    if option == menu_list[2]:
        AppDescription.app()
    if option == menu_list[3]:
        ContactUs.app()

if __name__ == '__main__':
    app()
