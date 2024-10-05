import streamlit as st

st.set_page_config(
    page_title="Filipino-to-Baybayin-Voice-Recognition-System",
    page_icon="App_Images/iconb.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import statements
import base64
from App_Pages import Home, AppDescription, Record, ContactUs
from streamlit_option_menu import option_menu


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("App_Images/sidebar.png")

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
      .st-emotion-cache-13k62yr {{
            position: absolute;

            color: rgb(193, 231, 247);
            inset: 0px;
            color-scheme: transparent;
            overflow: hidden;

        }}


        footer {{
            visibility: visible;
        }}
        footer:before {{
            content: 'Capstone1 @ 2023-2024: FBVRS';
            color: #FF4B4B;
            display: block;
            position: relative;
            padding: 2px;
            top: 3px;
        }}
        [data-testid="stSidebarContent"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size:cover;
        background-position: center; /* Aligns the image to the left */

        }}

         @media only screen and (max-width: 600px){{
        [data-testid="stSidebarContent"] {{
          object-fit: fill;
        }}
        }}
        
        [data-testid="stVerticalBlockBorderWrapper"] {{
         background-color: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
        background-color: #333333;
        }}
        [data-testid="stAppViewContainer"] {{
        background-color: white;
        }}
        [data-testid="stHeader"] {{
        
        }}

    </style>
""", unsafe_allow_html=True)

# JavaScript for toggling the sidebar collapse/expand
st.markdown("""
    <style>
        /* Button styling for collapsing/expanding the sidebar */
        .toggle-btn {
            position: fixed;
            top: 20px;
            left: 15px;
            z-index: 1000;
            cursor: pointer;
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
        }

        .toggle-btn:hover {
            background-color: #555;
        }

        /* Adjusting sidebar visibility */
        .collapsed .sidebar .css-fg4pbf {
            width: 80px;
        }

        .expanded .sidebar .css-fg4pbf {
            width: 300px;
        }
    </style>

    <script>
        function toggleSidebar() {
            var sidebar = document.querySelector("section[data-testid='stSidebar']");
            var mainblock = document.querySelector("section[data-testid='stAppViewContainer']");
            if (sidebar.style.display === "none") {
                sidebar.style.display = "block";
                mainblock.style.marginLeft = "300px";
                document.getElementById("toggle-btn").innerHTML = "&#9776;";  // Set to burger menu icon
            } else {
                sidebar.style.display = "none";
                mainblock.style.marginLeft = "0px";
                document.getElementById("toggle-btn").innerHTML = "&#x2715;";  // Set to cross icon
            }
        }
    </script>
""", unsafe_allow_html=True)

# Add the toggle button for collapsing/expanding sidebar
st.markdown("""
    <button class="toggle-btn" id="toggle-btn" onclick="toggleSidebar()">&#9776;</button>
""", unsafe_allow_html=True)


# Function to render the app
def app():
    menu_list = ["Home", "Transcribe", "Description", "Contact"]
    with st.sidebar:
        option = option_menu("MENU",
                             menu_list,
                             icons=['house', 'play', 'sliders', 'chat'],
                             menu_icon="app-indicator",
                             default_index=0,
                             styles={
                                 "container": {"padding": "5!important"},
                                 "icon": {"color": "#b77b82", "font-size": "20px"},  # Adjusted for smaller view
                                 "nav-link": {"font-size": "12px", "text-align": "left", "margin": "0px",
                                              "--hover-color": "#F6E1D3"},
                                 "nav-link-selected": {"background-color": "#00008B"},
                                 "nav-link-hover": {"background-color": "#f0f0f5"}  # Hover effect
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
