import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def app():
    st.empty()

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
            body {
                margin: 0;
            }
        </style>
        <div class="topnav"></div>
    """, unsafe_allow_html=True)

    # Encode the image to base64
    img_base64 = get_base64_image("App_Images/dashboard.png")

    # Set background image with encoded base64 string and adjust position
    st.markdown(f"""
        <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
                background-position: 20% center;
                width: auto;
                height: auto;
                object-fit: cover;
            }}
            [data-testid="stHeader"] {{
                background-color: #333333;
                padding: 0;
                margin: 0;
            }}
            .sidebar {{
                float: right;
                width: 50%;
                padding: 0 20px 20px 15px;
            }}
            .sidebar p {{
                display: block;
                color: black;
                text-align: left;
                padding: 5px 16px;
                font-size: 17px;
            }}
            #content {{
                text-align: left;
                width: 100%;
                padding: 5px 16px;
            }}
            .Paragraph {{
                overflow: hidden;
            }}
            .Paragraph a {{
                display: block;
                color: black;
                text-align: left;
                padding: 5px 40px;
                font-size: 17px;
            }}
            /* Mobile screen adjustments */
            @media only screen and (max-width: 600px) {{
                [data-testid="stAppViewContainer"] {{
                    background-size: auto 100%;
                    background-repeat: no-repeat;
                }}
            }}
        </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    app()
