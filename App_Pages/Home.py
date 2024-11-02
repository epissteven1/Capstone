import streamlit as st


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

    # Set background image and ensure it covers the full screen
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] {
                background-image: url("App_Images/dashboard.png");
                background-size: cover;
                background-position: center;
                padding: 0;
                margin: 0;
                height: 100vh;
            }
            [data-testid="stHeader"] {
                background-color: #333333;
                padding: 0;
                margin: 0;
            }
            .sidebar {
                float: right;
                width: 50%;
                padding: 0 20px 20px 15px;
            }
            .sidebar p {
                display: block;
                color: black;
                text-align: left;
                padding: 5px 16px;
                font-size: 17px;
            }
            #content {
                text-align: left;
                width: 100%;
                padding: 5px 16px;
            }
            .Paragraph {
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: left;
                padding: 5px 40px;
                font-size: 17px;
            }
            /* Mobile screen adjustments */
            @media only screen and (max-width: 600px) {
                [data-testid="stAppViewContainer"] {
                    background-size: auto 100%;
                    background-repeat: no-repeat;
                }
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    app()
