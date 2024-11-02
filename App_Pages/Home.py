import streamlit as st


def app():
    st.empty()
    # Add the top navigation bar
    st.markdown("""
        <style>
            .topnav {
                background-color: #0ed145;
                overflow: hidden;
                color: white;
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
                overflow: hidden; /* Prevent scrolling */
            }
        </style>
        <div class="topnav"></div>
    """, unsafe_allow_html=True)

    # Display the image with automatic scaling
    st.image(image="App_Images/dashboard.png", use_column_width=True)

    # CSS to control layout and prevent scrolling
    st.markdown("""
        <style>
            [data-testid="stAppViewBlockContainer"] {
                padding: 0;
                margin: 0;
                overflow: hidden; /* Prevent scrolling */
            }
            img {
                width: 100%;
                height: auto;
                object-fit: fill;
            }
            @media only screen and (max-width: 600px) {
                img {
                    width: 100%!important;
                    height: auto!important;
                    object-fit: contain;
                }
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
                color: black;
                text-align: left;
                padding: 5px 16px;
                font-size: 17px;
            }
        </style>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    app()
