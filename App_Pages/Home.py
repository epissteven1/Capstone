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
            }
        </style>
        <div class="topnav"></div>
    """, unsafe_allow_html=True)

    # Display the image with adjustments for layout
    st.image(image="App_Images/dashboard.png", use_column_width=True)

    # Improved CSS to control layout and sidebar appearance
    st.markdown("""
        <style>
            [data-testid="stAppViewBlockContainer"] {
                padding: 0;
                margin: 0;
            }
            img {
                width: 100%;
                height: auto;
                object-fit: cover;
            }
            @media only screen and (max-width: 600px) {
                img {
                    width: 100%!important;
                    height: auto!important;
                    object-fit: contain;
                }
            }
            [data-testid="stSidebar"] {
                background-color: #f4f4f4;
                padding: 20px;
            }
            .sidebar-content {
                text-align: left;
                padding: 10px;
            }
            .sidebar-content p {
                color: black;
                font-size: 17px;
            }
            .sidebar-content a {
                color: black;
                text-align: left;
                padding: 5px;
                font-size: 17px;
            }
        </style>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    app()
