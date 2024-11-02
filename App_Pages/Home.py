import streamlit as st


def app():
    st.empty()
    with st.container():
        st.markdown("""
            <style>
                .topnav {
                    background-color: #0ed145;
                    overflow: hidden;
                    text-color: white;
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
            <body>
                <div class="topnav"></div>
            </body>
        """, unsafe_allow_html=True)

    # Directly load and display the image using Streamlit's st.image
    st.image(image="App_Images/dashboard.png", use_column_width=True)

    st.markdown("""
            <style>
                [data-testid="stAppViewBlockContainer"] {
                    padding: 0;
                    margin: 0;
                    
                }
                 img {
                    width: auto;
                    height: auto;   
                    object-fit: cover;
            }
                 @media only screen and (max-width: 600px){
                img {
                    width:450px!important;
                    height:600px!important;
                    object-fit: fill;
                }
                 }
               [data-testid="stHeader"] {
                    background-color: #333333;
                    padding:0;
                    margin:0;
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
            </style>
            <body>
            </body>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    app()
