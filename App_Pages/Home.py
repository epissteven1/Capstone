import streamlit as st


def app():
    # Clear any initial elements
    st.empty()
    
    # Apply CSS styling to the page
    with st.container():
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
            <body>
                <div class="topnav"></div>
            </body>
        """, unsafe_allow_html=True)
    
    # Display the image with specific styling adjustments
    st.image("App_Images/dashboard.png", use_column_width=True)
    
    # Apply additional CSS to handle sidebar and image scaling
    st.markdown("""
            <style>
                [data-testid="stAppViewBlockContainer"] {
                    padding: 0;
                    margin: 0;
                }
                
                .sidebar {
                    width: 250px; /* Adjust width of sidebar */
                    padding: 0;
                }
                
                .sidebar .css-1v0mbdj {
                    padding: 20px;
                }

                /* Image adjustments */
                img {
                    width: 100%;
                    height: auto;   
                    object-fit: cover;
                    transition: width 0.5s ease-in-out;
                }

                /* Adjust image width based on sidebar state */
                @media (min-width: 768px) {
                    [data-testid="stSidebar"] + div {
                        max-width: calc(100% - 250px); /* Adjust main area width */
                    }
                }
                
                /* Sidebar appearance */
                [data-testid="stSidebar"] {
                    background-color: #333333;
                    color: white;
                    width: 250px; /* Sidebar width */
                    position: fixed;
                }
                
                /* Adjustments for small screens */
                @media only screen and (max-width: 600px){
                    img {
                        width: 100%!important;
                        height: auto!important;
                        object-fit: fill;
                    }
                }
            </style>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    app()
