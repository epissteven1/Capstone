import streamlit as st
 
def app():
    st.empty()
    c=st.container()
    with c :
        st.markdown("""
        <style>
            .topnav {
                background-color: #0ed145;
                overflow: hidden;
            }
            .topnav a {
                display: block;
                color: #f2f2f2;
                text-align: Left;
                padding: 14px 16px;
                text-decoration: none;
                font-size: 22px;
            }
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="topnav">
                <a>How to open streamlit file?</a>
            </div>
            <div class="Paragraph">
                <a>1.First locate the file location.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I1.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="Paragraph">
                <a>2.Open the App folder or Any folder where you've saved the File “main.py” or the Main app(.py).</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I2.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="Paragraph">
                <a>3.Open Command Prompt or CMD.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I3.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="Paragraph">
                <a>4.In CMD, encode “streamlit run main.py” or streamlit run (then the Main App (.py)) then Enter.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I4.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="Paragraph">
                <a>5.After that, the Streamlit App can be view in your browser.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I5.png',use_column_width=True)
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.markdown("""
        <style>
            .topnav {
                background-color: #0ed145;
                overflow: hidden;
            }
            .topnav a {
                display: block;
                color: #f2f2f2;
                text-align: Left;
                padding: 14px 16px;
                text-decoration: none;
                font-size: 22px;
            }
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="topnav">
                <a>How to use Corn Disease Idenintification App?</a>
            </div>
            <div class="Paragraph">
                <a>1.In Uploading an image. Just click the BROWSE FILE BUTTON.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)

        st.image('App_Images/I6.png',use_column_width=True)
        st.markdown("""
            <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
            </style>
            <body>
            <div class="Paragraph">
                <a>2.Select an image that you want to use.</a>
            </div>
            </body>
            """,unsafe_allow_html=True)
        st.image('App_Images/I7.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }   
        </style>
        <body>
            <div class="Paragraph">
                <a>3.After the image uploaded. Click PREDICT BUTTON and wait for the result.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I8.png',use_column_width=True)
        st.markdown("""
        <style>
            .Paragraph {
                background-color: #e1ffca;
                overflow: hidden;
            }
            .Paragraph a {
                display: block;
                color: black;
                text-align: Left;
                padding: 14px 16px;
                font-size: 17px;
            }
        </style>
        <body>
            <div class="Paragraph">
                <a>4.The application will predict the image and state the identification with the probability percentage.</a>
            </div>
        </body>
        """,unsafe_allow_html=True)
        st.image('App_Images/I9.png',use_column_width=True)
    
    












    
       
   
    
    
    
