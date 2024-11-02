import streamlit as st


def app():
    st.empty()
    c = st.container()
    with c:
        st.markdown("""
        <style>
            #site_content{
                    width: 100%;
                    overflow: hidden;
                    background-color: #e1ffca;
                } 
            .topnav {
                background-color: #5F9EA0;
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
                <a>Technology Development</a>
            </div>
            <div class="Paragraph">
                <a>This innovative project aims to develop a cutting-edge voice recognition system that translates spoken Tagalog into Baybayin script.
                 The system utilizes the ABAKADA alphabets and focuses on high accuracy and noise cancellation to deliver clear and precise transcriptions. 
                 By leveraging advanced speech-to-text technology, the project seeks to preserve and rejuvenate the Baybayin script, ensuring its continued relevance in the modern digital age, 
                 while incorporating noise-cancellation techniques will enhance the system's reliability and performance.</a>
            </div>
                
        </body>
        """, unsafe_allow_html=True)
    st.image('App_Images/CNN.png', use_column_width=True)
    st.markdown("""
        <style>
            .topnav {
                background-color: #5F9EA0;
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
                <a>Figure 1. Schematic diagram of Convolutional Neural Network (CNN).</a>
            </div>
            <div class="Paragraph">
                <a><p>Figure 1 shows that convolutional neural network (CNN) has one or more convolutional layers and are used mainly for image processing,
                segmentation, features extraction and classification and also for other auto correlated data. The CNN uses a system much like a multilayer
                perception that has been designed for reduced processing requirements. The layer of CNN consist of an input layer, an output layer and hidden
                layer that includes multiple convolutional layers and normalization layers. The effective way in removal of limitations and increase in
                efficiency for image processing results in a system.</p></a>
            </div>
            <div class="topnav">
                <a>How the App Work</a>
            </div>
            <div class="Paragraph">
                <a><p>The Filipino-to-Baybayin Voice Recognition System is designed to translate spoken Tagalog into the traditional Baybayin script seamlessly.
                 When a user speaks a phrase in Tagalog, the app captures the audio and applies noise cancellation to ensure a clear signal. 
                 The clean audio is then processed by a speech recognition engine trained to recognize the ABAKADA alphabet. 
                 This engine converts the spoken words into text, which is subsequently mapped to the corresponding Baybayin characters. 
                  The app's user-friendly interface enables users to view, copy, or export their transcriptions effortlessly. 
                  Additionally, the system continuously learns and adapts to different speech patterns, enhancing its accuracy over time.</p></a>
            </div>
        </body>
        """, unsafe_allow_html=True)
    st.image(image='App_Images/dataflow.png', use_column_width=True)
    st.markdown("""
        <style>
            .topnav {
                background-color: #5F9EA0;
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
        """, unsafe_allow_html=True)
