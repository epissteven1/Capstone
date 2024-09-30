import streamlit as st


def app():
    # Custom CSS
    st.markdown("""
    <style>
    /* Customize the font and color of the subheader */
    .css-h2z8e8 {
        font-size: 24px;
        color: #007BFF; /* Blue color */
    }

    /* Customize the text area */
    .stTextArea textarea {
        background-color: #f0f8ff; /* Light blue background */
        color: #000080; /* Navy text color */
        font-size: 16px;
        border-radius: 10px;
    }

    /* Customize the button */
    div.stButton > button {
        background-color: #4169E1;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 8px 20px;
        border: none;
        cursor: pointer;
    }

    /* Change the button color when hovered */
    div.stButton > button:hover {
        background-color: #0056b3;
        color: #F8F8FF;
    }

    /* Add custom spacing */
    .css-1cpxqw2 {
        padding-top: 3em;
    }
    [data-testid="stAppViewBlockContainer"] {
    background-color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for feedback
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""

    # Add some spacing
    st.write("\n" * 10)

    # Subheader for feedback
    st.subheader("Feedback")

    # Text area for feedback input
    feedback = st.text_area("Submit your feedback", value="", key="feedback_text", on_change=lambda: st.session_state.update({'feedback_text': ""}))
    # Button to submit feedback
    if st.button("Submit Feedback"):
        # Save the feedback to a text file
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")

        # Clear the feedback from the session state
        st.session_state.feedback = ""

        # Thank the user for their feedback
        st.write("Thank you for your feedback!")


if __name__ == '__main__':
    app()
