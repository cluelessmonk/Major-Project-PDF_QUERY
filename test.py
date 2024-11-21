import streamlit as st
import pandas as pd
import openai
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
from pandasai import SmartDataframe
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_token=OPENAI_API_KEY,
             model = "gpt-3.5-turbo-instruct",)
# Set the OpenAI API key (either set it in environment variables or use directly)



# File uploader to load CSV
file_uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if file_uploaded is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(file_uploaded)

    # Create a SmartDataframe for interactive querying
    df = SmartDataframe(data, config={"llm": llm})

    # Display the first few rows of the dataset
    st.write("Here is a preview of your dataset:")
    st.write(data.head(5))

    # Text input to accept the user’s query
    prompt = st.text_area("Enter your question here")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):
                # Generate the response based on the user’s query
                response = df.chat(prompt)
                st.write(response)
        else:
            st.warning("Please enter a prompt.")
