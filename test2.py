from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
model  = LocalLLM(
    api_base = "http://localhost:11434/v1",
    model = "llama3.1:8b"
)


uploaded_file = st.file_uploader("upload file here", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))
    df = SmartDataframe(data , config = {"llm":model})
    prompt = st.text_area("Enter prompt")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response ..... "):
                st.write(df.chat(prompt))
        else:
            st.warning("Enter prompt")

