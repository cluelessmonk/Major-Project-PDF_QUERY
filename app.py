import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.llms import Ollama 
from webscrape import get_webscrape_data
from pandasai import SmartDataframe
llm = Ollama(model="pdf_query")
import os
from dotenv import load_dotenv
load_dotenv()
from pandasai.llm.openai import OpenAI
llm2  = OpenAI(api_token=os.getenv("OPENAI_API_KEY"),model = "gpt-3.5-turbo-instruct")
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)


def get_pdf_text(pdf_docs):
    """Extracts text from multiple PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def custom_text_splitter(text, chunk_size=1000, overlap=200):
    """Manually splits text into chunks with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def compute_embeddings(text_chunks):
    """Generates embeddings for text chunks using a transformer model."""
    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        # Mean pooling for embeddings
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(chunk_embedding)
    return np.array(embeddings)


def cosine_similarity(vector1, vector2):
    """Computes cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    return dot_product / (norm_a * norm_b)


def search_vectorstore(query_embedding, embeddings, top_k=3):
    """Searches for the most similar chunks using cosine similarity."""
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    return sorted_indices, similarities


def manual_conversation_chain(query, embeddings, text_chunks):
    """Retrieves top chunks relevant to the query."""
    query_embedding = compute_embeddings([query])[0]
    top_indices, _ = search_vectorstore(query_embedding, embeddings)
    results = [text_chunks[i] for i in top_indices]
    return results


def generate_llm_response(query, context):
    """Uses LLM to generate a response based on retrieved context."""
    context_combined = "\n\n".join(context)
    prompt = f"Based on the following context, answer the question:\n\nContext:\n{context_combined}\n\nQuestion: {query}"
    for response in  llm.stream(prompt):
        yield response  


def handle_userinput(user_question, embeddings, text_chunks):
    """Handles user input, retrieves relevant text, and streams LLM response."""
    # Retrieve top chunks
    retrieved_chunks = manual_conversation_chain(user_question, embeddings, text_chunks)

    # Prepare for the streamed response display
    st.subheader("LLM Response:")
    response_placeholder = st.empty()  # Placeholder for the response
    full_response = ""  # To store the accumulated response

    # Stream and display the response
    for token in generate_llm_response(user_question, retrieved_chunks):
        full_response += token
        wrapped_response = full_response.replace("\n", "\n\n")  # Ensure vertical spacing
        response_placeholder.markdown(wrapped_response)  # Use Markdown for better rendering

topics = ["PDF" , "Web-query" , "Dataset visualization"]
def main():
    """Main function to handle the Streamlit app."""
    st.set_page_config(page_title="Chat with Custom LLM")

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    mode = st.selectbox("Select a Topic", topics)
    if mode == "PDF":
        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question and st.session_state.embeddings is not None:
            handle_userinput(user_question, st.session_state.embeddings, st.session_state.text_chunks)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Split text into chunks
                    text_chunks = custom_text_splitter(raw_text)

                    # Compute embeddings for chunks
                    embeddings = compute_embeddings(text_chunks)

                    # Store in session state
                    st.session_state.text_chunks = text_chunks
                    st.session_state.embeddings = embeddings

                    st.success("Documents processed successfully!")
    elif mode == "Web-query":
        st.header("Chat with  web-search query   :globe_with_meridians:")
        user_question = st.text_input("Ask a question about your Web-search:")
        if user_question and st.session_state.embeddings is not None:
            handle_userinput(user_question, st.session_state.embeddings, st.session_state.text_chunks)

        with st.sidebar:
            st.subheader("Your query")
            web_query = st.text_input("Enter your websearch query here")
            if st.button("Process"):
                with st.spinner("Processing the webscrape content"):
                    # Get PDF text
                    raw_text = get_webscrape_data(web_query)

                    # Split text into chunks
                    text_chunks = custom_text_splitter(raw_text)

                    # Compute embeddings for chunks
                    embeddings = compute_embeddings(text_chunks)

                    # Store in session state
                    st.session_state.text_chunks = text_chunks
                    st.session_state.embeddings = embeddings

                    st.success("Web search processed successfully!")
    elif mode == "Dataset visualization":
        st.header("Chat with your CSV Data and its visualizatation :chart_with_upwards_trend:")
        file_uploaded = st.file_uploader("upload your csv", type=["csv"])

        if file_uploaded is not None :
            data = pd.read_csv(file_uploaded)
            df = SmartDataframe(data, config = {"llm" : llm2})
            st.write(data.head(5))
            prompt = st.text_area("Enter your question here")
            if st.button("Generate"):
                if prompt:
                    with st.spinner("Generating Response ... "):
                        st.write(df.chat(prompt))
                else:
                    st.warning("Please Enter a Prompt") 
                                


if __name__ == '__main__':
    main()
