import streamlit as st
import json
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceEmbeddings

# Set up the embedding model for the vector store
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# Initialize the language model
llm = Ollama(model="automation")

# Template to instruct the model to respond in JSON
query_prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template="""
User Query: {user_query}
Please respond with JSON in the format: 
{{
  'type': 'code' or 'query',
  'output': 'C++ code or request for additional information'
}}
"""
)

# Define a dummy vector store since FAISS requires an embedding model (not used here)
vectorstore = FAISS.from_texts(texts=["dummy text"], embedding=embeddings)

def get_conversation_chain():
    # Set up the conversation memory and retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Dummy retriever for conversational memory only
        memory=memory
    )
    return conversation_chain

def get_llm_response(user_query):
    """
    Sends the query to the model and returns the JSON response.
    Args:
        user_query (str): The automation task description from the user.
    Returns:
        dict: JSON response with 'type' and 'output'.
    """
    formatted_prompt = query_prompt_template.format(user_query=user_query)
    response = st.session_state.conversation({'question': formatted_prompt})
    
    try:
        response_json = json.loads(response['answer'])
    except json.JSONDecodeError:
        st.error("Invalid response format from model.")
        return {"type": "error", "output": "Response format error"}

    return response_json

def handle_response(response_json):
    """
    Processes the model's JSON response and displays or requests additional info.
    """
    response_type = response_json.get("type")
    output = response_json.get("output")

    if response_type == "code":
        # Display the generated C++ code and prompt for execution
        st.code(output, language='cpp')
        if st.button("Execute Code"):
            with open("task.cpp", "w") as file:
                file.write(output)
            st.success("Code saved to task.cpp. Ready for compilation.")
    elif response_type == "query":
        st.write(output)
        followup_input = st.text_input("Provide the additional information:")
        if followup_input:
            # Update the conversation with the follow-up input and maintain context
            st.session_state.conversation({'question': followup_input})
            complete_query = f"{st.session_state.chat_history[-1]} {followup_input}"
            new_response = get_llm_response(complete_query)
            handle_response(new_response)

def main():
    st.set_page_config(page_title="PC Automation Assistant", page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation in session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("PC Automation Assistant")

    user_query = st.text_input("Describe the automation task you need:")
    if user_query:
        st.session_state.chat_history.append(user_query)
        response_json = get_llm_response(user_query)
        handle_response(response_json)

if __name__ == '__main__':
    main()
