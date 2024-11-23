# **MultiPDF & Web Chat App**

## **Overview**

The MultiPDF & Web Chat App is a powerful Python-based tool that allows you to interact with multiple data sources using natural language. You can upload PDFs, perform web searches, or analyze datasets, and the app will intelligently process your input to generate meaningful responses. Leveraging advanced language models and embeddings, this app ensures accurate and context-aware answers tailored to your queries.

---

## **Features**

- **Chat with Multiple PDFs**: Upload multiple PDF documents and query their content seamlessly.
- **Web Search Integration**: Perform web searches directly from the app, extract relevant information, and get insightful responses.
- **Dataset Analysis**: Upload CSV files to explore, analyze, and visualize your data interactively.
- **Customizable and Extensible**: Built with modular components for easy adaptation and scaling.

---

## **How It Works**

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

1. **Data Loading**:

   - **PDFs**: Extracts text content from uploaded PDFs.
   - **Web**: Scrapes web content for user-specified queries.
   - **Datasets**: Parses and loads CSV files for interactive analysis.

2. **Text Chunking**:

   - Divides the extracted text into manageable chunks for efficient processing and contextual understanding.

3. **Embedding Generation**:

   - Uses state-of-the-art transformer models to compute vector representations (embeddings) for text chunks.

4. **Similarity Matching**:

   - Employs cosine similarity to identify chunks most relevant to your query.

5. **Response Generation**:

   - Utilizes advanced language models like GPT or Ollama to synthesize accurate, context-aware responses.

6. **Interactive Interface**:
   - Built with Streamlit, providing an intuitive and responsive user experience.

---

## **Setup and Installation**

### **Prerequisites**

- Python 3.9 or above
- Virtual environment (recommended)

### **Installation Steps**

1. Clone the repository:

```bash
git clone https://github.com/your-repo/multipdf-web-chat.git
cd multipdf-web-chat
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your environment:

- Create a `.env` file and add your OpenAI API key:
  ```
  OPENAI_API_KEY=your_openai_api_key
  ```

4. Launch the application:

```bash
streamlit run app.py
```

---

## **Usage Guide**

### **PDF Chat**

1. Open the app in your browser.
2. Upload one or more PDFs using the sidebar.
3. Once processed, ask questions related to the PDFs in the chat interface.

### **Web Query Chat**

1. Enter a search query in the sidebar under the "Web Query" section.
2. Process the query to fetch and analyze the web content.
3. Interact with the chat interface to ask detailed questions about the retrieved content.

### **Dataset Analysis**

1. Upload a CSV file in the "Dataset Visualization" section.
2. Explore the data or ask questions to perform visual or textual analyses.

---

## **Technologies Used**

- **Frontend**:

  - **Streamlit**: For creating an interactive web application.

- **Backend**:

  - **PyPDF2**: For extracting text from PDFs.
  - **BeautifulSoup**: For web scraping.
  - **Transformers**: For embedding generation.
  - **Pandas & PandasAI**: For data analysis and visualization.
  - **Ollama & GPT-3.5-turbo**: For generating language-based responses.

- **Libraries**:
  - **Sentence Transformers**: For efficient semantic search.
  - **FAISS (Optional)**: For scalable embedding-based search (evaluated but not implemented).

---

## **Advantages**

- **User-Friendly**: No prior programming knowledge is required.
- **Scalable**: Supports additional data sources or larger datasets with minor adjustments.
- **Customizable**: Modular architecture enables easy modification.
- **Interactive**: Immediate feedback with real-time response generation.

---

## **Future Enhancements**

- **Integration with FAISS** for large-scale embedding management.
- **Enhanced Data Privacy**: Secure handling of sensitive user data.
- **Support for Additional Formats**: Expand beyond PDFs and CSVs.
- **Offline Mode**: Reduce dependency on external APIs for local usage.
- **Deployment**: Dockerized and hosted versions for seamless access.

---

## **Acknowledgments**

Inspired by modern language processing frameworks and educational resources in AI, including:

- Hugging Face Transformers
- Streamlit Community
- OpenAI APIs
- LangChain Framework

```

```

```

```
