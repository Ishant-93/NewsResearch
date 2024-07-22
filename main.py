import os
import streamlit as st
import dill 
import joblib
import time
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# Initialize session state variables
if 'retr' not in st.session_state:
    st.session_state.retr = None
if 'docs' not in st.session_state:
    st.session_state.docs = []

# Initialize LLM
llm = GoogleGenerativeAI(model='gemini-pro', google_api_key=os.getenv('GOOGLE_API_KEY'))

# Streamlit UI
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# Get URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(1)]

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

# Process URLs and build vector store
if process_url_clicked:
    with st.spinner("Processing URLs..."):
        try:
            loader = UnstructuredURLLoader(urls=urls)
            st.session_state.docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            st.session_state.docs = splitter.split_documents(st.session_state.docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore_content = FAISS.from_documents(st.session_state.docs, embeddings)
            st.session_state.retr = vectorstore_content.as_retriever()
            
            st.success("Processing complete!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Setup retrieval chain and query
retr = st.session_state.retr
if retr:
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    query = st.text_input("Question:")
    if query:
        chain = create_retrieval_chain(retr, combine_docs_chain)
        result = chain.invoke({"input": query})
        
        st.header("Answer")
        st.subheader(result["answer"])
else:
    st.info("Please process URLs to initialize the retriever.")
