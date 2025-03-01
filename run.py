# Import libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document  # Import correctly
from langchain_huggingface import HuggingFacePipeline
import os
import requests
import streamlit as st

# PDF Document Loader
loader = PyPDFLoader('bc_cat_2020.pdf')

# Load the document
data = loader.load()
print(data[0])

# Define a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', ' ', ''],
    chunk_size=200,  
    chunk_overlap=50  
)

# Split the document
chunks = text_splitter.split_documents(data)
print(chunks)
print([len(chunk.page_content) for chunk in chunks])

# Hugging Face Model Setup
api_url = "https://api-inference.huggingface.co/models/gpt2"
api_token = "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"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# Example usage of Hugging Face API
prompt = "Hugging Face is"
result = query({"inputs": prompt})
print(result)

# Other example prompts
prompt_1 = "Give me more details about B&C Oxford Line"
result_1 = query({"inputs": prompt_1})
print(result_1)

prompt_2 = "How many and what exactly colors has B&C KING HOODED?"
result_2 = query({"inputs": prompt_2})
print(result_2)

# Streamlit App Setup
st.title("Hugging Face GPT-2 Inference")

user_input = st.text_input("Enter your prompt:")

if user_input:
    result = query({"inputs": user_input})
    st.write(result)
