from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables
load_dotenv()

def text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def chunks_from_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    # Add embeddings to the FAISS index
    for chunk in text_chunks:
        embedding = embeddings.embed_query(chunk)
        embedding = np.array(embedding).reshape(1, -1)  # Convert to NumPy array and reshape to 2D
        index.add(embedding)  # Add the reshaped embedding to the index

    return index, embeddings

def search_vector_store(query, vector_store, embeddings, k=3):
    # Convert query to embedding and search the vector store
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    distances, indices = vector_store.search(query_embedding, k)  # Search for the top k nearest neighbors
    return indices

def llama_response(prompt):
    client = ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"), 
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    response = ""
    for chunk in client.stream([{"role":"user","content":prompt}]): 
        response += chunk.content
    return response

def main():
    st.set_page_config(
        page_title='Chat with PDF',
        page_icon=":books:"
    )
    st.header("Chat with multiple PDFs")
    user_query = st.text_input("Ask a question")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload files here", accept_multiple_files=True)  # Accept multiple files

        if pdf_docs:  # Trigger processing immediately upon upload
            with st.spinner("Processing"):
                # Convert PDFs to text
                raw_text = text_from_pdf(pdf_docs)

                # Split text into chunks
                text_chunks = chunks_from_text(raw_text)

                # Store embeddings in the vector store
                vector_store, embeddings = get_vector_store(text_chunks)

                st.success("Embeddings have been stored in the vector database.")

    if st.button("Submit"):
        if user_query:
            try:
                # Search for the most relevant text chunks
                indices = search_vector_store(user_query, vector_store, embeddings, k=3)
                context = "\n".join([text_chunks[i] for i in indices[0]])  # Retrieve the top k relevant chunks

                # Generate a response using the retrieved context
                summary = llama_response(f"{context}\n{user_query}")
                st.write(summary)
            except Exception as e:
                st.write(f"Error generating response: {str(e)}")

if __name__ == '__main__':
    main()
