import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pinecone import Pinecone
from langchain_groq import ChatGroq

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Ensure index exists
index_name = 'firstindex'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine'
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Extract text from pdf files
def text_from_pdf(pdf_files):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

        try:
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        finally:
            os.unlink(temp_file_path)
    return text

# Convert text to chunks
def chunks_from_text(raw_text):
    """Split raw text into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

# Convert chunks to embeddings
def initialize_pinecone_index(text_chunks):
    """Store text chunks as embeddings in Pinecone."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Process chunks in smaller batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        vectors = [(str(i + j), embeddings.embed_query(chunk), {"text": chunk}) 
                  for j, chunk in enumerate(batch)]
        index.upsert(vectors)

def get_response_from_query(query, k=3):
    """Retrieve response from LLM using Pinecone search results."""
    try:
        # Get embeddings for the query
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        query_embedding = embeddings.embed_query(query)

        # Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        if not search_results["matches"]:
            return "I cannot find the answer in the provided documents."

        # Combine search results
        context = "\n".join([match["metadata"]["text"] for match in search_results["matches"]])

        # Generate LLM response using ChatNVIDIA
        llm = ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",  # Updated to a valid model name
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )

        # groq_api_key = os.getenv("GROQ_API_KEY")

        # llm = llm=ChatGroq(groq_api_key=groq_api_key,
        #  model_name="mixtral-8x7b-32768")

        # Format the message for the chat model
        messages = [
            {
                "role": "user",
                "content": f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say \"I cannot find the answer in the provided documents.\"\n\nContext: {context}\n\nQuestion: {query}"""
            }
        ]

        # Get the response without streaming
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"Error in get_response_from_query: {str(e)}"

# Streamlit UI
st.title("PDF QA System")
st.write("Upload your PDFs and ask questions based on the content.")

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if uploaded_files:
    if st.button("Process Files"):
        with st.spinner("Processing files..."):
            try:
                raw_text = text_from_pdf(uploaded_files)
                text_chunks = chunks_from_text(raw_text)
                initialize_pinecone_index(text_chunks)
                st.success("Files processed successfully.")
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

question = st.text_input("Ask a question")
if question:
    if st.button("Get Answer"):
        with st.spinner("Retrieving answer..."):
            try:
                answer = get_response_from_query(question)
                st.success(answer)
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
