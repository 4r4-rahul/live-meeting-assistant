# LangChain and FAISS setup for context-aware Q&A
# This script demonstrates how to use LangChain to load documents, create a vector store, and perform retrieval-augmented generation with OpenAI.

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# Load your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Example: Load documents (replace with your own data source)
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "OpenAI provides powerful language models accessible via API."
]

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up retriever
retriever = vectorstore.as_retriever()

# Set up the QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=retriever
)

# Example query
query = "What is LangChain?"
result = qa.run(query)
print("Answer:", result)
