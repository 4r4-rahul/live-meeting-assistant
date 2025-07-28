# LangChain ConversationalRetrievalChain example for human-like conversations
# This script demonstrates how to use LangChain to enable multi-turn, context-aware conversations with retrieval.

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
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
retriever = vectorstore.as_retriever()

# Set up the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(openai_api_key=openai_api_key),
    retriever=retriever
)

# Example conversation
chat_history = []
query1 = "What is LangChain?"
result1 = qa({"question": query1, "chat_history": chat_history})
print("Q1:", query1)
print("A1:", result1["answer"])
chat_history.append((query1, result1["answer"]))

query2 = "What library does it use for vector search?"
result2 = qa({"question": query2, "chat_history": chat_history})
print("Q2:", query2)
print("A2:", result2["answer"])
chat_history.append((query2, result2["answer"]))

query3 = "Who provides the language models?"
result3 = qa({"question": query3, "chat_history": chat_history})
print("Q3:", query3)
print("A3:", result3["answer"])
