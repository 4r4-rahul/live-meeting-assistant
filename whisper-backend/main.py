
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import whisper
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Conversational Q&A endpoint using LangChain
from typing import List, Tuple
class ConversationalQARequest(BaseModel):
    transcript: str
    question: str
    chat_history: List[Tuple[str, str]] = []

@app.post("/conversational-qa/")
async def conversational_qa(req: ConversationalQARequest):
    print("[DEBUG] /conversational-qa/ request:", req)
    documents = [req.transcript]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=retriever
    )
    import traceback
    try:
        print("[DEBUG] Calling LangChain qa.invoke()...")
        result = qa.invoke({"question": req.question, "chat_history": req.chat_history})
        print("[DEBUG] LangChain result:", result)
        answer = result.get("answer", "")
        # Update chat history
        updated_history = req.chat_history + [(req.question, answer)]
        print("[DEBUG] Returning answer:", answer)
        print("[DEBUG] Returning chat_history:", updated_history)
        response = {"answer": answer, "chat_history": updated_history}
        print("[DEBUG] Response to client:", response)
        return response
    except Exception as e:
        print("[ERROR] LangChain exception:", str(e))
        traceback.print_exc()
        return {"error": str(e)}

# Context-aware Q&A endpoint using LangChain
class ContextQARequest(BaseModel):
    transcript: str
    question: str

@app.post("/context-qa/")
async def context_qa(req: ContextQARequest):
    # For demo: use transcript as the only document
    documents = [req.transcript]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        chain_type="stuff",
        retriever=retriever
    )
    try:
        answer = qa.run(req.question)
        print("[DEBUG] ContextQA answer:", answer)
        response = {"answer": answer}
        print("[DEBUG] Response to client:", response)
        return response
    except Exception as e:
        return {"error": str(e)}

# Load environment variables from custom env file and print the path for debug
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../environment/dev.env'))
print("Loading dotenv from:", dotenv_path)
load_dotenv(dotenv_path=dotenv_path)


