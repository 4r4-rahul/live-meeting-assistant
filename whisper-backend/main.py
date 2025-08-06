# Load environment variables from custom env file and print the path for debug
import os
from dotenv import load_dotenv
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../environment/dev.env'))
print("Loading dotenv from:", dotenv_path)
load_dotenv(dotenv_path=dotenv_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# Import and include smart_respond router
from smart_respond import router as smart_respond_router
app.include_router(smart_respond_router)

# File upload endpoint for context extraction (must be after app = FastAPI())
@app.post("/upload-context/")
async def upload_context(file: UploadFile = File(...)):
    import io, traceback
    text = ""
    try:
        filename = file.filename.lower()
        print(f"[DEBUG] Received file: {filename}, content_type: {file.content_type}")
        if filename.endswith(".pdf"):
            import pdfplumber
            file.file.seek(0)
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text
            print(f"[DEBUG] Extracted PDF text length: {len(text)}")
        elif filename.endswith(".txt"):
            file.file.seek(0)
            text = (await file.read()).decode("utf-8", errors="ignore")
            print(f"[DEBUG] Extracted TXT text length: {len(text)}")
        elif filename.endswith(".docx"):
            import docx
            file.file.seek(0)
            doc = docx.Document(file.file)
            text = "\n".join([para.text for para in doc.paragraphs])
            print(f"[DEBUG] Extracted DOCX text length: {len(text)}")
        else:
            print(f"[ERROR] Unsupported file type: {filename}")
            return {"error": "Unsupported file type. Please upload PDF, TXT, or DOCX."}
    except Exception as e:
        print(f"[ERROR] File extraction failed: {str(e)}")
        traceback.print_exc()
        return {"error": f"File extraction failed: {str(e)}"}
    return {"context_text": text}

# Conversational Q&A endpoint using LangChain
from typing import List, Tuple
class ConversationalQARequest(BaseModel):
    transcript: str
    question: str
    chat_history: List[Tuple[str, str]] = []

# Streaming conversational QA endpoint
from fastapi.responses import StreamingResponse
@app.post("/conversational-qa/stream/")
async def conversational_qa_stream(req: ConversationalQARequest):
    print("[DEBUG] /conversational-qa/stream/ request:", req)
    # Validate input: transcript must not be empty or only whitespace
    if not req.transcript or not req.transcript.strip():
        def error_stream():
            yield "[ERROR] Transcript is empty. Please provide meeting text or audio."
        return StreamingResponse(error_stream(), media_type="text/plain")
    prompt = f"Meeting transcript: {req.transcript}\n\nQuestion: {req.question}\n\nReply:"
    def token_stream():
        try:
            response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and hasattr(delta, "content") and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"[ERROR] {str(e)}"
    return StreamingResponse(token_stream(), media_type="text/plain")

@app.post("/conversational-qa/")
async def conversational_qa(req: ConversationalQARequest):
    print("[DEBUG] /conversational-qa/ request:", req)
    # Validate input: transcript must not be empty or only whitespace
    if not req.transcript or not req.transcript.strip():
        return {"error": "Transcript is empty. Please provide meeting text or audio."}
    documents = [req.transcript]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Prevent FAISS error if docs or embeddings are empty
    if not docs or not docs[0].page_content.strip():
        return {"error": "Transcript is empty after preprocessing. Please provide valid meeting text or audio."}
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


