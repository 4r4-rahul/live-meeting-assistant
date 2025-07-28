from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Use the new OpenAI client for openai>=1.0.0
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SmartRequest(BaseModel):
    transcript: str
    role: str = "You are a senior software engineer. Respond with smart, friendly, high-energy insights that add value."

@app.post("/smart-respond/")
async def smart_respond(req: SmartRequest):
    prompt = f"{req.role}\n\nMeeting transcript: {req.transcript}\n\nReply:"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}
