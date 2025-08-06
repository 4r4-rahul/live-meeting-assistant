from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
import os


router = APIRouter()


# Use the new OpenAI client for openai>=1.0.0
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SmartRequest(BaseModel):
    transcript: str
    role: str = "You are a senior software engineer. Respond with smart, friendly, high-energy insights that add value."

@router.post("/smart-respond/")
async def smart_respond(req: SmartRequest):
    import traceback
    # Validate input: transcript must not be empty or only whitespace
    if not req.transcript or not req.transcript.strip():
        print("[ERROR] Transcript is empty or whitespace. Input:", req.transcript)
        raise HTTPException(status_code=400, detail="Transcript is empty. Please provide meeting text or audio.")
    prompt = f"{req.role}\n\nMeeting transcript: {req.transcript}\n\nReply:"
    try:
        print("[DEBUG] Sending prompt to OpenAI:", prompt)
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content
        print("[DEBUG] OpenAI response:", reply)
        return {"response": reply}
    except Exception as e:
        print("[ERROR] Exception in smart_respond:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
