from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn

from typing import List
from models import Message
from database import save_message, fetch_messages
from openai_intergration import generate_response 

app = FastAPI()

messages: List[Message] = []

@app.post("/messages/", response_model=Message)
async def main_(question_message: Message):
    """
    사용자로부터 메시지를 받아 처리하고, OpenAI API를 통해 생성된 응답을 반환합니다.
    """
    try:
        messages.append(
            question_message
        )
        save_message(question_message)
        # OpenAI API를 호출하여 메시지에 대한 응답을 생성합니다.
        response = generate_response(question_message)

        # 생성된 응답을 클라이언트에게 반환합니다.
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app , host = "0.0.0.0" , port = 8000)