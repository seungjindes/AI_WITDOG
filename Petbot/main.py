from fastapi import FASTAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn

from models import Message
fromdatabase import save_message, fetch_messages
from openai_intergration import generate_response 

app = FastAPI()

@app.post("")
