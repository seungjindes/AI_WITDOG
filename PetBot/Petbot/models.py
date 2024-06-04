from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Message(BaseModel):
    id: Optional[int] = Field(default=None, description="메시지의 고유 식별자")
    text: str = Field(..., description="메시지의 내용")
    created_at: datetime = Field(default_factory=datetime.now, description="메시지 생성 시간")
    user_id: str = Field(default=None, description="메시지를 보낸 사용자의 식별자 , 쓰레드")
    is_user_message: bool = Field(..., description="사용자로부터 온 메시지인지 여부")
