from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from .chat_message import Message

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    new_conversation_title: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    message: Message
    sources: List[Dict[str, Any]] = []