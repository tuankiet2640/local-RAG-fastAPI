from typing import List
from .chat_message import Message
from pydantic import BaseModel

class ChatHistory(BaseModel):
    messages: List[Message] = []
