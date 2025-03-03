from typing import Dict
from ..models.conversation import Conversation
from fastapi import HTTPException, status

class ConversationService:
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}

    def get_conversation(self, conversation_id: str):
        if conversation_id not in self.conversations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        return self.conversations[conversation_id]

    def create_conversation(self, title: str):
        conversation = Conversation(title=title)
        self.conversations[conversation.id] = conversation
        return conversation

    def delete_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]