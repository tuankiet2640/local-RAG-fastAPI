from datetime import datetime
from typing import Dict
from ..models.conversation import Conversation, Message
from fastapi import HTTPException, status


class ConversationService:
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}

    def create_conversation(self, conversation_id: str, title: str) -> Conversation:
        """Create a new conversation."""
        if conversation_id in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} already exists")

        conversation = Conversation(id=conversation_id, title=title)
        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID."""
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation with ID {conversation_id} not found")
        return self.conversations[conversation_id]

    def add_message(self, conversation_id: str, message: Message):
        """Add a message to a conversation."""
        conversation = self.get_conversation(conversation_id)
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation by ID."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]