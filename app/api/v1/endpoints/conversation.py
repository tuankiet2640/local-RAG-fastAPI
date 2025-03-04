from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import logging
from ....models.conversation import Conversation
from ....services.conversation_service import ConversationService

router = APIRouter()
logger = logging.getLogger(__name__)
conversation_service = ConversationService()

@router.get("/conversations", response_model=List[Conversation])
async def list_conversations():
    """Get all conversations"""
    try:
        return list(conversation_service.conversations.values())
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing conversations: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation_by_id(conversation_id: str):
    """Get a specific conversation by ID"""
    try:
        return conversation_service.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation"""
    try:
        conversation_service.delete_conversation(conversation_id)
        return None
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}"
        )