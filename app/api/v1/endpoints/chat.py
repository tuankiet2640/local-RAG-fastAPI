from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from ....models.chat_request_response import ChatRequest, ChatResponse
from ....models.chat_message import Message
from ....services.rag_service import RAGService
from ....services.conversation_service import ConversationService
from datetime import datetime
import uuid
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
rag_service = RAGService()
conversation_service = ConversationService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and get a response using RAG"""
    
    # Create or retrieve conversation
    if request.conversation_id:
        # Existing conversation
        conversation = conversation_service.get_conversation(request.conversation_id)
        conversation_id = request.conversation_id
    else:
        # New conversation
        conversation_id = str(uuid.uuid4())
        title = request.new_conversation_title or f"Conversation {len(conversation_service.conversations) + 1}"
        conversation = conversation_service.create_conversation(conversation_id, title)
    
    # Add user message to conversation
    user_message = Message(
        id=str(uuid.uuid4()),
        role="user", 
        content=request.message,
        timestamp=datetime.now()
    )
    conversation_service.add_message(conversation_id, user_message)
    
    # Get RAG response
    try:
        chain = rag_service.get_rag_chain(conversation_id, conversation_service)
        result = chain({"question": request.message})
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            for i, doc in enumerate(result["source_documents"]):
                source = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source)
        
        # Create assistant message
        assistant_message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=result["answer"],
            timestamp=datetime.now()
        )
        
        # Add to conversation history
        conversation_service.add_message(conversation_id, assistant_message)
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )