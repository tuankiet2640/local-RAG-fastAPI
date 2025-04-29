from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.endpoints import chat, conversation, document
from .api.v1.endpoints.knowledge_base import router as kb_router
import logging
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Local RAG API Service",
    description="A FastAPI service for Retrieval Augmented Generation with chat history",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers DRAFTING
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(document.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(conversation.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(kb_router, prefix="/api/v1", tags=["knowledge_bases"])

@app.get("/")
async def root():
    return {"message": "Welcome to Local RAG API Service :3"}