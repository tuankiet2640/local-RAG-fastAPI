# Local RAG Service with FastAPI

A flexible Retrieval-Augmented Generation (RAG) service that supports multiple LLM and embedding providers, built with FastAPI.

## Features

- Support for multiple LLM providers:
  - Ollama (local models)
  - OpenAI
  - Azure OpenAI
- Support for multiple embedding providers
- Document processing for PDF, CSV, and TXT files
- FAISS vector store for efficient similarity search
- Conversation memory for chat history
- Fully configurable through environment variables
- Centralized configuration management
- Comprehensive error handling and logging
- Dependency validation for different providers

## Recent Improvements

- **Enhanced Provider Architecture**: Implemented a flexible provider-based architecture that allows easy switching between Ollama, OpenAI, and Azure OpenAI without code changes.
  
- **Centralized Configuration**: All settings are now managed through a central `.env` file with proper validation.
  
- **Improved Error Handling**: Added comprehensive error handling with helpful error messages for missing API keys, invalid configurations, and more.
  
- **Dependency Management**: Automatic validation of required dependencies based on the selected provider.
  
- **Updated LLM Integration**: Updated to the latest LangChain packages, including `langchain-ollama`, `langchain-openai`, and fixed compatibility issues with the newer OpenAI client.
  
- **Enhanced Logging**: Added detailed logging throughout the application for better debugging and monitoring.

## Prerequisites

- Python 3.8+
- Ollama (if using Ollama provider)
- OpenAI API key (if using OpenAI provider)
- Azure OpenAI credentials (if using Azure OpenAI provider)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd local-Rag-FastAPI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your preferred provider configuration:

### For Ollama (default)
```env
# Provider settings
PROVIDER_TYPE=ollama

# Ollama settings
OLLAMA_EMBEDDING_MODEL=llama2
OLLAMA_LLM_MODEL=mistral
OLLAMA_TEMPERATURE=0.7
```

### For OpenAI
```env
# Provider settings
PROVIDER_TYPE=openai

# OpenAI settings
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_LLM_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
```

### For Azure OpenAI
```env
# Provider settings
PROVIDER_TYPE=azure_openai

# Azure OpenAI settings
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_BASE=your-api-base
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
AZURE_OPENAI_LLM_MODEL=gpt-35-turbo
AZURE_OPENAI_TEMPERATURE=0.7
```

## Usage

### Starting the Service

```bash
uvicorn app.main:app --reload
```

### API Endpoints

- `POST /api/v1/documents/upload`: Upload documents for RAG processing
- `POST /api/v1/chat`: Send messages and get responses
- `GET /api/v1/health`: Check service health

### Example Usage

```python
from app.services.rag import create_rag_service

# Create RAG service with configuration from .env
rag_service = create_rag_service()

# Access the embeddings and LLMs
embeddings = rag_service.embedding_provider.get_embeddings()
llm = rag_service.llm_provider.get_llm()
```

## Project Structure

```
app/
├── config.py                # Centralized configuration 
├── main.py                  # FastAPI application entry point
├── models/                  # Data models
│   ├── document.py
│   └── chat_message.py
├── services/
│   ├── rag_service.py       # Main RAG service wrapper
│   └── rag/                 # Provider-based architecture
│       ├── __init__.py
│       ├── base.py          # Base provider interfaces
│       ├── factory.py       # Provider factory
│       ├── service.py       # Core RAG implementation
│       └── providers/       # Provider implementations
│           ├── embeddings.py
│           └── llm.py
├── utils/                   # Utility functions
│   ├── file_utils.py
│   └── metadata_utils.py
└── api/                     # API endpoints
    └── v1/
        ├── documents.py
        ├── chat.py
        └── health.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
