# Local RAG Service with FastAPI

A flexible Retrieval-Augmented Generation (RAG) service that supports multiple LLM and embedding providers, built with FastAPI.

## Features

- Support for multiple LLM providers:
  - Ollama (default)
  - OpenAI
  - Azure OpenAI
- Support for multiple embedding providers
- Document processing for PDF, CSV, and TXT files
- FAISS vector store for efficient similarity search
- Conversation memory for chat history
- Configurable through environment variables

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
RAG_PROVIDER_TYPE=ollama
OLLAMA_EMBEDDING_MODEL=llama2
OLLAMA_LLM_MODEL=mistral
OLLAMA_TEMPERATURE=0.7
```

### For OpenAI
```env
RAG_PROVIDER_TYPE=openai
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_LLM_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
```

### For Azure OpenAI
```env
RAG_PROVIDER_TYPE=azure_openai
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

- `POST /upload`: Upload documents for RAG processing
- `POST /chat`: Send messages and get responses
- `GET /health`: Check service health

### Example Usage

```python
from app.services.rag import create_rag_service

# Create RAG service with default configuration
rag_service = create_rag_service()

# Or with custom configuration
from app.services.rag import RAGConfig, ProviderType
config = RAGConfig(
    provider_type=ProviderType.OPENAI,
    openai_config=OpenAIConfig(
        api_key="your-api-key",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0.7
    )
)
rag_service = create_rag_service(config)
```

## Project Structure

```
app/
├── models/
│   ├── document.py
│   └── chat_message.py
├── services/
│   └── rag/
│       ├── __init__.py
│       ├── base.py
│       ├── config.py
│       ├── factory.py
│       ├── service.py
│       └── providers/
│           ├── __init__.py
│           ├── embeddings.py
│           └── llm.py
└── main.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
