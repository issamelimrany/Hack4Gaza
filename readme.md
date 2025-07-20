# Expert Finder API

A FastAPI application that finds relevant experts using local vector search and provides AI-generated answers to queries.

## Setup

1. **Install dependencies**:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

3. **Configure ChromaDB**:
   - create a data folder
   - ChromaDB will be automatically initialized with sample data
   - Data is stored locally in `data/chroma_db/`
   - No external setup required

4. **Configure your LLM provider** :
   - Add your LLM provider API key to `.env` for AI responses


## Running the Application

```bash

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
