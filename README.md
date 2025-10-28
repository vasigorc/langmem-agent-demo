# LangMem Agent Demo

A demonstration of LangGraph agents with persistent memory using LangMem, DeepSeek for chat, and Ollama for embeddings.

## Prerequisites

- **uv**: Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama**: For local embeddings ([install guide](https://ollama.ai/download))
- **DeepSeek API Key**: Get one from [DeepSeek Platform](https://platform.deepseek.com/api_keys)

## Setup

1. **Clone and navigate to project:**
   ```bash
   cd langmem-agent-demo
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Pull the embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

4. **Set up DeepSeek API key** (choose one method):
   
   - **Option A**: Add to `.env` file in project root:
     ```bash
     DEEPSEEK_API_KEY=your-api-key-here
     ```
   
   - **Option B**: Add to shell profile (`~/.bashrc` or `~/.zshrc`):
     ```bash
     export DEEPSEEK_API_KEY=your-api-key-here
     ```

## Run

```bash
uv run agent.py
```

The agent will demonstrate memory persistence across different conversation threads.

## Current Implementation

- **Chat Model**: DeepSeek (`deepseek-chat`)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Memory Store**: `InMemoryStore` (ephemeral)
- **Checkpointer**: `MemorySaver` (ephemeral)

## Project Goals

- [x] Basic agent with memory using `InMemoryStore`
- [ ] Replace `InMemoryStore` with Redis for persistent memory
- [ ] Investigate Valkey compatibility as Redis alternative

## Notes

- The deprecation warning for `create_react_agent` is expected and will be addressed in a future update
- `InMemoryStore` data is lost between restarts - Redis integration will fix this
