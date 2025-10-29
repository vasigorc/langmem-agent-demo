# LangMem Agent Demo

A demonstration of LangGraph agents with persistent memory using LangMem, DeepSeek for chat, and Ollama for embeddings. Includes both in-memory and Redis-backed implementations.

## Prerequisites

- **uv**: Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama**: For local embeddings ([install guide](https://ollama.ai/download))
- **Redis**: For persistent memory (Docker or local install)
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

5. **Start Redis** (for persistent memory):

   ```bash
   docker run -d --name langmem-redis -p 6379:6379 redis:latest
   ```

## Run

### Option 1: In-Memory (Ephemeral)

```bash
uv run agent.py
```

Memory is lost when the script exits.

### Option 2: Redis (Persistent)

```bash
uv run agent_redis.py
```

Running this script twice demonstrates the usage of persistent memory by the agent that we wrote:

```bash
# Example first run:
uv run agent_redis.py

🔴 Using Redis for persistent memory storage
📡 Connecting to Redis at redis://localhost:6379

11:22:48 langgraph.store.redis INFO   Redis standalone client detected for RedisStore.
11:22:48 langgraph.checkpoint.redis INFO   Redis client is a standalone client
/Users/vasilegorcinschi/persorepo/langmem-agent-demo/agent_redis.py:61: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
  agent = create_react_agent(
============================================================
First interaction (thread-a):
============================================================
11:22:50 httpx INFO   HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
11:22:51 httpx INFO   HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
Agent: I don't currently have any information stored about your display mode preferences. You haven't told me your preferred display settings yet.

Would you like to share your display mode preference with me? For example, you might prefer:
- Light mode
- Dark mode
- Auto/adaptive mode
- High contrast mode
- Specific color schemes
...
============================================================
✅ Memory persisted in Redis!
💡 Tip: Restart this script and the agent will remember!
============================================================

# Second run of the same script (Redis still running)
uv run agent_redis.py
...
============================================================
First interaction (thread-a):
============================================================
11:27:16 httpx INFO   HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
11:27:17 httpx INFO   HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
Agent: Yes! I remember that you prefer **dark display mode**. I saved this preference in my memory, so I'll always know you like dark mode for our conversations.
...
```

### View Redis Memories

```bash
# View all stored memories and checkpoints
redis-cli
127.0.0.1:6379> KEYS *
 1) "write_keys_zset:thread-a::1f0b4db2-460b-6aa2-8002-85cd57f7f991"
 2) "checkpoint:thread-a:__empty__:1f0b4db2-5f05-6abc-8004-9131a8788fd0"
 3) "write_keys_zset:thread-a::1f0b4db2-5fcb-6ea6-8005-970e4262831b"
 4) "checkpoint_write:thread-b:__empty__:1f0b4db2-788c-61d4-bfff-f173bc7b11ef:435b2051-dd4c-1e96-1504-f1d77296d22b:0"
 5) "write_keys_zset:thread-a::1f0b4db2-1222-66e6-bfff-f63de648e026"
 6) "checkpoint:thread-a:__empty__:1f0b4db2-5fcb-6ea6-8005-970e4262831b"
 7) "write_keys_zset:thread-a::1f0b4db2-1225-6a9e-8000-1592e4e218a1"
 8) "store_vectors:01K8R919NNY9M8WNF6CP8B1A30"
 9) "checkpoint_write:thread-a:__empty__:1f0b4db2-460d-60a0-8003-3b891
 ...

# Check the contents for a specific key
redis-cli JSON.GET "checkpoint:thread-a:__empty__:1f0b4db2-5fcb-6ea6-8005-970e4262831b" | jq .
{
  "thread_id": "thread-a",
  "checkpoint_ns": "__empty__",
  "checkpoint_id": "1f0b4db2-5fcb-6ea6-8005-970e4262831b",
  "parent_checkpoint_id": "1f0b4db2-5f05-6abc-8004-9131a8788fd0",
  "checkpoint_ts": 1761751377665.546,
  "checkpoint": {
    "type": "json",
    "v": 4,
    "ts": "2025-10-29T15:22:57.664353+00:00",
    "id": "1f0b4db2-5fcb-6ea6-8005-970e4262831b",
    "channel_values": {
      "messages": [
        {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "HumanMessage"
          ],
          "kwargs": {
            "content": "Know which display mode I prefer?",
            "type": "human",
            "id": "9b167956-3362-4556-a4e4-8ce32b444e3b"
          }
        },
        {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "I don't currently have any information stored about your display mode preferences. You haven't told me your preferred display settings yet.\n\nWould you like to share your display mode preference with me? For example, you might prefer:\n- Light mode\n- Dark mode  \n- Auto/adaptive mode\n- High contrast mode\n- Specific color schemes\n\nOnce you tell me your preference, I can remember it for future conversations!",
            "additional_kwargs": {
              ...
```

## Architecture

### In-Memory Implementation (`agent.py`)

- **Chat Model**: DeepSeek (`deepseek-chat`)
- **Embeddings**: Ollama (`nomic-embed-text`, 768 dimensions)
- **Memory Store**: `InMemoryStore` (ephemeral)
- **Checkpointer**: `MemorySaver` (ephemeral)

### Redis Implementation (`agent_redis.py`)

- **Chat Model**: DeepSeek (`deepseek-chat`)
- **Embeddings**: Ollama (`nomic-embed-text`, 768 dimensions)
- **Memory Store**: `RedisStore` (persistent, with vector search)
- **Checkpointer**: `RedisSaver` (persistent)
- **Connection**: `redis://localhost:6379`

## Project Goals

- [x] Basic agent with memory using `InMemoryStore`
- [x] Replace `InMemoryStore` with Redis for persistent memory
- [ ] Investigate Valkey compatibility as Redis alternative

## Redis Integration Details

The Redis implementation uses:

- **`langgraph-checkpoint-redis`**: Official Redis integration for LangGraph
- **RedisJSON**: For storing structured data
- **RediSearch**: For vector search capabilities (included in Redis 8.0+)

Memory is stored in two types of keys:

- `store_vectors:..`: Long-term memories (cross-thread)
- `checkpoint:<thread_id>:...`: Conversation state (thread-specific)

## Notes

- The deprecation warning for `create_react_agent` is expected and will be addressed in a future update
- NumPy is recommended for better vector operation performance
- Redis Stack or Redis 8.0+ is required for full functionality (includes RedisJSON and RediSearch modules)
