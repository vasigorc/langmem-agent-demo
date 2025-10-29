from typing import Any, Dict, List, cast
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.redis import RedisStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()

REDIS_URI = "redis://localhost:6379"


def prompt(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Prepare the messages for the LLM with relevant memories.

    Args:
        state: The current state containing messages

    Returns:
        List of messages with system prompt including memories
    """
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


def create_agent():
    # Initialize Redis store with vector search for embeddings
    with RedisStore.from_conn_string(
        REDIS_URI,
        index={
            "dims": 768,  # nomic-embed-text uses 768 dimensions
            "embed": "ollama:nomic-embed-text",
            "distance_type": "cosine",
        },
    ) as store:
        store.setup()

        # Initialize Redis checkpointer for conversation history
        with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
            checkpointer.setup()

            agent = create_react_agent(
                ChatDeepSeek(model="deepseek-chat"),
                prompt=prompt,
                tools=[create_manage_memory_tool(namespace=("memories",))],
                store=store,
                checkpointer=checkpointer,
            )

            return agent


def main() -> None:
    """Demo an agent with Redis-backed persistent memory."""
    print("\n🔴 Using Redis for persistent memory storage")
    print(f"📡 Connecting to Redis at {REDIS_URI}\n")

    try:
        agent = create_agent()

        config = cast(RunnableConfig, {"configurable": {"thread_id": "thread-a"}})

        # First interaction - agent doesn't know preferences yet
        print("=" * 60)
        print("First interaction (thread-a):")
        print("=" * 60)
        response = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Know which display mode I prefer?"}
                ]
            },
            config=config,
        )
        print(f"Agent: {response['messages'][-1].content}\n")

        # Tell the agent to remember something
        print("=" * 60)
        print("Teaching the agent (thread-a):")
        print("=" * 60)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "dark. Remember that."}]},
            config=config,
        )
        print(f"Agent: {response['messages'][-1].content}\n")

        # New thread - but the memory should persist!
        print("=" * 60)
        print("New conversation (thread-b):")
        print("=" * 60)
        new_config = cast(RunnableConfig, {"configurable": {"thread_id": "thread-b"}})
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hey there. Do you remember me? What are my preferences?",
                    }
                ]
            },
            new_config,
        )
        print(f"Agent: {response['messages'][-1].content}\n")

        print("=" * 60)
        print("✅ Memory persisted in Redis!")
        print("💡 Tip: Restart this script and the agent will remember!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("\n🔍 Troubleshooting:")
        print("  1. Is Redis running? Try: docker ps | grep redis")
        print("  2. Can you connect? Try: redis-cli ping")
        print("  3. Do you have the right modules? Redis 8.0+ includes them by default")
        raise


if __name__ == "__main__":
    main()

