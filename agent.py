from typing import cast
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def prompt(state):
    """Prepare the messages for the LLM"""
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


# nomic-embed-text uses 768 dimensions
store = InMemoryStore(index={"dims": 768, "embed": "ollama:nomic-embed-text"})

checkpointer = MemorySaver()

agent = create_react_agent(
    ChatDeepSeek(model="deepseek-chat"),
    prompt=prompt,
    tools=[create_manage_memory_tool(namespace=("memories",))],
    store=store,
    checkpointer=checkpointer,
)


def main():
    """Demo an agent with memory."""
    config = cast(RunnableConfig, {"configurable": {"thread_id": "thread-a"}})

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

    print("=" * 60)
    print("Teaching the agent (thread-a):")
    print("=" * 60)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "dark. Remember that."}]},
        config=config,
    )

    print(f"Agent: {response['messages'][-1].content}\n")

    print("=" * 60)
    print("new conversation (thread-b)")
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
        config=new_config,
    )
    print(f"Agent: {response['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
