from typing import Sequence
from langchain_core.prompt_values import PromptValue
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

model = init_chat_model("phi3", model_provider="ollama")

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm Zedro"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def main():
    # Define a new graph
    workflow = StateGraph(state_schema=State)
    print(model)

    # Define the function that calls the model
    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}

    query = "What's my name?"
    language = "Portugese"
    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language}, 
        config
    )
    output["messages"][-1].pretty_print()  # output contains all messages in state

    return 0


def call_model(state: State) -> dict:
    """
    Call the model

    :param state: The state of the graph
    :type state: MessagesState
    :return: Dictionary with messages
    :rtype: dict{messages: Sequence[BaseMessage]}
    """
    trimmed_messages: Sequence[BaseMessage] = trimmer.invoke(state["messages"])
    prompt: PromptValue = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response: BaseMessage = model.invoke(prompt)
    return {"messages": response}


if __name__ == "__main__":
    main()
