import operator
import os
import getpass
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Claves
# ---------------------------------------------------------------------------
def load_keys():
    """Carga las claves del .env (el .env manda) y las pide si faltan."""
    load_dotenv(find_dotenv(), override=True)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key.startswith("sk-ant-"):
        anthropic_key = getpass.getpass("Anthropic API key (sk-ant-...): ").strip()
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n").strip()


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


# ---------------------------------------------------------------------------
# FUNCIÓN 1: construir el agente con la search tool
# ---------------------------------------------------------------------------
def build_search_agent():
    """Crea y compila el agente (modelo + herramientas + grafo)."""
    model = init_chat_model("claude-haiku-4-5-20251001", temperature=0)

    search_tool = TavilySearch(max_results=5, topic="general")

    tools = [multiply, search_tool]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. Use the search tool to look up "
            "up-to-date information on the web, and the multiply tool for "
            "multiplications. Answer in the same language as the user."
        )
    )

    # Nodo: el modelo decide si responde o pide una herramienta
    def llm_call(state: dict):
        response = model_with_tools.invoke([system_prompt] + state["messages"])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Nodo: ejecuta las herramientas que pidió el modelo
    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        return {"messages": result}

    # Decisión: ¿seguimos con una herramienta o terminamos?
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    # Grafo: START -> llm_call -> (tool_node -> llm_call) o END
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder.compile()


# ---------------------------------------------------------------------------
# FUNCIÓN 2: ejecutar el agente
# ---------------------------------------------------------------------------
def run_agent(agent, question: str, verbose: bool = True) -> str:
    """Lanza el agente con una pregunta y devuelve la respuesta final."""
    messages = [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})

    if verbose:
        for m in result["messages"]:
            m.pretty_print()

    # La respuesta final es el contenido del último mensaje
    final = result["messages"][-1].content
    if isinstance(final, list):  # a veces viene como lista de bloques
        final = "".join(b.get("text", "") for b in final if isinstance(b, dict))
    return final


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_keys()
    agent = build_search_agent()

    respuesta = run_agent(agent, "¿Qué es LangGraph? Búscalo en la web.")
    print("\nRESPUESTA FINAL:\n", respuesta)
