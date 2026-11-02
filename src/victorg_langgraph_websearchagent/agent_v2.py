import getpass
import json
import operator
import os
from datetime import date
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

MODEL_NAME = "claude-haiku-4-5-20251001"
KNOWLEDGE_CUTOFF = "julio de 2025"  # training data cutoff de claude-haiku-4-5
SEARCH_TOOL_NAME = "tavily_search"
PRICE_INPUT_PER_MTOK = 1.0   # USD por millón de tokens de entrada (claude-haiku-4-5)
PRICE_OUTPUT_PER_MTOK = 5.0  # USD por millón de tokens de salida (claude-haiku-4-5)


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
# TAREA 1: system prompt con el corte de conocimiento
# ---------------------------------------------------------------------------
def build_system_prompt(use_search: bool = True) -> SystemMessage:
    """Prompt del sistema: conocimiento estático antes del corte, web después.

    Con use_search=False se usa el MISMO prompt, pero avisando de que en esa
    ejecución no existe la herramienta de búsqueda (sirve para comparar).
    """
    today = date.today().strftime("%d/%m/%Y")
    text = (
        f"Eres un asistente útil. Hoy es {today}.\n\n"
        f"CORTE DE CONOCIMIENTO: tu conocimiento llega hasta {KNOWLEDGE_CUTOFF}. "
        "No sabes nada fiable de lo ocurrido después, ni de información que cambia "
        "con el tiempo.\n\n"
        "REGLAS:\n"
        f"1. Preguntas sobre conocimiento anterior a {KNOWLEDGE_CUTOFF} o atemporal "
        "(historia, ciencia, definiciones, matemáticas, programación general): "
        f"responde con tu conocimiento estático. NO uses {SEARCH_TOOL_NAME}.\n"
        "2. Preguntas que requieren información actual o reciente (noticias, precios, "
        f"clima, versiones, resultados, 'último', 'actual', 'hoy') o hechos posteriores "
        f"a {KNOWLEDGE_CUTOFF}: usa {SEARCH_TOOL_NAME}. No respondas de memoria.\n"
        "3. Si la pregunta es sobre algo que no sabes o no puedes contestar con seguridad: "
        f"usa {SEARCH_TOOL_NAME}.\n"
        "4. Cuando uses la web, basa la respuesta en los resultados e indícalo.\n"
        "5. Para multiplicaciones usa la herramienta multiply.\n"
        "6. Responde en el mismo idioma del usuario."
    )
    if not use_search:
        text += (
            f"\n\nIMPORTANTE: en esta ejecución NO tienes la herramienta {SEARCH_TOOL_NAME}. "
            "Responde con lo que sepas e indica claramente si no puedes confirmar "
            "algo por no tener acceso a información actual."
        )
    return SystemMessage(content=text)


# ---------------------------------------------------------------------------
# Construir el agente (use_search=False => el mismo agente SIN búsqueda web)
# ---------------------------------------------------------------------------
def build_search_agent(use_search: bool = True):
    """Crea y compila el agente (modelo + herramientas + grafo)."""
    model = init_chat_model(MODEL_NAME, temperature=0)

    tools = [multiply]
    if use_search:
        tools.append(TavilySearch(max_results=5, topic="general"))

    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    system_prompt = build_system_prompt(use_search)

    def llm_call(state: dict):
        response = model_with_tools.invoke([system_prompt] + state["messages"])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    # JSON para poder recuperar y mostrar el contenido completo
                    content=json.dumps(observation, ensure_ascii=False, default=str),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder.compile()


# ---------------------------------------------------------------------------
# Ejecutar el agente (devuelve el estado completo para poder analizarlo)
# ---------------------------------------------------------------------------
def run_agent(agent, question: str, verbose: bool = False) -> dict:
    """Lanza el agente y devuelve el resultado completo (mensajes + contador)."""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    if verbose:
        for m in result["messages"]:
            m.pretty_print()

    return result


def get_answer(result: dict) -> str:
    """Devuelve solo el texto de la respuesta final."""
    final = result["messages"][-1].content
    if isinstance(final, list):  # a veces viene como lista de bloques
        final = "".join(b.get("text", "") for b in final if isinstance(b, dict))
    return final


def used_search(result: dict) -> bool:
    """True si el agente usó la búsqueda web en esa ejecución."""
    return any(
        isinstance(m, ToolMessage) and m.name == SEARCH_TOOL_NAME
        for m in result["messages"]
    )


# ---------------------------------------------------------------------------
# TAREA 2: tokens totales y tokens del tool call
# ---------------------------------------------------------------------------
def count_tokens(result: dict) -> dict:
    """Cuenta los tokens de una ejecución.

    - input / output / total: suma de todas las llamadas al modelo.
    - tool_call_tokens: tokens que el modelo genera para PEDIR la herramienta
      (output del mensaje con tool_calls).
    - tool_result_tokens: tokens que el resultado de la herramienta añade a la
      entrada de la llamada siguiente (input siguiente - input actual - output
      actual). Es una estimación muy ajustada.
    - tool_total: tool_call_tokens + tool_result_tokens.
    """
    ai_messages = [
        m for m in result["messages"] if isinstance(m, AIMessage) and m.usage_metadata
    ]

    calls = []  # una entrada por cada llamada al modelo
    for i, m in enumerate(ai_messages):
        usage = m.usage_metadata
        tool_names = [tc["name"] for tc in m.tool_calls]

        tool_call_tokens = usage["output_tokens"] if tool_names else 0
        tool_result_tokens = 0
        if tool_names and i + 1 < len(ai_messages):
            next_usage = ai_messages[i + 1].usage_metadata
            tool_result_tokens = max(
                next_usage["input_tokens"]
                - usage["input_tokens"]
                - usage["output_tokens"],
                0,
            )

        calls.append(
            {
                "llamada": i + 1,
                "input": usage["input_tokens"],
                "output": usage["output_tokens"],
                "total": usage["total_tokens"],
                "herramientas": tool_names,
                "tool_call_tokens": tool_call_tokens,
                "tool_result_tokens": tool_result_tokens,
            }
        )

    tool_call_tokens = sum(c["tool_call_tokens"] for c in calls)
    tool_result_tokens = sum(c["tool_result_tokens"] for c in calls)
    return {
        "llamadas": calls,
        "input": sum(c["input"] for c in calls),
        "output": sum(c["output"] for c in calls),
        "total": sum(c["total"] for c in calls),
        "tool_call_tokens": tool_call_tokens,
        "tool_result_tokens": tool_result_tokens,
        "tool_total": tool_call_tokens + tool_result_tokens,
    }


def print_tokens(result: dict):
    """Muestra el desglose de tokens de forma legible."""
    tokens = count_tokens(result)
    for c in tokens["llamadas"]:
        herr = ", ".join(c["herramientas"]) or "respuesta final"
        print(
            f"Llamada {c['llamada']} ({herr}): "
            f"input={c['input']} | output={c['output']} | total={c['total']}"
        )
    print(
        f"TOTAL: input={tokens['input']} | output={tokens['output']} "
        f"| total={tokens['total']}"
    )
    if tokens["tool_total"]:
        print(
            f"TOOL CALL: petición={tokens['tool_call_tokens']} | "
            f"resultado={tokens['tool_result_tokens']} | "
            f"total herramienta={tokens['tool_total']}"
        )
    else:
        print("TOOL CALL: no hubo llamada a herramientas")


# ---------------------------------------------------------------------------
# TAREA 3: inspeccionar los ToolMessage de forma legible
# ---------------------------------------------------------------------------
def get_tool_messages(result: dict) -> list[ToolMessage]:
    """Devuelve la lista de ToolMessage de la ejecución."""
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def print_tool_messages(result: dict, max_chars: int = 300):
    """Imprime cada ToolMessage con la petición que lo originó y su contenido."""
    tool_messages = get_tool_messages(result)
    if not tool_messages:
        print("No hay ToolMessages: el agente no usó ninguna herramienta.")
        return

    # argumentos con los que el modelo llamó a cada herramienta (por id)
    args_by_id = {
        tc["id"]: tc["args"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    }

    for i, tm in enumerate(tool_messages, start=1):
        print("=" * 70)
        print(f"ToolMessage {i}: {tm.name}")
        print(f"Argumentos: {args_by_id.get(tm.tool_call_id)}")
        print("-" * 70)

        try:
            data = json.loads(tm.content)
        except (json.JSONDecodeError, TypeError):
            data = tm.content

        if isinstance(data, dict) and "results" in data:  # resultado de Tavily
            print(f"Query: {data.get('query')}")
            if data.get("answer"):
                print(f"Respuesta Tavily: {data['answer']}")
            print(f"Tiempo de respuesta: {data.get('response_time')} s")
            print(f"Nº de resultados: {len(data['results'])}\n")
            for j, r in enumerate(data["results"], start=1):
                content = str(r.get("content", "")).replace("\n", " ")
                print(f"[{j}] {r.get('title')}")
                print(f"    URL:   {r.get('url')}")
                print(f"    Score: {r.get('score')}")
                print(f"    Texto: {content[:max_chars]}{'...' if len(content) > max_chars else ''}\n")
        else:  # otras herramientas (p. ej. multiply)
            print(f"Resultado: {data}")


# ---------------------------------------------------------------------------
# PASO 1: peso de los ToolMessage (NO los tool_calls) en los tokens
# ---------------------------------------------------------------------------
def toolmessage_tokens(result: dict) -> dict:
    """Calcula cuántos tokens aportan los ToolMessage a la entrada del modelo.

    Idea: la API no da los tokens de cada mensaje, solo el input_tokens de cada
    llamada al modelo. Pero en cada llamada el contexto es el anterior MÁS lo nuevo:

        llamada 1 (input I1): system + schemas de las tools + pregunta
                  -> el modelo genera el tool_call (output O1)
        llamada 2 (input I2): system + schemas + pregunta + tool_call + ToolMessage

    Lo común (system, schemas, pregunta) se cancela, así que:

        tokens ToolMessage = I2 - I1 - O1

    O1 se resta porque el tool_call, tras generarlo el modelo, vuelve a entrar como
    input en la llamada 2. Lo que queda es el ToolMessage (resultado de la tool).
    """
    ai_messages = []  # mensajes del modelo con usage_metadata (uno por llamada)
    tool_messages_by_call = {}  # ToolMessages que siguen a cada llamada del modelo
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.usage_metadata:
            ai_messages.append(m)
        elif isinstance(m, ToolMessage) and ai_messages:
            tool_messages_by_call.setdefault(len(ai_messages) - 1, []).append(m)

    rounds = []  # una entrada por ronda de herramientas
    for i, m in enumerate(ai_messages[:-1]):
        if not m.tool_calls:
            continue
        usage = m.usage_metadata
        next_usage = ai_messages[i + 1].usage_metadata
        tokens = max(
            next_usage["input_tokens"] - usage["input_tokens"] - usage["output_tokens"], 0
        )
        later_calls = len(ai_messages) - (i + 1)  # llamadas que arrastran el ToolMessage
        rounds.append(
            {
                "ronda": len(rounds) + 1,
                "herramientas": [tc["name"] for tc in m.tool_calls],
                "input_antes (I1)": usage["input_tokens"],
                "output_tool_call (O1)": usage["output_tokens"],
                "input_despues (I2)": next_usage["input_tokens"],
                "toolmessage_tokens": tokens,
                "peso_en_input_siguiente": tokens / next_usage["input_tokens"],
                "llamadas_que_lo_arrastran": later_calls,
                "input_facturado": tokens * later_calls,
                "toolmessages": tool_messages_by_call.get(i, []),
            }
        )

    total_input = sum(m.usage_metadata["input_tokens"] for m in ai_messages)
    total_tokens = sum(m.usage_metadata["total_tokens"] for m in ai_messages)
    billed = sum(r["input_facturado"] for r in rounds)
    return {
        "rondas": rounds,
        "toolmessage_tokens": sum(r["toolmessage_tokens"] for r in rounds),
        "input_facturado": billed,
        "peso_en_input_total": billed / total_input if total_input else 0.0,
        "peso_en_total": billed / total_tokens if total_tokens else 0.0,
        "coste_usd": billed * PRICE_INPUT_PER_MTOK / 1_000_000,
    }


def print_toolmessage_weight(result: dict):
    """Muestra el peso de los ToolMessage con la cuenta paso a paso."""
    info = toolmessage_tokens(result)
    if not info["rondas"]:
        print("No hubo ToolMessages: el agente no usó ninguna herramienta.")
        return

    for r in info["rondas"]:
        print(f"Ronda {r['ronda']} ({', '.join(r['herramientas'])})")
        print(f"  I1 (input antes del tool_call)      = {r['input_antes (I1)']}")
        print(f"  O1 (output: el tool_call)           = {r['output_tool_call (O1)']}")
        print(f"  I2 (input con el ToolMessage)       = {r['input_despues (I2)']}")
        print(
            f"  ToolMessage = I2 - I1 - O1          = {r['toolmessage_tokens']} tokens "
            f"({r['peso_en_input_siguiente']:.0%} del input de la llamada siguiente)"
        )
        print(
            f"  Llamadas que lo arrastran: {r['llamadas_que_lo_arrastran']} "
            f"-> input facturado por el ToolMessage = {r['input_facturado']}"
        )
    print(
        f"TOTAL ToolMessages: {info['toolmessage_tokens']} tokens | "
        f"{info['peso_en_input_total']:.0%} del input total | "
        f"{info['peso_en_total']:.0%} de los tokens totales | "
        f"coste ~ ${info['coste_usd']:.5f}"
    )


def count_text_tokens(text: str):
    """Cuenta los tokens de un texto con el endpoint count_tokens de Anthropic.

    Devuelve None si no se puede (sin clave, sin red...). Se resta lo que cuesta
    un mensaje mínimo ("a" = 1 token) para quedarse solo con el texto.
    """
    try:
        import anthropic

        client = anthropic.Anthropic()
        base = client.messages.count_tokens(
            model=MODEL_NAME, messages=[{"role": "user", "content": "a"}]
        ).input_tokens
        n = client.messages.count_tokens(
            model=MODEL_NAME, messages=[{"role": "user", "content": text}]
        ).input_tokens
        return n - (base - 1)
    except Exception:
        return None


def verify_toolmessage_tokens(result: dict):
    """Contrasta el cálculo I2 - I1 - O1 con el recuento directo del texto del ToolMessage."""
    for r in toolmessage_tokens(result)["rondas"]:
        direct = [count_text_tokens(tm.content) for tm in r["toolmessages"]]
        if None in direct:
            print(f"Ronda {r['ronda']}: no se pudo usar count_tokens (¿clave o red?).")
            continue
        direct_total = sum(direct)
        diff = r["toolmessage_tokens"] - direct_total
        print(
            f"Ronda {r['ronda']}: por diferencia (I2-I1-O1) = {r['toolmessage_tokens']} | "
            f"texto del ToolMessage con count_tokens = {direct_total} | diferencia = {diff}"
        )
    print("(la diferencia son los tokens de la estructura tool_use/tool_result, pocos y constantes)")


def print_weight_summary(queries: list[dict], results: list[dict]):
    """Tabla: tokens de ToolMessage y su peso en cada query que usó herramientas."""
    print(f"{'#':<3} {'Query':<46} {'TM tok':>7} {'Facturado':>10} {'% input':>8} {'USD':>9}")
    for i, (q, r) in enumerate(zip(queries, results), start=1):
        info = toolmessage_tokens(r)
        if not info["rondas"]:
            continue
        text = q["query"] if len(q["query"]) <= 44 else q["query"][:41] + "..."
        print(
            f"{i:<3} {text:<46} {info['toolmessage_tokens']:>7} {info['input_facturado']:>10} "
            f"{info['peso_en_input_total']:>8.0%} {info['coste_usd']:>9.5f}"
        )


# ---------------------------------------------------------------------------
# Utilidades para el notebook de pruebas
# ---------------------------------------------------------------------------
def print_summary(queries: list[dict], results: list[dict]):
    """Tabla: query | esperaba web | usó web | tokens totales | tokens herramienta."""
    print(f"{'#':<3} {'Query':<52} {'Esperado':<9} {'Usó web':<8} {'Tokens':>7} {'Tool':>6}  OK")
    for i, (q, r) in enumerate(zip(queries, results), start=1):
        used = used_search(r)
        tokens = count_tokens(r)
        ok = "✓" if used == q["expected_search"] else "✗"
        text = q["query"] if len(q["query"]) <= 50 else q["query"][:47] + "..."
        print(
            f"{i:<3} {text:<52} {'web' if q['expected_search'] else 'estático':<9} "
            f"{'sí' if used else 'no':<8} {tokens['total']:>7} {tokens['tool_total']:>6}  {ok}"
        )


def compare_agents(question: str, agent_search, agent_plain):
    """Responde la misma pregunta con y sin búsqueda web para ver si fue justificada."""
    with_search = run_agent(agent_search, question)
    without_search = run_agent(agent_plain, question)

    print(f"PREGUNTA: {question}\n")
    print(f"--- CON búsqueda web (usó web: {used_search(with_search)}) ---")
    print(get_answer(with_search))
    print(f"\n--- SIN búsqueda web ---")
    print(get_answer(without_search))
    return with_search, without_search


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_keys()
    agent = build_search_agent()

    result = run_agent(agent, "¿Cuál es la última versión estable de LangGraph?")
    print("\nRESPUESTA FINAL:\n", get_answer(result))
    print("\nTOKENS:")
    print_tokens(result)
    print("\nTOOL MESSAGES:")
    print_tool_messages(result)
