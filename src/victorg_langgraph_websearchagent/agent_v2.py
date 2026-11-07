import getpass
import json
import operator
import os
import re
import unicodedata
from datetime import date
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, Field

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


DEFAULT_SEARCH_KWARGS = {"max_results": 5, "topic": "general"}


def build_tools(use_search: bool = True, search_kwargs: dict | None = None) -> list:
    """Lista de herramientas del agente. search_kwargs configura TavilySearch."""
    tools = [multiply]
    if use_search:
        kwargs = DEFAULT_SEARCH_KWARGS if search_kwargs is None else search_kwargs
        tools.append(TavilySearch(**kwargs))
    return tools


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
def build_search_agent(use_search: bool = True, search_kwargs: dict | None = None):
    """Crea y compila el agente (modelo + herramientas + grafo)."""
    model = init_chat_model(MODEL_NAME, temperature=0)

    tools = build_tools(use_search, search_kwargs)

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
            try:
                observation = selected_tool.invoke(tool_call["args"])
            except Exception as e:  # p. ej. Tavily sin resultados (ToolException)
                observation = {"error": str(e)}
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
# PASO 2: entender y evaluar la search tool
# ---------------------------------------------------------------------------
# --- 2.1 Estructura de lo que devuelve Tavily -------------------------------
def get_tavily_outputs(result: dict) -> list[dict]:
    """Salidas de Tavily (como diccionarios) de una ejecución, una por búsqueda."""
    outputs = []
    for tm in get_tool_messages(result):
        if tm.name == SEARCH_TOOL_NAME:
            try:
                outputs.append(json.loads(tm.content))
            except (json.JSONDecodeError, TypeError):
                outputs.append({"error": tm.content})
    return outputs


def print_tavily_structure(output: dict):
    """Estructura de una salida de Tavily y dónde se va su tamaño (orientativo)."""
    results = output.get("results", [])
    total_chars = len(json.dumps(output, ensure_ascii=False, default=str))

    print("Claves de nivel superior:", list(output.keys()))
    print(f"Nº de resultados: {len(results)} | response_time: {output.get('response_time')}")
    if results:
        print("Campos de cada resultado:", list(results[0].keys()))
    print(f"Tamaño del JSON: {total_chars} caracteres (~{total_chars // 4} tokens, orientativo)\n")

    chars = {}  # caracteres acumulados por campo de resultado
    for r in results:
        for key, value in r.items():
            chars[key] = chars.get(key, 0) + len(str(value))
    print("Peso por campo (suma de todos los resultados):")
    for key, c in sorted(chars.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<12} {c:>6} caracteres  ({c / total_chars:.0%} del JSON)")

    print("\nResultados (ordenados como los devuelve Tavily):")
    for i, r in enumerate(results, start=1):
        content = r.get("content") or ""
        print(f"  {i}. score={r.get('score')} | content={len(content)} caracteres | {r.get('title')}")
        print(f"     {r.get('url')}")


# --- 2.2 Qué ve el LLM con bind_tools() --------------------------------------
LLM_OVERRIDABLE = [
    "include_domains", "exclude_domains", "search_depth",
    "include_images", "time_range", "topic",
]  # el LLM los ve, pero si se fijan al crear la tool, el valor del LLM se ignora
INIT_ONLY = [
    "max_results", "include_answer", "include_raw_content", "include_image_descriptions",
    "country", "include_favicon", "include_usage", "exact_match", "auto_parameters",
]  # solo se pueden fijar al crear la tool; el LLM no los ve


def tool_schemas(tools) -> list[dict]:
    """Schemas en el formato exacto que Anthropic recibe tras bind_tools()."""
    try:
        from langchain_anthropic.chat_models import convert_to_anthropic_tool

        return [dict(convert_to_anthropic_tool(t)) for t in tools]
    except ImportError:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.tool_call_schema.model_json_schema(),
            }
            for t in tools
        ]


def print_tool_schemas(tools):
    """Muestra qué ve el LLM de cada herramienta: nombre, descripción y parámetros."""
    for schema in tool_schemas(tools):
        props = schema["input_schema"].get("properties", {})
        required = schema["input_schema"].get("required", [])
        print("=" * 70)
        print(f"Herramienta: {schema['name']}")
        print(f"Descripción: {len(schema['description'])} caracteres")
        print(f"Parámetros que ve el LLM: {len(props)}")
        for name, prop in props.items():
            if "anyOf" in prop:
                types = [t.get("type", "?") for t in prop["anyOf"] if t.get("type") != "null"]
                type_ = "|".join(types)
            else:
                type_ = prop.get("type", "?")
            req = "obligatorio" if name in required else "opcional"
            print(f"  - {name:<16} {type_:<14} {req:<12} descripción: {len(prop.get('description', ''))} caracteres")
        print(f"JSON del schema: {len(json.dumps(schema))} caracteres")


def print_tavily_parameters(search_tool):
    """Qué parámetros decide el LLM y cuáles están fijados al crear la tool."""
    visible = list(search_tool.args_schema.model_fields.keys())
    print("Parámetros que VE el LLM:")
    for name in visible:
        fixed = getattr(search_tool, name, None) if name in LLM_OVERRIDABLE else None
        if name in LLM_OVERRIDABLE and fixed:
            print(f"  - {name:<16} fijado al crear la tool = {fixed!r} -> el valor del LLM se IGNORA")
        elif name == "query":
            print(f"  - {name:<16} lo escribe el LLM")
        else:
            print(f"  - {name:<16} lo decide el LLM en cada llamada")
    print("\nParámetros que NO ve el LLM (solo al crear la tool):")
    for name in INIT_ONLY:
        print(f"  - {name:<28} = {getattr(search_tool, name, None)!r}")


def count_tool_schema_tokens(tools):
    """Tokens que añaden los schemas de las tools a CADA llamada (count_tokens de Anthropic).

    total = tokens con las tools - tokens sin ellas.
    por_tool = aportación marginal de cada una (con todas - con todas menos esa).
    fijo = lo que Anthropic añade por usar tools, aunque sea una sola.
    Devuelve None si no se puede (sin clave o red).
    """
    try:
        import anthropic

        client = anthropic.Anthropic()
        schemas = tool_schemas(tools)
        messages = [{"role": "user", "content": "a"}]

        def count(tool_list):
            extra = {"tools": tool_list} if tool_list else {}
            return client.messages.count_tokens(
                model=MODEL_NAME, messages=messages, **extra
            ).input_tokens

        base = count([])
        with_all = count(schemas)
        per_tool = {}
        for k, schema in enumerate(schemas):
            others = schemas[:k] + schemas[k + 1:]
            per_tool[schema["name"]] = with_all - count(others)
        total = with_all - base
        return {"total": total, "por_tool": per_tool, "fijo": total - sum(per_tool.values())}
    except Exception:
        return None


def print_tool_schema_tokens(tools):
    """Muestra los tokens de los schemas de las tools en cada llamada al modelo."""
    info = count_tool_schema_tokens(tools)
    if info is None:
        print("No se pudo usar count_tokens (¿clave o red?).")
        return
    for name, n in info["por_tool"].items():
        print(f"  {name:<16} {n:>5} tokens")
    print(f"  {'fijo (uso de tools)':<20} {info['fijo']:>5} tokens")
    print(f"TOTAL schemas de tools: {info['total']} tokens, en CADA llamada al modelo")


def first_call_input(result: dict) -> int:
    """input_tokens de la primera llamada al modelo (system + schemas + pregunta)."""
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.usage_metadata:
            return m.usage_metadata["input_tokens"]
    return 0


# --- 2.3 / 2.4 Snippets: cómo se generan y si son la parte relevante ---------
_STOPWORDS = {
    "cual", "cuales", "como", "cuando", "donde", "quien", "quienes", "para", "porque",
    "sobre", "entre", "desde", "hasta", "este", "esta", "esto", "estos", "estas",
    "ultima", "ultimo", "actual", "dime", "hoy", "what", "which", "when", "where",
    "who", "how", "the", "and", "for", "that", "with", "from", "this",
}


def _normalize(text: str) -> str:
    """Minúsculas, sin acentos y con espacios normalizados."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(text.lower().split())


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"\w+", _normalize(query))
    return sorted({t for t in terms if len(t) > 3 and t not in _STOPWORDS})


def _term_coverage(text: str, terms: list[str]) -> float:
    """Fracción de términos de la query que aparecen en el texto."""
    if not terms:
        return 0.0
    norm = _normalize(text)
    return sum(t in norm for t in terms) / len(terms)


def _best_window_coverage(text: str, terms: list[str], window: int = 500) -> tuple[float, float]:
    """Mejor cobertura de términos en una ventana de `window` caracteres de la página.

    Devuelve (cobertura, posición relativa de la ventana: 0 = inicio, 1 = final).
    """
    norm = _normalize(text)
    if not norm:
        return 0.0, 0.0
    step = max(window // 2, 1)
    best, best_pos = 0.0, 0.0
    for start in range(0, max(len(norm) - window, 0) + 1, step):
        cov = sum(t in norm[start:start + window] for t in terms) / len(terms) if terms else 0.0
        if cov > best:
            best, best_pos = cov, start / max(len(norm), 1)
    return best, best_pos


def analyze_snippets(query: str, output: dict, window: int = 500) -> list[dict]:
    """Compara el snippet (content) de cada resultado con la página completa (raw_content).

    Necesita que la tool se creó con include_raw_content=True.
    """
    terms = _query_terms(query)
    report = []
    for r in output.get("results", []):
        content = r.get("content") or ""
        page = r.get("raw_content") or ""
        page_norm = " ".join(page.split())
        chunks = [c.strip() for c in content.split("[...]") if c.strip()]

        positions = []  # dónde está cada chunk dentro de la página (0 = inicio, 1 = final)
        for chunk in chunks:
            pos = page_norm.find(" ".join(chunk.split()))
            if pos >= 0:
                positions.append(pos / max(len(page_norm), 1))

        best_cov, best_pos = _best_window_coverage(page, terms, window)
        report.append(
            {
                "titulo": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "n_chunks": len(chunks),
                "chars_snippet": len(content),
                "chars_pagina": len(page),
                "pct_pagina": len(content) / len(page) if page else None,
                "chunks_literales": len(positions) / len(chunks) if chunks else None,
                "posicion_media_en_pagina": sum(positions) / len(positions) if positions else None,
                "cobertura_snippet": _term_coverage(content, terms),
                "cobertura_mejor_ventana": best_cov,
                "posicion_mejor_ventana": best_pos,
                "terminos_query": terms,
            }
        )
    return report


def inspect_snippets(query: str, search_kwargs: dict | None = None, window: int = 500) -> list[dict]:
    """Lanza la búsqueda pidiendo también la página completa y analiza los snippets."""
    kwargs = dict(DEFAULT_SEARCH_KWARGS if search_kwargs is None else search_kwargs)
    kwargs["include_raw_content"] = True
    output = TavilySearch(**kwargs).invoke({"query": query})
    return analyze_snippets(query, output, window)


def print_snippet_report(query: str, report: list[dict]):
    """Tabla legible del análisis de snippets de una query."""
    print(f"Query: {query}")
    if report:
        print(f"Términos de la query usados: {report[0]['terminos_query']}")
    print(f"{'#':<3} {'score':>6} {'snippet':>8} {'página':>8} {'%pág':>6} {'literal':>8} {'pos.':>5} {'cob.snip':>9} {'cob.mejor':>10}  título")
    for i, r in enumerate(report, start=1):
        pct = f"{r['pct_pagina']:.0%}" if r["pct_pagina"] is not None else "-"
        lit = f"{r['chunks_literales']:.0%}" if r["chunks_literales"] is not None else "-"
        pos = f"{r['posicion_media_en_pagina']:.2f}" if r["posicion_media_en_pagina"] is not None else "-"
        flag = " <-- hay un pasaje mejor" if r["cobertura_mejor_ventana"] > r["cobertura_snippet"] else ""
        title = (r["titulo"] or "")[:40]
        print(
            f"{i:<3} {r['score']:>6.2f} {r['chars_snippet']:>8} {r['chars_pagina']:>8} {pct:>6} "
            f"{lit:>8} {pos:>5} {r['cobertura_snippet']:>9.0%} {r['cobertura_mejor_ventana']:>10.0%}  {title}{flag}"
        )
    print("literal = % de chunks que aparecen tal cual en la página | pos. = 0 inicio, 1 final")
    print("cob.snip / cob.mejor = % de términos de la query en el snippet / en el mejor pasaje de la página")


# --- 2.5 Score vs relevancia (juez LLM con schema pydantic) ---------------------
class RelevanceJudgement(BaseModel):
    """Valoración de la relevancia de un resultado de búsqueda."""

    relevancia: int = Field(
        description="0 = irrelevante, 1 = poco relevante, 2 = relevante, 3 = responde directamente a la consulta"
    )
    explicacion: str = Field(description="Explicación breve (1-2 frases) de la puntuación")


def judge_results_relevance(query: str, output: dict, judge_model=None) -> list[dict]:
    """Un LLM puntúa (0-3) la relevancia de cada resultado para la consulta."""
    model = judge_model or init_chat_model(MODEL_NAME, temperature=0)
    judge = model.with_structured_output(RelevanceJudgement)

    judged = []
    for r in output.get("results", []):
        prompt = (
            "Valora si el fragmento devuelto por un buscador es relevante para la consulta.\n\n"
            f"Consulta: {query}\n\n"
            f"Título: {r.get('title')}\n"
            f"Fragmento:\n{r.get('content')}"
        )
        verdict = judge.invoke(prompt)
        judged.append(
            {
                "titulo": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "relevancia": verdict.relevancia,
                "explicacion": verdict.explicacion,
            }
        )
    return judged


def pairwise_agreement(scores: list[float], relevances: list[int]):
    """Fracción de pares (con score y relevancia distintos) en los que el orden coincide.

    1.0 = el score ordena igual que el juez; 0.5 = azar; None si no hay pares comparables.
    """
    concordant = discordant = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] == scores[j] or relevances[i] == relevances[j]:
                continue
            if (scores[i] - scores[j]) * (relevances[i] - relevances[j]) > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total else None


def print_score_vs_relevance(judged_by_query: dict[str, list[dict]]):
    """¿El score de Tavily coincide con la relevancia juzgada por el LLM?"""
    all_pairs_scores, all_pairs_rels = [], []
    relevant_scores, irrelevant_scores = [], []

    for query, judged in judged_by_query.items():
        print(f"\nQuery: {query}")
        print(f"  {'score':>6} {'relev.':>6}  título / explicación")
        for j in judged:
            print(f"  {j['score']:>6.2f} {j['relevancia']:>6}  {(j['titulo'] or '')[:50]}")
            print(f"{'':>16}{j['explicacion'][:110]}")
            (relevant_scores if j["relevancia"] >= 2 else irrelevant_scores).append(j["score"])

        scores = [j["score"] for j in judged]
        rels = [j["relevancia"] for j in judged]
        agree = pairwise_agreement(scores, rels)
        print(f"  Coincidencia orden score/relevancia en esta query: {'-' if agree is None else f'{agree:.0%}'}")

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    print("\nRESUMEN")
    print(f"  Score medio de resultados relevantes (>=2):   {mean(relevant_scores):.2f}  (n={len(relevant_scores)})")
    print(f"  Score medio de resultados poco relevantes (<=1): {mean(irrelevant_scores):.2f}  (n={len(irrelevant_scores)})")


# --- 2.6 ¿El modelo usa los resultados o alucina? --------------------------------
class GroundednessJudgement(BaseModel):
    """Valoración de si una respuesta está respaldada por los resultados de búsqueda."""

    respaldada: bool = Field(
        description="True si TODAS las afirmaciones factuales de la respuesta están respaldadas por las fuentes"
    )
    afirmaciones_sin_respaldo: list[str] = Field(
        description="Afirmaciones de la respuesta que NO aparecen en las fuentes (lista vacía si no hay)"
    )
    explicacion: str = Field(description="Explicación breve (1-2 frases)")


def sources_text(outputs: list[dict], extra_sources: list[str] | None = None) -> str:
    """Texto de todas las fuentes que recibió el modelo.

    Incluye los snippets de Tavily y, en extra_sources, los resultados de las demás
    herramientas (p. ej. multiply): la respuesta también puede apoyarse en ellos.
    """
    parts = []
    for output in outputs:
        for i, r in enumerate(output.get("results", []), start=1):
            parts.append(f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content')}")
    parts.extend(extra_sources or [])
    return "\n\n".join(parts)


def tool_sources(result: dict) -> list[str]:
    """Resultados de las herramientas que NO son la búsqueda web (p. ej. multiply)."""
    args_by_id = {
        tc["id"]: tc["args"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    }
    return [
        f"Resultado de la herramienta {tm.name} con argumentos {args_by_id.get(tm.tool_call_id)}: {tm.content}"
        for tm in get_tool_messages(result)
        if tm.name != SEARCH_TOOL_NAME
    ]


def _number_values(token: str) -> set:
    """Valores numéricos posibles de un texto como '126,210.50' o '1.234,5'.

    Admite coma o punto como separador de miles o decimal. Si es ambiguo
    ('82.707'), devuelve las dos lecturas.
    """
    t = token.strip()
    candidates = []
    try:
        if "." in t and "," in t:
            if t.rfind(",") > t.rfind("."):  # 1.234,56
                candidates.append(t.replace(".", "").replace(",", "."))
            else:  # 1,234.56
                candidates.append(t.replace(",", ""))
        elif "," in t:
            if re.fullmatch(r"\d{1,3}(,\d{3})+", t):  # 82,707 -> miles
                candidates += [t.replace(",", ""), t.replace(",", ".") if t.count(",") == 1 else t.replace(",", "")]
            else:
                candidates.append(t.replace(",", "."))
        elif "." in t:
            if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):  # 82.707 -> ¿miles o decimal?
                candidates += [t.replace(".", ""), t]
            else:
                candidates.append(t)
        else:
            candidates.append(t)
        return {round(float(c), 6) for c in candidates}
    except ValueError:
        return set()


def unsupported_numbers(answer: str, sources: str) -> list[str]:
    """Números de la respuesta (de 2+ caracteres) que no aparecen en las fuentes.

    Compara por VALOR, no por texto: '126,210.50' y '126210.5' son el mismo número.
    Limitación: no entiende abreviaturas como '126K' ni cifras escritas con letras.
    """
    pattern = r"\d+(?:[.,]\d+)*"
    source_values = set()
    for token in re.findall(pattern, sources):
        source_values |= _number_values(token)

    missing = set()
    for token in re.findall(pattern, answer):
        if len(token) < 2 or token in sources:
            continue
        if not (_number_values(token) & source_values):
            missing.add(token)
    return sorted(missing)


def judge_groundedness(
    answer: str, outputs: list[dict], judge_model=None, extra_sources: list[str] | None = None
) -> GroundednessJudgement:
    """Un LLM comprueba si la respuesta se apoya SOLO en las fuentes recibidas."""
    model = judge_model or init_chat_model(MODEL_NAME, temperature=0)
    judge = model.with_structured_output(GroundednessJudgement)
    prompt = (
        "Comprueba si la RESPUESTA está respaldada por las FUENTES. Usa solo las fuentes, "
        "no tu propio conocimiento: si una afirmación no aparece en ellas, no está respaldada.\n"
        "- Los resultados de herramientas (p. ej. multiply) también son fuentes válidas.\n"
        "- Si una afirmación aparece en las fuentes, aunque sea en otro idioma, SÍ está respaldada.\n"
        "- Una afirmación respaldada no debe aparecer en 'afirmaciones_sin_respaldo'.\n\n"
        f"FUENTES:\n{sources_text(outputs, extra_sources)}\n\nRESPUESTA:\n{answer}"
    )
    return judge.invoke(prompt)


def evaluate_groundedness(queries: list[dict], results: list[dict], judge_model=None) -> list[dict]:
    """Evalúa, en las queries que buscaron en la web, si la respuesta se apoya en las fuentes.

    Las fuentes son los snippets de Tavily MÁS los resultados de las demás herramientas.
    """
    evaluations = []
    for q, r in zip(queries, results):
        outputs = [o for o in get_tavily_outputs(r) if "results" in o]
        if not outputs:
            continue
        extra = tool_sources(r)
        answer = get_answer(r)
        verdict = judge_groundedness(answer, outputs, judge_model, extra)
        evaluations.append(
            {
                "query": q["query"],
                "respaldada": verdict.respaldada,
                "sin_respaldo": verdict.afirmaciones_sin_respaldo,
                "numeros_sin_respaldo": unsupported_numbers(answer, sources_text(outputs, extra)),
                "explicacion": verdict.explicacion,
            }
        )
    return evaluations


def print_groundedness(evaluations: list[dict]):
    """Tabla y detalle de la evaluación de alucinaciones."""
    print(f"{'Query':<55} {'Respaldada':<11} {'Nº sin respaldo':>15}")
    for e in evaluations:
        text = e["query"] if len(e["query"]) <= 53 else e["query"][:50] + "..."
        print(f"{text:<55} {'sí' if e['respaldada'] else 'NO':<11} {len(e['sin_respaldo']):>15}")
    for e in evaluations:
        if e["sin_respaldo"] or e["numeros_sin_respaldo"]:
            print(f"\n- {e['query']}")
            for claim in e["sin_respaldo"]:
                print(f"    sin respaldo: {claim}")
            if e["numeros_sin_respaldo"]:
                print(f"    números que no están en las fuentes: {e['numeros_sin_respaldo']}")
            print(f"    juez: {e['explicacion']}")


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
