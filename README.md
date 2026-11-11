# Websearch Agent: evaluación de Tavily como search tool

Agente **LangGraph** con **Claude Haiku 4.5** que decide si responde con su conocimiento o busca en la web con **Tavily**. El proyecto evalúa Tavily como `search_tool` de un chatbot de empresa: cuándo se usa, cuántos tokens cuesta y cómo reducir su coste sin perder calidad.

| Paso | Contenido | Notebook |
|---|---|---|
| 1 | El agente busca cuando debe · acceso a los `ToolMessage` · tokens y peso de los `ToolMessage` | `notebook-websearch-eval.ipynb` |
| 2 | Estructura de Tavily, schema de `bind_tools()`, snippets, score, alucinaciones | `notebook-websearch-eval.ipynb` |
| 3 | Mejorar coste y relevancia: schema mínimo, parámetros, filtro por score, juez LLM (pydantic), prompt caching | `notebook-websearch-mejoras.ipynb` |

## Agente

Grafo `START → llm_call ⇄ tool_node → END`. Herramientas: `tavily_search` y `multiply` (de prueba). Los errores de herramienta se devuelven al modelo como `ToolMessage` para que el grafo no se rompa.

En lugar de una tool de fecha, el **system prompt** inyecta el corte de conocimiento (`julio de 2025`) y la fecha de hoy: lo anterior al corte se responde sin buscar; lo actual o dudoso, con `tavily_search`. `build_search_agent(use_search=False)` crea el mismo agente sin Tavily para comparar.

## Cómo se miden los tokens de un `ToolMessage` (Paso 1)

La API da `usage_metadata` por llamada, no por mensaje, así que el `ToolMessage` se deduce. La llamada 2 repite todo el contexto de la 1 y añade el `tool_call` (que el modelo generó como `O1`) más el `ToolMessage`:

```
tokens del ToolMessage = I2 - I1 - O1
```

Se paga como input en cada llamada posterior que lo arrastra. El `tool_call` se mide directamente (`output_tokens` de la llamada 1). Contrastado con el endpoint `count_tokens` de Anthropic: 12 tokens de diferencia.

## Resultados

Ejecución real con `claude-haiku-4-5` y Tavily.

**Paso 1.** El agente acierta cuándo buscar en 14 de 15 consultas (la excepción, Champions 2024-25, es defendible por el corte de conocimiento). Un `ToolMessage` de Tavily cuesta de media **2.867 tokens (35 % del input)**. En una búsqueda, el coste se reparte así: schemas de las tools 49 %, `ToolMessage` 26 %, output 17 %, resto 8 %.

**Paso 2.** El LLM ve 9 parámetros de Tavily; los schemas suman 2.260 tokens en cada llamada. El 82 % del JSON de Tavily es el `content`. 8 de 8 respuestas con web respaldadas por las fuentes (con el evaluador corregido).

**Paso 3** (8 consultas: 6 con respuesta y 2 sin respuesta posible; variantes acumulativas):

| Variante | Cambio | Coste | Ahorro | Resultados inyectados | Precisión |
|---|---|---|---|---|---|
| base | `TavilySearch` estándar | $0,0775 | – | 5,0 | 60 % |
| schema_min | tool propia, solo `query` | $0,0486 | −37 % | 5,0 | 60 % |
| params | 3 resultados, 1 chunk, 400 caracteres | $0,0309 | −60 % | 3,0 | 67 % |
| score | + umbral de score 0,6 | $0,0296 | −62 % | 2,2 | 83 % |
| juez_llm | + juez LLM (pydantic) | $0,0464 | −40 % | 1,9 | 94 %\* |

\* Circular: el mismo modelo filtra y evalúa. Una ejecución anterior dio ahorros casi iguales (−38 %, −59 %, −60 %, −38 %).

- **Schema:** de 2.260 a 682 tokens por llamada (−70 %).
- **Score:** el umbral 0,6 conserva el 95 % de los resultados relevantes y descarta el 68 % de la basura; a 0,7 el recall cae al 67 %.
- **Caché:** en una pregunta suelta pierde (escribe 4.900-7.300 tokens que nadie lee). En una conversación de 6 turnos con memoria: **$0,1619 → $0,0555 (−66 %)**; el primer turno cuesta un 13 % más.

## Conclusiones

1. **El mayor coste es la tool, no el modelo:** el schema de Tavily se envía en cada llamada y su salida se reenvía en las siguientes.
2. **Schema mínimo** (solo `query`): −37 % sin cambiar lo que ve el modelo. **Recortar resultados** (3, 1 chunk, 400 caracteres): −60 % acumulado sin pérdida de calidad medida, aunque el recorte puede cortar frases.
3. **El score es un filtro barato** (0,6), pero **no detecta fiablemente las búsquedas sin respuesta**: resultados irrelevantes puntuaron 0,95-0,98 en una ejecución y 0,01-0,03 en otra.
4. **El juez LLM no compensa:** añade 10.790 tokens y baja el ahorro del 62 % al 40 %.
5. **La caché solo compensa en conversaciones largas con memoria**, que es el caso de un chatbot real.
6. **Calidad:** filtrar no añade alucinaciones en las preguntas con respuesta. Si no se inyecta nada, el modelo especula ("la empresa no existe"); `STRICT_EMPTY_MESSAGE` lo endurece y se prueba en la sección 9 del notebook.
7. **Configuración recomendada:** tool lean, `max_results=3`, `chunks_per_source=1`, `max_chars=400`, `min_score≈0,6`, sin juez LLM, mensaje estricto en búsquedas vacías y caché en conversaciones con memoria.

## Limitaciones

- Muestras pequeñas (8 y 15 consultas) y una pasada por variante.
- El evaluador (precisión y alucinaciones) es Haiku, la misma familia que el agente. Se auditó y corrigió durante el trabajo: la primera versión daba falsos positivos (la fecha actual no contaba como fuente).
- Los resultados de Tavily varían con el momento.
- `ToolMessage` por diferencia: incluye unos pocos tokens de estructura.

## Estructura y uso

```
.
├── .env                  # claves (NO se sube a git)
├── pyproject.toml · uv.lock
├── notebooks/
│   ├── notebook-websearch-eval.ipynb      # Pasos 1 y 2
│   └── notebook-websearch-mejoras.ipynb   # Paso 3
└── src/victorg_langgraph_websearchagent/
    ├── agent.py          # agente básico (v1)
    └── agent_v2.py       # agente completo y utilidades (v2)
```

```bash
uv sync    # Python 3.12 y uv
```

Crea `.env` con `ANTHROPIC_API_KEY=sk-ant-...` y `TAVILY_API_KEY=tvly-...` (sin comillas) y comprueba que está en `.gitignore`. Abre un notebook con el kernel `.venv` y ejecútalo en orden: **hace llamadas reales y consume créditos** (el del Paso 3 cuesta ≈ 1 USD).

```python
from victorg_langgraph_websearchagent.agent_v2 import *

load_keys()
agent = build_search_agent()
print(get_answer(run_agent(agent, "¿Cuál es la última versión estable de LangGraph?")))
```

**Funciones principales de `agent_v2.py`:**

| Función | Qué hace |
|---|---|
| `build_search_agent(use_search, search_tool, cache, memory)` | Crea el agente (con o sin búsqueda, tool propia, caché, memoria) |
| `run_agent(agent, pregunta, thread_id)` | Ejecuta y devuelve el resultado completo |
| `count_tokens` · `toolmessage_tokens` · `cost_breakdown` | Tokens por llamada, peso de los `ToolMessage` y coste en USD (con caché y juez) |
| `build_lean_search_tool(...)` | Tool ligera: solo `query`; filtros `min_score`, `top_k`, juez LLM; datos descartados en un artifact sin coste de tokens |
| `score_threshold_sweep` · `choose_min_score` | Barrido del umbral de score frente a la relevancia real |
| `evaluate_variant` · `run_conversation` | Comparan variantes y miden la caché por turno |
| `evaluate_groundedness` | ¿La respuesta se apoya en las fuentes? |

## Referencias

[Claude Haiku 4.5](https://platform.claude.com/docs/en/models/haiku-4-5/overview) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [Tavily](https://tavily.com/)
