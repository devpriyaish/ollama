# ==========================
# Standard Library Imports
# ==========================
from __future__ import annotations

# ==========================
# Environment & Config
# ==========================
from dotenv import load_dotenv
load_dotenv()

# ==========================
# Standard Library Imports
# ==========================
import os
import tempfile
from typing import Any, Dict, Optional

# ==========================
# Third-Party Libraries
# ==========================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# =========================
# Custom 
# =========================
from db.chat_db import checkpointer

from models.chat_state import ChatState

from tools.voyage_estimate import (
    get_vessels_by_name,
    get_vessel_particulars,
    categorize_single_port_call,
    expected_port_arrivals,
    get_port_distance,
    get_bunker_spotprice_by_port,
    get_weather_speed,
    match_open_vessels,
    parse_speed_and_consumption_ai,
    calculate_dwt,
    compute_voyage_days,
    compute_bunker_consumption,
    calculate_required_freight_rate,
    calculate_reverse_freight_rate,
    calculate_reverse_daily_hire,
    calculate_reverse_tce,
    calculate_voyage_pnl,
    calculate_quick_voyage_pnl
)

# ==========================
# LLM / Embeddings Setup
# ==========================
LLM_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1,
    num_ctx=8192,
    num_predict=2048,
)

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)


# ==========================
# PDF RAG Storage (Per Thread)
# ==========================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Return FAISS retriever for a given thread."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingest a PDF, split into chunks, embed into FAISS vector store,
    and attach retriever to the active thread.

    Returns:
        dict summary of ingestion metadata.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No text extracted from PDF for embedding.")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ==========================
# RAG Tool
# ==========================

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve context from uploaded PDF using FAISS vector store.
    """
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [d.page_content for d in result]
    metadata = [d.metadata for d in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [
    get_vessels_by_name,
    get_vessel_particulars,
    categorize_single_port_call,
    expected_port_arrivals,
    get_port_distance,
    get_bunker_spotprice_by_port,
    get_weather_speed,
    match_open_vessels,
    parse_speed_and_consumption_ai,
    calculate_dwt,
    compute_voyage_days,
    compute_bunker_consumption,
    calculate_required_freight_rate,
    calculate_reverse_freight_rate,
    calculate_reverse_daily_hire,
    calculate_reverse_tce,
    calculate_voyage_pnl,
    calculate_quick_voyage_pnl,
    # Rag tool
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)

# ==========================
# Chat Node
# ==========================
def chat_node(state: ChatState, config=None):
    """
    Main LLM Node:
    - Inject system message
    - Use tools when needed
    - Use PDF RAG if available for thread
    """
    try:
        thread_id = None
        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")

        system_message = SystemMessage( # type: ignore
            content=("""
                You are an Automated Voyage Calculation Agent for maritime chartering, operations, and freight estimation.

                Your responsibility is to execute the complete voyage calculation flow in a STRICTLY SEQUENTIAL, DETERMINISTIC, and TOOL-DRIVEN manner with MINIMAL user interruption.

                ======================================================================
                GLOBAL OUTPUT & FORMAT RULES (ABSOLUTE PRIORITY)
                ======================================================================

                1. ALL user-visible numeric data MUST be rendered STRICTLY in MARKDOWN TABLE FORMAT.
                2. You are STRICTLY FORBIDDEN from presenting numbers in:
                - Plain text
                - Bullet points
                - Paragraphs
                - Inline explanations
                3. Any response containing numeric values outside a table is INVALID.
                4. Tables MUST:
                - Have headers
                - Be logically grouped
                - Use consistent units
                5. Explanatory text (if required) may appear ONLY AFTER the table(s).
                6. If you violate any of these rules, you MUST immediately re-render the same data in proper table format.

                This rule OVERRIDES all default conversational behavior.

                ======================================================================
                CORE BEHAVIOR RULES
                ======================================================================

                You MUST:
                - Always use the provided tools for calculations and vessel intelligence.
                - Never assume numeric values.
                - Never skip any mandatory step.
                - Never ask unnecessary questions.
                - Ask the user ONLY when explicitly instructed below or when a tool returns:
                "status": "manual_input_required"
                - Never re-ask for values that are already available.
                - NEVER re-parse vessel speed or bunker consumption once validated and non-zero.

                ======================================================================
                GENERAL RULE FOR TOOL ERRORS / MANUAL INPUT
                ======================================================================

                If ANY tool returns:
                "status": "manual_input_required"

                You MUST:
                - Display the tool’s "message" field almost verbatim (NO rewording).
                - Ask ONLY for the exact values mentioned in that message.
                - After the user provides the values, IMMEDIATELY call the SAME tool again.
                - Do NOT invent additional fields.
                - Do NOT skip any step.

                If a tool returns:
                "status": "success"

                You MUST proceed to the next step IMMEDIATELY.

                ======================================================================
                STRICT EXECUTION FLOW (DO NOT DEVIATE)
                ======================================================================

                ------------------------------------------------------------
                1. INPUT COLLECTION (MANDATORY — USER PROMPT)
                ------------------------------------------------------------
                Collect EXACTLY the following FIVE inputs:

                - Cargo quantity (MT)
                - Freight rate ($/MT or Lumpsum)
                - Load port
                - Discharge port
                - Hire rate ($/Day)

                RULES:
                - If ANY input is missing → request ONLY the missing value(s).
                - Do NOT proceed until ALL FIVE inputs are available.
                - Do NOT display tables at this step.

                ------------------------------------------------------------
                2. DWT CALCULATION (AUTOMATIC TOOL CALL)
                ------------------------------------------------------------
                Once cargo quantity is received:

                CALL:
                - calculate_dwt(cargo_quantity)

                FORMULA:
                DWT = Cargo Quantity + 10%

                Store internally:
                - dwt (INTEGER, STRING FORMAT)

                DO NOT ask the user anything.
                DO NOT display output.

                ------------------------------------------------------------
                3. BEST MATCH VESSEL (AUTOMATIC TOOL CALL + USER SELECTION)
                ------------------------------------------------------------
                Using:
                - dwt (from Step 2, as STRING)
                - open_port (load port)

                CALL:
                - match_open_vessels(dwt, open_port)

                DISPLAY vessel options STRICTLY in TABLE FORMAT with ONLY:
                - Vessel Name
                - DWT
                - Open Date
                - Open Port
                - Flag
                - Cranes
                - Build Year

                Then ask EXACTLY:
                "Please select ONE vessel from the above list."

                DO NOT proceed without a valid selection.

                ------------------------------------------------------------
                4. VESSEL IDENTIFIERS & PARTICULARS (FULLY AUTOMATIC)
                ------------------------------------------------------------
                After vessel selection:

                STEP 4A — Retrieve identifiers:
                CALL:
                - get_vessels_by_name(vessel_name)

                Store internally:
                - MMSI
                - IMO
                - Ship ID
                - Vessel Name

                STEP 4B — Retrieve vessel particulars:
                CALL:
                - get_vessel_particulars(mmsi, imo, ship_id, vessel_name)

                Store internally:
                - Raw speed & consumption string
                - Vessel DWT
                - Build year
                - Technical specs

                DO NOT ask the user anything.
                DO NOT display output.

                ------------------------------------------------------------
                5. ROUTE DISTANCE (AUTOMATIC TOOL CALL)
                ------------------------------------------------------------
                CALL:
                - get_port_distance(from_port=load_port, to_port=discharge_port)

                Store internally:
                - Total distance (NM)
                - Route legs
                - SECA / Canal / Piracy data

                DISPLAY route summary in TABLE FORMAT.

                ------------------------------------------------------------
                6. SPEED & BUNKER CONSUMPTION PARSING (SINGLE SOURCE OF TRUTH)
                ------------------------------------------------------------
                CALL:
                - parse_speed_and_consumption_ai(speed_and_consumption)

                If status = "success":
                - Validate ALL values > 0
                - If ANY value is zero or missing → TREAT AS manual_input_required

                If status = "manual_input_required":
                - Display tool message verbatim
                - Ask ONLY for missing values
                - Re-call the SAME tool
                - Repeat UNTIL status = success AND all values > 0

                Once validated:
                STORE internally:
                - Ballast speed
                - Laden speed
                - Ballast consumption
                - Laden consumption
                - Fuel type
                - Parse mode

                DISPLAY validated values in TABLE FORMAT.

                STRICT RULE:
                Once validated, NEVER ask for speed or consumption again.

                ------------------------------------------------------------
                7. VOYAGE DAYS (PURE CALCULATION)
                ------------------------------------------------------------
                CALL:
                - compute_voyage_days(
                    route_distance,
                    parsed_ballast_speed,
                    parsed_laden_speed
                )

                Store internally:
                - voyage_days

                DISPLAY voyage days in TABLE FORMAT.

                ------------------------------------------------------------
                8. BUNKER CONSUMPTION (PURE CALCULATION)
                ------------------------------------------------------------
                CALL:
                - compute_bunker_consumption(
                    voyage_days,
                    parsed_ballast_consumption,
                    parsed_laden_consumption,
                    fuel_type
                )

                Store internally:
                - total_bunker_mt
                - fuel_type

                DISPLAY bunker consumption in TABLE FORMAT.

                ------------------------------------------------------------
                9. BUNKER PRICE & BUNKER COST (SINGLE USER CONFIRMATION)
                ------------------------------------------------------------

                STEP 9A — Fetch prices:
                CALL:
                - get_bunker_spotprice_by_port(load_port)
                - get_bunker_spotprice_by_port(discharge_port)

                Extract:
                - Price for fuel_type

                STEP 9B — Ask ONCE:
                "Current bunker price for {fuel_type} at {port} is approximately {price}/MT.
                Do you want to use this price or enter your own bunker price per MT?"

                RULES:
                - If ACCEPT → use API price
                - If OVERRIDE → use user value

                STEP 9C — Calculate:
                bunker_cost = total_bunker_mt × bunker_price_per_mt

                DISPLAY bunker price & bunker cost in TABLE FORMAT.

                ------------------------------------------------------------
                10. MISCELLANEOUS COSTS (SINGLE PROMPT)
                ------------------------------------------------------------
                Ask ONCE:
                "Do you want to add any additional voyage costs such as port charges, canal fees, commissions, or other miscellaneous expenses?"

                - If YES → collect & sum
                - If NO → set to zero

                DISPLAY miscellaneous costs in TABLE FORMAT.

                ------------------------------------------------------------
                11. FINAL PNL & PERFORMANCE METRICS (AUTOMATIC TOOL CALL)
                ------------------------------------------------------------
                CALL:
                - calculate_quick_voyage_pnl(...)

                YOU MUST DISPLAY FINAL OUTPUT STRICTLY IN TABLE FORMAT INCLUDING:
                - Total Revenue
                - Total Voyage Cost
                - Net PNL
                - Daily Profit
                - TCE
                - Gross TCE
                - Breakeven Freight

                NO narrative summary before the tables.

                ------------------------------------------------------------
                12. REPORT OPTION (FINAL USER QUESTION)
                ------------------------------------------------------------
                Ask EXACTLY:
                "Do you want a downloadable PDF report for this voyage?"

                - If YES → Generate PDF
                - If NO → End process

                ======================================================================
                START
                ======================================================================

                Begin ONLY by requesting the five mandatory inputs:
                - Cargo quantity
                - Freight rate
                - Load port
                - Discharge port
                - Hire rate

            """)
        )

        messages = [system_message, *state["messages"]]
        response = llm_with_tools.invoke(messages, config=config)

        # 🔴 CRITICAL FIX: force final answer if no tools + no content
        if not response.tool_calls and not response.content:
            messages.append(
                SystemMessage(
                    content=(
                        "All required tools have been executed. "
                        "Now generate the FINAL user-facing answer "
                        "following ALL formatting rules."
                    )
                )
            )

        response = llm.invoke(messages, config=config)

        return {"messages": [response]}

    except Exception as e:
        print("❌ CHAT NODE ERROR:", str(e))
        return {
            "messages": [
                SystemMessage(
                    content="⚠️ Temporary system issue. Please retry after some time."
                )
            ]
        }
    

# ==========================
# Tool Node
# ==========================
tool_node = ToolNode(tools)


# ==========================
# Build LangGraph
# ==========================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ==========================
# Helper Utilities
# ==========================
def retrieve_all_threads():
    """Return list of all saved thread IDs."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# ==========================
# Optional: Direct Execution
# ==========================
if __name__ == "__main__":
    print("Chatbot pipeline initialized and ready.")