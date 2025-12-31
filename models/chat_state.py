# ==========================
# Standard Library Imports
# ==========================
from __future__ import annotations
from typing import Annotated, Optional, TypedDict

# ==========================
# Third-Party Libraries
# ==========================
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ==========================
# Chat State Definition
# ==========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Step 1 — Mandatory Inputs
    cargo_quantity: Optional[float]
    freight_rate: Optional[float]
    load_port: str | None
    discharge_port: str | None
    hire_rate: Optional[float]

    # Optional inputs
    vessel_name: Optional[str]

    # Step 2 — Derived
    dwt: Optional[float]
    vessel_type: Optional[str]
    vessel_particulars: Optional[dict]

    # Step 3 — Market Intake (from best_match_vessel)
    selected_vessel_name: str | None
    selected_vessel_speed_consumption: str | None
    selected_vessel_draft: str | None
    selected_vessel_tpc: str | None
    selected_vessel_loa: str | None
    selected_vessel_beam: str | None
    selected_vessel_flag: str | None
    selected_vessel_cranes: str | None
    selected_vessel_build_year: str | None
    selected_vessel_open_date: str | None
    selected_vessel_open_port: str | None
    selected_vessel_type: str | None
    selected_vessel_subtype: str | None
    selected_vessel_port_id: str | None

    # Step 4 — Registry Vessel Data
    vessel_details: dict | None

    # ✅ Step 6A — Route Distance (CRITICAL)
    route_distance: dict | None

    # ✅ Step 6B — Voyage Days (DERIVED)
    voyage_days: dict | None

    # Step 5 — Misc Costs
    misc_costs: list[dict] | None

    # Step 6B — Final PNL Output
    pnl: dict | None

    # Step 7 — Report Flag
    pdf_report_requested: bool | None
