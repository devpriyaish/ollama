# ==========================
# Environment & Config
# ==========================
import json
from dotenv import load_dotenv
load_dotenv()

# ==========================
# Standard Library Imports
# ==========================
import os
import re

from typing import Dict, List, Optional

# ==========================
# Third-Party Libraries
# ==========================
from langchain_core.tools import tool
import requests

# ==========================
# OCEAN Setup
# ==========================
OCEANN_JWT_TOKEN = os.getenv("OCEANN_JWT_TOKEN")


from langchain_community.chat_models import ChatOllama

llm_parser = ChatOllama(
    model=os.getenv("OLLAMA_PARSER_MODEL", "gpt-oss:20b"),
    temperature=0,
    num_ctx=4096,
    num_predict=512,
    format="json"  # 🔒 forces JSON-only output
)


# ==========================
# VOYAGE INTERNAL TOOLS
# ==========================

@tool
def get_vessels_by_name(query: str) -> dict:
    """
    Fetch vessel list by vessel name or partial name with error handling.
    """
    url = f"https://your/url/get-vessels-name/{query}"

    headers = {
        "accept": "*/*",
        "authorization": OCEANN_JWT_TOKEN,
        "endpoint": "Map Intelligence",
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "url": url,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann API",
            "details": str(conn_err),
            "url": url,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "Request to TheOceann API timed out",
            "url": url,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "url": url,
        }

@tool
def get_vessel_particulars(mmsi: str, imo: str, ship_id: str, vessel_name: str) -> dict:
    """
    Fetch detailed vessel particulars (Lloyd's data) including dimensions, 
    tonnages, ownership, hull details, and technical specs.

    Args:
        mmsi (str): MMSI number of the vessel. Example: "403591001"
        imo (str): IMO number of the vessel. Example: "9837119"
        ship_id (str): Internal SHIP_ID. Example: "12836167"
        vessel_name (str): Name of the vessel. Example: "SARA"

    Returns:
        dict: JSON response containing full vessel particulars, or error dict.
    """
    
    try:
        url = f"https://your/url/get-vessel-particulars/{mmsi}/{imo}/{ship_id}/{vessel_name}"

        headers = {
            "accept": "*/*",
            "authorization": OCEANN_JWT_TOKEN,
            "endpoint": "Map Intelligence",
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle null or empty response
        if data is None:
            return {
                "error": "No data returned",
                "message": "API returned null response for the vessel"
            }
        
        # Handle empty dict or list
        if not data:
            return {
                "error": "Empty response",
                "message": "API returned empty data for the vessel"
            }
            
        return data
        
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "message": "The API request timed out after 30 seconds"
        }
    
    except requests.exceptions.HTTPError as e:
        return {
            "error": "HTTP error",
            "status_code": e.response.status_code,
            "message": str(e)
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "error": "Request failed",
            "message": f"Failed to fetch vessel particulars: {str(e)}"
        }
    
    except ValueError as e:
        return {
            "error": "Invalid JSON",
            "message": "API response could not be parsed as JSON"
        }
    
    except Exception as e:
        return {
            "error": "Unexpected error",
            "message": f"An unexpected error occurred: {str(e)}"
        }

@tool
def categorize_single_port_call(v: str, shipid: str, msgtype: str) -> dict:
    """
    Categorize all port calls (arrivals & departures) for a given vessel 
    using SHIP_ID and parameters v & msgtype.
    """

    url = "https://your/url/categorize-single-port-call"
    params = {
        "v": v,
        "shipid": shipid,
        "msgtype": msgtype,
    }

    headers = {
        "accept": "*/*",
        "authorization": OCEANN_JWT_TOKEN,
        "endpoint": "Map Intelligence",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "params": params,
            "url": url,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann API.",
            "details": str(conn_err),
            "params": params,
            "url": url,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "TheOceann API request timed out.",
            "params": params,
            "url": url,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "params": params,
            "url": url,
        }

@tool
def expected_port_arrivals(port_name: str, msg_type: str = "simple") -> dict:
    """
    Get expected port arrivals for a given port.

    Args:
        port_name (str): Name of the port. Example: "SIKKA"
        msg_type (str): Message type. Options: "simple" or "extended".
    """

    url = "https://your/url/expected-port-arrivals"

    params = {
        "portName": port_name,
        "msgType": msg_type
    }

    headers = {
        "accept": "*/*",
        "authorization": OCEANN_JWT_TOKEN,
        "endpoint": "Map Intelligence",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "params": params,
            "url": url,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann API.",
            "details": str(conn_err),
            "params": params,
            "url": url,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "TheOceann API request timed out.",
            "params": params,
            "url": url,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "params": params,
            "url": url,
        }

@tool
def get_port_distance(
    from_port: str,
    to_port: str,
    localEca: int = 1,
    seca: int = 3,
    canalOptions: str = "111",
    piracyArea: str = "001"
) -> dict:
    """
    Get port-to-port distance with route geometry, SECA length, canal options,
    HRA length, and detailed LineString coordinates.
    """

    url = "https://your/url/distance"

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": OCEANN_JWT_TOKEN,
        "Content-Type": "application/json",
        "endpoint": "Chartering Dashboard",
    }

    payload = {
        "from": from_port,
        "to": to_port,
        "localEca": localEca,
        "seca": seca,
        "canalOptions": canalOptions,
        "piracyArea": piracyArea
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "url": url,
            "payload": payload,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann Distance API.",
            "details": str(conn_err),
            "url": url,
            "payload": payload,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "TheOceann Distance API request timed out.",
            "url": url,
            "payload": payload,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "url": url,
            "payload": payload,
        }

@tool
def get_bunker_spotprice_by_port(port_name: str) -> dict:
    """
    Fetch bunker spot prices and future price curves by searching a port name.
    Uses partial search API: searchport-full?portName=<name>
    """

    url = (
        "https://your/url/port-bunker-activity/"
        f"searchport-full?portName={port_name}"
    )

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": OCEANN_JWT_TOKEN,   # <-- Use your stored token
        "cache-control": "no-cache",
        "endpoint": "Bunker Prices",
        "origin": "https://devmail-thor.theoceann.com",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://devmail-thor.theoceann.com/",
        "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "url": url,
            "port_name": port_name,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann API.",
            "details": str(conn_err),
            "url": url,
            "port_name": port_name,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "TheOceann API request timed out.",
            "url": url,
            "port_name": port_name,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "url": url,
            "port_name": port_name,
        }

@tool
def get_weather_speed(payload: dict) -> dict:
    """
    Calls: https://devapiservices.theoceann.com/marine/get-weather-speed
    Payload must contain:
      fuel_cons, vessel_speed, piracyArea, api_call, canalOptions,
      multiple_ports, vessel_name, vessel_type, DWT, IMO, MMSI, date
    """

    url = "https://your/url/get-weather-speed"

    headers = {
        "Authorization": OCEANN_JWT_TOKEN,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "endpoint": "Chartering Dashboard",
        "User-Agent": "Python/Requests Script"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "url": url,
            "payload": payload,
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to TheOceann Weather Speed API.",
            "details": str(conn_err),
            "url": url,
            "payload": payload,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "Weather Speed API request timed out.",
            "url": url,
            "payload": payload,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "url": url,
            "payload": payload,
        }

@tool
def best_match_cargo(cargo_size: int, cargo_type: str, load_port: str, change_tab: str) -> dict:
    """
    Calls the Best-Match-Cargo API to retrieve the best matched vessels for a cargo.
    """

    url = "https://your/url/best-match-cargo"

    headers = {
        "Authorization": OCEANN_JWT_TOKEN,
        "Content-Type": "application/json",
        "Accept": "*/*",
        "endpoint": "Cargo",
        "Origin": "https://devmail-thor.theoceann.com",
        "Referer": "https://devmail-thor.theoceann.com/"
    }

    payload = {
        "cargo_size": cargo_size,
        "cargo_type": cargo_type,
        "load_port": load_port,
        "change_tab": change_tab
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "type": "http_error",
            "message": str(http_err),
            "url": url,
            "payload": payload
        }

    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "type": "connection_error",
            "message": "Failed to connect to Best-Match-Cargo API.",
            "details": str(conn_err),
            "url": url,
            "payload": payload
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "type": "timeout",
            "message": "Best-Match-Cargo API request timed out.",
            "url": url,
            "payload": payload
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "unknown_error",
            "message": str(e),
            "url": url,
            "payload": payload
        }

@tool
def match_open_vessels(dwt: str, open_port: str) -> dict:
    """
    Calls the Best-Match-Vessel API to retrieve matched vessels.
    """

    url = "https://your/url/best_match_vessel"

    headers = {
        "Authorization": OCEANN_JWT_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "dwt": dwt,
        "open_port": open_port
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        
        return {
            "status": "success",
            "source": "best_match_vessel_api",
            "summary_hint": "Summarize the best matching open vessels for the given DWT and port.",
            "input": payload,
            "vessels": response.json()
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "reason": "network_or_http",
            "message": "Vessel service reachable but request failed.",
            "debug": {
                "error": str(e),
                "raw_response": getattr(e.response, "text", None)
            }
        }

@tool
def calculate_dwt(cargo_quantity: float) -> dict:
    """
    Calculate the DWT as cargo quantity + 10%.

    Args:
        cargo_quantity (float): Cargo quantity in MT.

    Returns:
        dict: {"dwt": <calculated_dwt>}
    """
    return {"dwt": cargo_quantity + (cargo_quantity / 10)}

@tool
def parse_speed_and_consumption_ai(
    speed_and_consumption: str = None,
    manual_ballast_speed: float = None,
    manual_laden_speed: float = None,
    manual_ballast_consumption: float = None,
    manual_laden_consumption: float = None,
    manual_fuel_type: str = None,
) -> dict:
    """
    AI → Try to extract speed + consumption.
    MANUAL → If AI fails, force user to enter all values.
    """

    # ---------------------------------
    # ✅ 1. TRY AI EXTRACTION FIRST
    # ---------------------------------
    if speed_and_consumption and isinstance(speed_and_consumption, str):

        prompt = f"""
            You are a maritime technical data parser.

            Extract and normalize this vessel data:

            INPUT:
            {speed_and_consumption}

            Return STRICT JSON:

            {{
            "ballast_speed": float | null,
            "laden_speed": float | null,
            "ballast_consumption": float | null,
            "laden_consumption": float | null,
            "fuel_type": string | null
            }}

            Rules:
            - Fix broken decimals like "11, 80" → 11.8
            - If only one speed → use it for both
            - If only one consumption → use it for both
            - If nothing found → return all fields as null
        """

        try:
            resp = llm_parser.invoke(prompt)

            raw = resp.content.strip()

            # Safety: extract JSON if model wraps text accidentally
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    raise ValueError("Ollama returned invalid JSON")
                parsed = json.loads(match.group(0))


            # ✅ If AI got ALL REQUIRED FIELDS → return success
            if (
                parsed.get("ballast_speed") is not None and
                parsed.get("laden_speed") is not None and
                parsed.get("ballast_consumption") is not None and
                parsed.get("laden_consumption") is not None
            ):
                return {
                    "status": "auto_extracted",
                    "ballast_speed": parsed["ballast_speed"],
                    "laden_speed": parsed["laden_speed"],
                    "ballast_consumption": parsed["ballast_consumption"],
                    "laden_consumption": parsed["laden_consumption"],
                    "fuel_type": parsed.get("fuel_type", "VLSFO"),
                    "mode": "ai"
                }

        except Exception:
            pass  # ✅ HARD FAIL → Fall through to manual

    # ---------------------------------
    # ✅ 2. MANUAL ENTRY MODE
    # ---------------------------------
    if (
        manual_ballast_speed is None or
        manual_laden_speed is None or
        manual_ballast_consumption is None or
        manual_laden_consumption is None
    ):
        return {
            "status": "manual_input_required",
            "message": (
                "Unable to extract vessel speed & bunker consumption automatically.\n"
                "Please enter the following manually:\n"
                "- Ballast Speed (knots)\n"
                "- Laden Speed (knots)\n"
                "- Ballast Consumption (MT/day)\n"
                "- Laden Consumption (MT/day)\n"
                "- Fuel Type (e.g., VLSFO, MGO)"
            ),
            "required_inputs": [
                "manual_ballast_speed",
                "manual_laden_speed",
                "manual_ballast_consumption",
                "manual_laden_consumption",
                "manual_fuel_type"
            ],
            "mode": "manual"
        }

    # ---------------------------------
    # ✅ 3. USER HAS PROVIDED MANUAL VALUES
    # ---------------------------------
    if isinstance(speed_and_consumption, dict):
        return {
            "status": "manual_extracted",
            "ballast_speed": speed_and_consumption.get("ballast_speed"),
            "laden_speed": speed_and_consumption.get("laden_speed"),
            "ballast_consumption": speed_and_consumption.get("ballast_consumption"),
            "laden_consumption": speed_and_consumption.get("laden_consumption"),
            "fuel_type": speed_and_consumption.get("fuel_type", "VLSFO"),
            "mode": "manual"
        }

@tool
def compute_voyage_days(
    route_distance: float,
    laden_speed: float
) -> Dict[str, float]:
    """
    Computes total voyage days using route distance and FINAL vessel speed values.
    This tool MUST NOT perform any parsing or manual prompting.
    """

    # ✅ Strict validation
    if not route_distance or route_distance <= 0:
        return {"status": "error", "message": "Invalid route distance"}

    if not laden_speed or laden_speed <= 0:
        return {"status": "error", "message": "Invalid laden speed"}

    # ✅ Correct average-speed-based voyage calculation
    voyage_days = route_distance / (laden_speed * 24)

    return {
        "status": "success",
        "route_distance_nm": round(route_distance, 2),
        "laden_speed_knots": float(laden_speed),
        "voyage_days": round(voyage_days, 2)
    }

@tool
def compute_bunker_consumption(
    voyage_days: float,
    laden_consumption: float,
    fuel_type: str
) -> Dict[str, float]:
    """
    Computes total bunker consumption using voyage days and FINAL bunker rates.
    This tool MUST NOT perform any parsing or manual prompting.
    """

    # ✅ Strict validation
    if not voyage_days or voyage_days <= 0:
        return {"status": "error", "message": "Invalid voyage days"}

    if laden_consumption is None or laden_consumption < 0:
        return {"status": "error", "message": "Invalid laden consumption"}

    # ✅ Total bunker calculation
    total_bunker = voyage_days * laden_consumption

    return {
        "status": "success",
        "voyage_days": round(voyage_days, 2),
        "laden_consumption_mt_per_day": float(laden_consumption),
        "total_bunker_mt": round(total_bunker, 2),
        "fuel_type": fuel_type or "VLSFO"
    }

@tool
def calculate_required_freight_rate(target_tce: float, voyage_days: int, voyage_cost: float, cargo_qty: float) -> dict:
    """
    Calculates Required Gross Freight and Required Freight Rate ($/mt).
    
    Formulas:
    Gross Freight (L) = TCE * Voyage Days + Voyage Costs
    Freight Rate (F) = Gross Freight / Cargo Quantity
    """

    try:
        if cargo_qty <= 0:
            return {
                "status": "error",
                "type": "invalid_input",
                "message": "Cargo quantity must be greater than zero.",
                "inputs": {
                    "target_tce": target_tce,
                    "voyage_days": voyage_days,
                    "voyage_cost": voyage_cost,
                    "cargo_qty": cargo_qty
                }
            }

        # Step 1 — Required Gross Freight
        gross_freight = (target_tce * voyage_days) + voyage_cost

        # Step 2 — Required Freight Rate
        freight_rate = gross_freight / cargo_qty

        return {
            "status": "success",
            "target_tce": target_tce,
            "voyage_days": voyage_days,
            "voyage_cost": voyage_cost,
            "cargo_qty": cargo_qty,
            "gross_freight": gross_freight,
            "freight_rate": freight_rate,
            "freight_rate_unit": "$/mt"
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "calculation_error",
            "message": str(e),
            "inputs": {
                "target_tce": target_tce,
                "voyage_days": voyage_days,
                "voyage_cost": voyage_cost,
                "cargo_qty": cargo_qty
            }
        }

@tool
def calculate_reverse_freight_rate(cargo_qty: float, voyage_cost: float, expected_profit: float, commission_pct: float = 0.0) -> dict:
    """
    Reverse calculates Freight Rate based on expected profit.
    
    Formula:
    Freight Rate ($/mt) =
    (Total Voyage Cost + Expected Profit)
    / (Cargo Qty * (1 - Commission%))
    
    Commission% should be given as a decimal (e.g., 2.5% → 0.025)
    """

    try:
        if cargo_qty <= 0:
            return {
                "status": "error",
                "type": "invalid_input",
                "message": "Cargo quantity must be greater than zero.",
                "inputs": {
                    "cargo_qty": cargo_qty,
                    "voyage_cost": voyage_cost,
                    "expected_profit": expected_profit,
                    "commission_pct": commission_pct
                }
            }

        if commission_pct >= 1:
            return {
                "status": "error",
                "type": "invalid_input",
                "message": "Commission percentage must be less than 1 (i.e., <100%).",
                "inputs": {
                    "cargo_qty": cargo_qty,
                    "voyage_cost": voyage_cost,
                    "expected_profit": expected_profit,
                    "commission_pct": commission_pct
                }
            }

        # Calculation
        denominator = cargo_qty * (1 - commission_pct)
        freight_rate = (voyage_cost + expected_profit) / denominator

        return {
            "status": "success",
            "cargo_qty": cargo_qty,
            "voyage_cost": voyage_cost,
            "expected_profit": expected_profit,
            "commission_pct": commission_pct,
            "freight_rate": freight_rate,
            "freight_rate_unit": "$/mt"
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "calculation_error",
            "message": str(e),
            "inputs": {
                "cargo_qty": cargo_qty,
                "voyage_cost": voyage_cost,
                "expected_profit": expected_profit,
                "commission_pct": commission_pct
            }
        }

@tool
def calculate_reverse_daily_hire(cargo_qty: float, freight_rate: float, hire_days: int,
                                 voyage_cost_excl_hire: float, expected_profit: float = 0.0) -> dict:
    """
    Reverse Calculates Required Daily Hire Rate (TCE target).

    Formula:
    Total Revenue = Cargo Qty × Freight Rate

    Hire Rate ($/day) =
    (Total Revenue - Voyage Cost Excl. Hire - Expected Profit) / Hire Days
    """

    try:
        if cargo_qty <= 0 or hire_days <= 0:
            return {
                "status": "error",
                "type": "invalid_input",
                "message": "Cargo quantity and hire days must be greater than zero.",
                "inputs": {
                    "cargo_qty": cargo_qty,
                    "freight_rate": freight_rate,
                    "hire_days": hire_days,
                    "voyage_cost_excl_hire": voyage_cost_excl_hire,
                    "expected_profit": expected_profit
                }
            }

        # Step 1: Total Revenue
        total_revenue = cargo_qty * freight_rate

        # Step 2: Required Hire Rate (Daily TCE)
        hire_rate = (total_revenue - voyage_cost_excl_hire - expected_profit) / hire_days

        status = "target_reached_or_better" if hire_rate >= 0 else "loss_condition"

        return {
            "status": "success",
            "total_revenue": total_revenue,
            "hire_rate": hire_rate,
            "hire_rate_unit": "$/day",
            "hire_days": hire_days,
            "cargo_qty": cargo_qty,
            "freight_rate": freight_rate,
            "voyage_cost_excl_hire": voyage_cost_excl_hire,
            "expected_profit": expected_profit,
            "profit_status": status
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "calculation_error",
            "message": str(e),
            "inputs": {
                "cargo_qty": cargo_qty,
                "freight_rate": freight_rate,
                "hire_days": hire_days,
                "voyage_cost_excl_hire": voyage_cost_excl_hire,
                "expected_profit": expected_profit
            }
        }

@tool
def calculate_reverse_tce(total_revenue: float, total_voyage_cost: float, voyage_days: int) -> dict:
    """
    Reverse Calculates TCE (Time Charter Equivalent).

    Formula:
    TCE = (Revenue - Total Cost) / Total Voyage Days
    """

    try:
        if voyage_days <= 0:
            return {
                "status": "error",
                "type": "invalid_input",
                "message": "Voyage days must be greater than zero.",
                "inputs": {
                    "total_revenue": total_revenue,
                    "total_voyage_cost": total_voyage_cost,
                    "voyage_days": voyage_days
                }
            }

        # Step 1: Profit component
        profit_value = total_revenue - total_voyage_cost

        # Step 2: TCE rate
        tce = profit_value / voyage_days

        status = "profit" if tce >= 0 else "loss"

        return {
            "status": "success",
            "tce": tce,
            "tce_unit": "$/day",
            "profit_value": profit_value,
            "profit_status": status,
            "total_revenue": total_revenue,
            "total_voyage_cost": total_voyage_cost,
            "voyage_days": voyage_days,
        }

    except Exception as e:
        return {
            "status": "error",
            "type": "calculation_error",
            "message": str(e),
            "inputs": {
                "total_revenue": total_revenue,
                "total_voyage_cost": total_voyage_cost,
                "voyage_days": voyage_days
            }
        }

@tool
def calculate_voyage_pnl(
    cargo_rows: List[Dict],
    demurrage_rows: list,
    despatch_rows: list,
    mis_revenue: float,
    broker_commission: float,
    voyage_days: float,
    hire_rate: float,
    tci_add_com: float,
    tci_broker_com: float,
    port_expenses: float,
    misc_expenses: float,
    bunkers: Dict,
    
    total_bunker_expense: float = 0.0,

    address_commission: float = 0.0,
    option_percentage: float = 0.0,
    freight_tax_pct: float = 0.0,
    demurrage_commission_pct: Optional[float] = None,
    despatch_commission_pct: float = 0.0,
    canal_cost: float = 0.0,
    ballast_bonus: float = 0.0,
    suez_bonus: float = 0.0,
    weather_factor: float = 1.0,
    speed_knots: Optional[float] = None,
    port_days: Optional[float] = None,
    # bunker_sea_cost_per_mt: float = None,
    # bunker_port_cost_per_mt: float = None,
    vessel_name: Optional[str] = None,
    vessel_dwt: float = None,
    vessel_year: int = None,
    cp_qty: float = None,
    option_qty: float = None
):
    """
    Calculate full Voyage P&L, TCE, gross TCE and breakeven freight
    for a voyage based on cargo rows, demurrage/despatch, hire, 
    bunker consumption, port costs and other voyage expenses.

    Returns:
        dict: {
            "status": "success" | "error",
            "results": {
                "pnl": float,
                "daily_profit": float,
                "tce": float,
                "gross_tce": float,
                "break_even_freight_usd_per_mt": float,
                "total_cargo_qty_mt": float,
            }
        }
    """

    try:
        # ------------------------------
        # 1. HANDLE DEFAULTS
        # ------------------------------
        if demurrage_commission_pct is None:
            demurrage_commission_pct = broker_commission

        cargo_rows = cargo_rows or []
        demurrage_rows = demurrage_rows or []
        despatch_rows = despatch_rows or []

        # bunkers = bunkers or {}
        # sea_days = bunkers.get("sea_days", 0.0)
        # sea_cons = bunkers.get("sea_cons", 0.0)
        # sea_cost = bunker_sea_cost_per_mt if bunker_sea_cost_per_mt is not None else bunkers.get("sea_cost", 0.0)

        # base_port_days = bunkers.get("port_days", 0.0)
        # effective_port_days = port_days if port_days is not None else base_port_days
        # port_cons = bunkers.get("port_cons", 0.0)
        # port_cost = bunker_port_cost_per_mt if bunker_port_cost_per_mt is not None else bunkers.get("port_cost", 0.0)

        # effective_sea_days = sea_days * weather_factor

        # ------------------------------
        # 2. FREIGHT
        # ------------------------------
        total_freight = 0.0
        total_freight_commission = 0.0
        total_freight_tax = 0.0

        for row in cargo_rows:
            row_cp_qty = float(row.get("cp_qty", 0.0))
            row_option_pct = row.get("option_pct", None)
            frt_rate = float(row.get("frt_rate", 0.0))
            lumpsum = float(row.get("lumpsum", 0.0))

            if option_qty is not None:
                effective_qty = row_cp_qty + float(option_qty)
            else:
                pct = row_option_pct if row_option_pct is not None else option_percentage
                effective_qty = row_cp_qty * (1.0 + float(pct))

            freight = lumpsum if lumpsum > 0 else effective_qty * frt_rate

            freight_commission = freight * float(broker_commission)
            freight_tax = freight * float(freight_tax_pct)

            total_freight += freight
            total_freight_commission += freight_commission
            total_freight_tax += freight_tax

        # ------------------------------
        # 3. DEMURRAGE & DESPATCH
        # ------------------------------
        total_demurrage = sum(float(x.get("amount", 0.0)) for x in demurrage_rows)
        total_despatch = sum(float(x.get("amount", 0.0)) for x in despatch_rows)

        total_demurrage_commission = total_demurrage * float(demurrage_commission_pct)
        total_despatch_commission = total_despatch * float(despatch_commission_pct)

        # ------------------------------
        # 4. REVENUE
        # ------------------------------
        gross_revenue = total_freight + mis_revenue + total_demurrage - total_despatch

        total_revenue_commissions = (
            total_freight_commission +
            total_demurrage_commission +
            total_despatch_commission +
            total_freight_tax
        )

        net_revenue = gross_revenue - total_revenue_commissions

        # ------------------------------
        # 5. HIRE
        # ------------------------------
        vessel_hire_cost = float(hire_rate) * float(voyage_days)

        tci_add_commission_value = vessel_hire_cost * float(tci_add_com)
        tci_broker_commission_value = vessel_hire_cost * float(tci_broker_com)

        # ------------------------------
        # 6. BUNKERS
        # ------------------------------
        # bunker_sea_amount = effective_sea_days * sea_cons * sea_cost
        # bunker_port_amount = effective_port_days * port_cons * port_cost
        # total_bunker_expense = bunker_sea_amount + bunker_port_amount
        total_bunker_expense = 0.0

        # ------------------------------
        # 7. EXPENSE
        # ------------------------------
        gross_expense = (
            vessel_hire_cost +
            port_expenses +
            misc_expenses +
            total_bunker_expense +
            canal_cost +
            ballast_bonus +
            suez_bonus
        )

        net_expense = gross_expense - (tci_add_commission_value + tci_broker_commission_value)

        # ------------------------------
        # 8. RESULTS
        # ------------------------------
        pnl = net_revenue - net_expense
        daily_profit = pnl / voyage_days if voyage_days > 0 else 0.0

        address_commission_value_on_hire = vessel_hire_cost * float(address_commission)

        tce_numerator = net_revenue - (
            gross_expense - (vessel_hire_cost + ballast_bonus + suez_bonus - address_commission_value_on_hire)
        )

        tce = tce_numerator / voyage_days if voyage_days > 0 else 0.0
        gross_tce = tce / (1.0 - float(address_commission)) if address_commission < 1 else tce

        # ------------------------------
        # 9. BREAKEVEN
        # ------------------------------
        if cp_qty is not None:
            total_cargo_qty = float(cp_qty) + float(option_qty or 0.0)
        else:
            total_cargo_qty = sum(
                float(r.get("cp_qty", 0.0)) * (1.0 + float(r.get("option_pct", option_percentage)))
                for r in cargo_rows
            )

        if total_cargo_qty > 0:
            effective_comm_factor = 1.0 - float(broker_commission)
            break_even_freight = net_expense / (total_cargo_qty * effective_comm_factor)
        else:
            break_even_freight = 0.0

        return {
            "status": "success",
            "results": {
                "pnl": pnl,
                "daily_profit": daily_profit,
                "tce": tce,
                "gross_tce": gross_tce,
                "break_even_freight_usd_per_mt": break_even_freight,
                "total_cargo_qty_mt": total_cargo_qty,
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Voyage P&L calculation failed",
            "error": str(e)
        }

@tool
def calculate_quick_voyage_pnl(
    cargo_quantity_mt: float,
    freight_rate: float,
    freight_is_lumpsum: bool,
    voyage_days: float,
    hire_rate_per_day: float,
    total_bunker_mt: float,
    bunker_price_per_mt: float,
    port_cost_usd: float = 0.0,
    misc_cost_usd: float = 0.0,
    canal_cost_usd: float = 0.0,
    broker_commission_pct: float = 0.0,
    address_commission_pct: float = 0.0,
) -> Dict[str, float]:
    """
    Quick Voyage P&L calculator for the simple flow.

    EXPECTED DATA FLOW (like your example):
    - cargo_quantity_mt       → from user ("Cargo quantity (MT): 39855")
    - freight_rate            → from user (2000 $/MT OR lumpsum)
    - freight_is_lumpsum      → True if freight_rate is lumpsum, else False
    - voyage_days             → from compute_voyage_days(...) e.g. 57.31
    - total_bunker_mt         → from compute_bunker_consumption(...) e.g. 2865.5
    - bunker_price_per_mt     → from bunker API or user input e.g. 123
    - hire_rate_per_day       → from user ("Hire rate ($/Day): 9")
    - port_cost_usd           → summed port/canal/etc if user gives any, else 0
    - misc_cost_usd           → other misc voyage costs, else 0

    Returns a dict with:
    - total_freight
    - gross_revenue
    - net_revenue
    - bunker_cost
    - hire_cost
    - total_voyage_cost
    - pnl
    - daily_profit
    - tce
    - gross_tce
    - break_even_freight_usd_per_mt
    """

    # --------- FREIGHT / REVENUE ---------
    if freight_is_lumpsum:
        total_freight = float(freight_rate)
    else:
        total_freight = float(cargo_quantity_mt) * float(freight_rate)

    gross_revenue = total_freight
    freight_commission_value = gross_revenue * float(broker_commission_pct)
    net_revenue = gross_revenue - freight_commission_value

    # --------- COSTS ---------
    hire_cost = float(hire_rate_per_day) * float(voyage_days)
    bunker_cost = float(total_bunker_mt) * float(bunker_price_per_mt)

    total_voyage_cost = (
        hire_cost +
        float(port_cost_usd) +
        float(misc_cost_usd) +
        float(canal_cost_usd) +
        bunker_cost
    )

    # --------- PNL & DAILY PROFIT ---------
    pnl = net_revenue - total_voyage_cost
    daily_profit = pnl / voyage_days if voyage_days > 0 else 0.0

    # --------- TCE & GROSS TCE ---------
    # TCE = (Net revenue - (voyage costs excluding hire)) / voyage_days
    voyage_costs_excl_hire = (
        bunker_cost + float(port_cost_usd) + float(misc_cost_usd) + float(canal_cost_usd)
    )
    tce_numerator = net_revenue - voyage_costs_excl_hire
    tce = tce_numerator / voyage_days if voyage_days > 0 else 0.0

    gross_tce = (
        tce / (1.0 - float(address_commission_pct))
        if 0.0 <= address_commission_pct < 1.0 else tce
    )

    # --------- BREAKEVEN FREIGHT ---------
    if cargo_quantity_mt > 0 and (1.0 - float(broker_commission_pct)) > 0:
        break_even_freight = (
            total_voyage_cost / (cargo_quantity_mt * (1.0 - float(broker_commission_pct)))
        )
    else:
        break_even_freight = 0.0

    return {
        "total_freight": total_freight,
        "gross_revenue": gross_revenue,
        "net_revenue": net_revenue,
        "bunker_cost": bunker_cost,
        "hire_cost": hire_cost,
        "total_voyage_cost": total_voyage_cost,
        "pnl": pnl,
        "daily_profit": daily_profit,
        "tce": tce,
        "gross_tce": gross_tce,
        "break_even_freight_usd_per_mt": break_even_freight,

    }
