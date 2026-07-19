"""
Central configuration for the AI Data Analyst Agent.

Everything here can be overridden with environment variables, so the same
code runs unchanged in dev vs. a locked-down internal server.
"""

import os

# --- LLM backend (Ollama, running locally or on an internal host) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# qwen3 (the default 8B tag) is a solid default: verified tool-calling +
# thinking support, ~5GB, runs comfortably on a single modern GPU or a
# recent Apple Silicon Mac. Swap for qwen3:4b on weaker hardware, or
# qwen3:14b / qwen3:30b-a3b if you have more headroom and want fewer
# tool-calling mistakes on complex, multi-step questions.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
DEFAULT_USE_THINKING = os.environ.get("OLLAMA_THINK", "true").lower() == "true"

# --- Safety limits ---
MAX_SQL_ROWS_RETURNED = 500     # rows returned to the LLM/UI per query
SANDBOX_TIMEOUT_SECONDS = 10    # kill execute_python calls that run longer than this
MAX_TOOL_ITERATIONS = 6         # stop the agent loop after N tool round-trips

# --- Privacy ---
# Columns whose name contains any of these (case-insensitive) get masked in
# the UI/agent output when "Privacy mode" is on. Tune this list to match
# your actual schema (MSISDN/IMSI/IMEI are GSM/telecom identifiers).
SENSITIVE_COLUMN_HINTS = [
    "msisdn", "phone", "mobile", "cnic", "imsi", "imei",
    "email", "address", "account_no", "iban", "card_no", "passport",
]
