# Telecom AI Data Analyst Agent

Chat with your data in plain English and get back SQL, Python analysis, and
charts — running entirely on your own machine, with no data sent to a
third-party API.

Upload CSV/Excel files, or connect to a database. Ask a question. The
agent (a local LLM via [Ollama](https://ollama.com)) decides which tools
to call — inspect schema, run SQL, run pandas code, draw a chart — and
answers using only what those tools return.

## Why local (Ollama) instead of a cloud LLM API?

Telecom usage data is customer PII (MSISDNs, CDRs, location-adjacent
usage patterns). Routing it through a third-party LLM API means it leaves
the building. Ollama runs the model on your own hardware/on-prem server,
so prompts, tool results, and data never leave your network.

## Quick start

1. **Install Ollama** — [ollama.com/download](https://ollama.com/download)
2. **Pull a tool-calling model:**
   ```bash
   ollama pull qwen3
   ```
   `qwen3` (the default 8B tag, ~5.2GB) supports both tool calling and
   "thinking" mode and is a solid default. On weaker hardware use
   `qwen3:4b`; with more headroom, `qwen3:14b` or the MoE `qwen3:30b-a3b`
   (runs at small-model speed since it only activates ~3B params/token)
   will make fewer tool-calling mistakes on complex, multi-step questions.
3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
5. **Load data and ask questions** — upload `sample_data/sample_telecom_usage.csv`
   to try it immediately, or connect a real (ideally read-only) database
   connection string in the sidebar.

Run `python3 test_pipeline.py` any time to sanity-check the data layer,
SQL guard, sandbox, charts, and the agent's tool-calling loop (mocked —
no live model needed) after making changes.

## Architecture

```
                 ┌─────────────────────────┐
  CSV / Excel ──▶│                          │
                 │      DataManager         │◀── SQL / schema lookups
  Database   ──▶ │  (DuckDB or SQLAlchemy)  │
                 └─────────────▲────────────┘
                                │
                     ┌──────────────────┐          ┌────────────────┐
  "Which region  ──▶ │   Agent (Ollama) │◀────────▶│  Tool schemas   │
   had highest        │  system prompt   │  tool     │ list_tables    │
   churn?"             │  + conversation │  calls    │ get_schema     │
                      └──────────────────┘          │ execute_sql    │
                                │                     │ execute_python │
                                ▼                     │ create_chart   │
                      Answer + SQL/code shown         └────────────────┘
                      + table + chart in the UI
```

The LLM never touches the data directly — it only ever sees whatever a
tool call returns. That's what makes `sql_guard`, `sandbox`, and the
privacy mask effective: they sit *between* the model and the data,
not inside the model's judgment.

| File | Role |
|---|---|
| `core/data_manager.py` | Loads CSV/Excel into DuckDB, or connects to an external DB via SQLAlchemy |
| `core/tools.py` | The 5 tool schemas + the dispatcher that executes them |
| `core/agent.py` | The Ollama chat + tool-calling loop |
| `core/sql_guard.py` | Blocks anything except SELECT/WITH/EXPLAIN |
| `core/sandbox.py` | Restricted `exec()` for the `execute_python` tool, with a timeout |
| `core/charts.py` | Plotly chart builder for the `create_chart` tool |
| `utils/privacy.py` | Masks columns that look like MSISDN/CNIC/IMEI/etc. in agent output |
| `app.py` | Streamlit chat UI |

## Security & privacy notes

This is a working prototype, not a hardened production system. Before
pointing it at real subscriber data, read this section properly:

- **Read-only SQL is defense-in-depth, not the boundary.** `sql_guard.py`
  blocks non-SELECT statements, but connect with a genuinely **read-only
  database role** too — a keyword filter can in principle be worked
  around, a DB permission can't.
- **The Python sandbox is a guardrail, not isolation.** It restricts
  builtins/imports and enforces a timeout, but it's in-process `exec()`,
  not a container or subprocess. For production use on sensitive data,
  run `execute_python` in a network-disabled subprocess or container
  (e.g. `nsjail`, `gVisor`, or a locked-down Docker container with
  `--network none`) instead.
- **Privacy mode masks display output, not storage.** `utils/privacy.py`
  masks values the agent shows back to the user; the underlying file or
  database is untouched. It's not anonymization or encryption.
- **Prompt injection via data content is a real risk class here.** The
  agent reads back real row values, including free-text fields. A crafted
  value in the data (e.g. a support-ticket comment) could try to steer
  the model. The hard boundaries — `sql_guard`, the sandbox, the masking —
  are what actually hold regardless of what the model is told to do; the
  system prompt instruction to treat data as data, not instructions, is a
  second layer, not the main defense.
- **Log questions and generated SQL/code before any real rollout.** An
  audit trail of who asked what, and what was actually executed, is
  standard practice for internal tools that touch customer data — this
  prototype doesn't include it yet.
- **Check with legal/compliance** if this will touch real subscriber
  data — PTA data-handling requirements, retention limits, and access
  logging are compliance questions, not engineering ones, and are worth
  confirming before moving past a sandboxed pilot.

## Known limitations (good next steps)

- External DB mode assumes simple, unquoted table identifiers; reserved
  words or special characters in table names aren't handled.
- `get_schema` on external databases uses `LIMIT 0`, which works on
  Postgres/MySQL/SQLite/DuckDB but not SQL Server (`TOP 0`) or Oracle —
  a small dialect-aware tweak would fix that.
- The sandbox's timeout *abandons* a runaway thread rather than killing
  it (Python can't force-kill a thread) — fine for a demo, not for
  untrusted multi-user load.
- No authentication on the Streamlit app itself — put it behind your
  existing SSO/VPN before sharing it beyond your own machine.
- No conversation persistence — refreshing the page clears the chat.

## Extending it

- Swap the LLM backend behind `core/agent.py` for an OpenAI-compatible
  endpoint (vLLM, LM Studio, etc.) if you outgrow Ollama — the tool
  schemas in `core/tools.py` are already in the standard OpenAI
  function-calling shape.
- Add a `forecast` tool (e.g. wrapping `statsmodels` or `prophet`) if
  forecasting comes up often enough to deserve its own tool rather than
  ad hoc `execute_python` calls.
- Add `core/audit.py` to log every question + generated SQL/code to a
  local file or table before this goes anywhere near real subscriber data.
