"""
Streamlit UI for the AI Data Analyst Agent.
Run with: streamlit run app.py
"""

import json
import os
import pandas as pd
import streamlit as st

from core.data_manager import DataManager
from core.agent import Agent
import config

st.set_page_config(page_title="Telecom AI Data Analyst", page_icon="📊", layout="wide")

# ---------- Session state ----------
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "privacy_mode" not in st.session_state:
    st.session_state.privacy_mode = True
if "use_thinking" not in st.session_state:
    st.session_state.use_thinking = config.DEFAULT_USE_THINKING
if "model_choice" not in st.session_state:
    st.session_state.model_choice = config.DEFAULT_MODEL

dm: DataManager = st.session_state.data_manager


def rebuild_agent():
    st.session_state.agent = Agent(
        dm,
        model=st.session_state.model_choice,
        privacy_mode=st.session_state.privacy_mode,
        use_thinking=st.session_state.use_thinking,
    )


def render_trace(trace: list):
    with st.expander("🔧 Steps the agent took"):
        for step in trace:
            st.markdown(f"**{step['tool']}**")
            st.code(json.dumps(step["args"], indent=2, default=str), language="json")
            st.text(step["result"][:2000])


def render_turn(turn: dict):
    st.markdown(turn["content"])
    if turn.get("trace"):
        render_trace(turn["trace"])
    if turn.get("dataframe") is not None:
        st.dataframe(turn["dataframe"], use_container_width=True)
    if turn.get("chart") is not None:
        st.plotly_chart(turn["chart"], use_container_width=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📁 Data source")
    tab_files, tab_db = st.tabs(["Upload files", "Connect database"])

    with tab_files:
        uploaded = st.file_uploader(
            "CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True
        )
        if uploaded and st.button("Load files", use_container_width=True):
            for f in uploaded:
                try:
                    if f.name.lower().endswith(".csv"):
                        dm.load_csv(f, f.name)
                    else:
                        dm.load_excel(f, f.name)
                except Exception as exc:
                    st.error(f"Couldn't load {f.name}: {exc}")
            rebuild_agent()
            st.success("Loaded.")

    with tab_db:
        st.caption(
            "SQLAlchemy connection URL, e.g. "
            "`postgresql+psycopg2://user:pass@host:5432/dbname` or "
            "`mysql+pymysql://user:pass@host:3306/dbname`. "
            "Use a **read-only** DB user for this if at all possible."
        )
        default_conn = os.environ.get("DB_CONNECTION_STRING", "")
        conn_str = st.text_input("Connection string", value=default_conn, type="password")
        if st.button("Connect", use_container_width=True) and conn_str:
            try:
                dm.connect_database(conn_str)
                rebuild_agent()
                st.success(f"Connected ({dm.db_dialect}).")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

    st.divider()
    st.header("⚙️ Settings")
    st.session_state.privacy_mode = st.toggle(
        "Mask sensitive columns (phone/CNIC/IMEI…)", value=st.session_state.privacy_mode
    )
    st.session_state.model_choice = st.text_input(
        "Ollama model", value=st.session_state.model_choice,
        help="Must be a tool-calling-capable model you've pulled, e.g. qwen3, qwen3:14b, llama3.1",
    )
    st.session_state.use_thinking = st.toggle(
        "Enable model 'thinking' mode", value=st.session_state.use_thinking,
        help="Turn off if your chosen model doesn't support thinking mode.",
    )
    if st.button("Apply settings", use_container_width=True):
        rebuild_agent()
        st.success("Agent reloaded with new settings.")

    st.divider()
    st.header("🗂️ Available tables")
    tables = dm.list_tables()
    if not tables:
        st.caption("No data loaded yet.")
    else:
        for name, info in tables.items():
            label = f"{name} (~{info['rows']} rows)" if info["rows"] is not None else name
            with st.expander(label):
                try:
                    schema = dm.get_schema(name)
                    st.dataframe(pd.DataFrame(schema["sample_rows"]), use_container_width=True)
                except Exception as exc:
                    st.caption(f"Preview unavailable: {exc}")

# ---------- Main chat ----------
st.title("📊 Telecom AI Data Analyst")
st.caption("Ask questions about your data in plain English. All processing stays on this machine.")

if not dm.list_tables():
    st.info("Upload a CSV/Excel file or connect a database in the sidebar to get started.")

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        render_turn(turn)

question = st.chat_input("e.g. Which region had the highest churn last quarter?")
if question:
    if not dm.list_tables():
        st.warning("Load a data source first.")
    else:
        if st.session_state.agent is None:
            rebuild_agent()

        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    result = st.session_state.agent.ask(question)
                except Exception as exc:
                    hint = ""
                    low = str(exc).lower()
                    if "connection" in low or "refused" in low:
                        hint = (
                            " (Is Ollama running? Try `ollama serve` in a terminal, and make "
                            f"sure you've pulled the model with `ollama pull {st.session_state.model_choice}`.)"
                        )
                    result = {"answer": f"Something went wrong: {exc}{hint}",
                              "trace": [], "chart": None, "dataframe": None}
            render_turn({"content": result["answer"], **result})

        st.session_state.chat.append({
            "role": "assistant", "content": result["answer"],
            "trace": result["trace"], "chart": result["chart"], "dataframe": result["dataframe"],
        })
