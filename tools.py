"""
Tool definitions the agent can call, in Ollama/OpenAI function-calling
format, plus a dispatcher that executes them against the active
DataManager. This is the whole "AI agent + tool calling" surface area —
the LLM never touches data directly, it only ever sees what these
functions choose to return.
"""

from __future__ import annotations
import pandas as pd

from core.sql_guard import ensure_read_only, UnsafeSQLError
from core.sandbox import run_python, SandboxTimeoutError
from core.charts import build_chart
import config

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List every table/data source currently available to query, with row "
                "counts if known. Always call this first if you don't already know "
                "what data is available."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Get the column names, types, and a few sample rows for one table. "
                "Call this before writing SQL/Python against a table you haven't "
                "inspected yet in this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Exact table name from list_tables."},
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Run a single read-only SQL SELECT (or WITH ... SELECT) query against "
                "the active data source and return the resulting rows. Use this for "
                f"filtering, aggregation, joins, and sorting. Results are capped at "
                f"{config.MAX_SQL_ROWS_RETURNED} rows — aggregate or add LIMIT yourself "
                "for large tables rather than relying on the cap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Run pandas/numpy code for analysis that's awkward in SQL (forecasting, "
                "statistics, multi-step transforms). Every loaded table is available as "
                "a DataFrame variable named after the table. Assign your final answer "
                "to a variable called `result`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code. Must set a `result` variable."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Render a chart from a SQL query's results. Pick chart_type based on "
                "what best answers the question: trend over time -> line, comparison "
                "across categories -> bar, share of a whole -> pie, relationship "
                "between two numeric columns -> scatter, distribution -> histogram."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SELECT statement producing the data to chart."},
                    "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "pie", "histogram"]},
                    "x": {"type": "string", "description": "Column for the x-axis / category / pie names."},
                    "y": {"type": "string", "description": "Column for the y-axis / values. Omit for histogram."},
                    "color": {"type": "string", "description": "Optional column to split/color series by."},
                    "title": {"type": "string", "description": "Chart title."},
                },
                "required": ["query", "chart_type"],
            },
        },
    },
]


class ToolExecutor:
    """Binds the tool schemas above to a live DataManager + privacy setting."""

    def __init__(self, data_manager, privacy_mode: bool = True):
        self.dm = data_manager
        self.privacy_mode = privacy_mode
        self.last_chart = None
        self.last_dataframe: pd.DataFrame | None = None

    def dispatch(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name == "list_tables":
                return self._list_tables()
            if name == "get_schema":
                return self._get_schema(**args)
            if name == "execute_sql":
                return self._execute_sql(**args)
            if name == "execute_python":
                return self._execute_python(**args)
            if name == "create_chart":
                return self._create_chart(**args)
            return f"Error: unknown tool '{name}'."
        except (UnsafeSQLError, SandboxTimeoutError, ValueError, RuntimeError, TypeError) as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures to the model, not a crash
            return f"Error: unexpected failure running {name}: {exc}"

    def _maybe_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.privacy_mode:
            return df
        from utils.privacy import mask_dataframe
        return mask_dataframe(df)

    def _list_tables(self) -> str:
        tables = self.dm.list_tables()
        if not tables:
            return "No data loaded yet. Ask the user to upload a file or connect a database."
        lines = [
            f"- {name} (~{info['rows']} rows)" if info["rows"] is not None else f"- {name}"
            for name, info in tables.items()
        ]
        return "Available tables:\n" + "\n".join(lines)

    def _get_schema(self, table_name: str) -> str:
        schema = self.dm.get_schema(table_name)
        cols = ", ".join(f"{c['column_name']} ({c['column_type']})" for c in schema["columns"])
        return f"Columns for {table_name}: {cols}\nSample rows: {schema['sample_rows']}"

    def _execute_sql(self, query: str) -> str:
        ensure_read_only(query)
        df = self.dm.run_sql(query, max_rows=config.MAX_SQL_ROWS_RETURNED)
        self.last_dataframe = df
        shown = self._maybe_mask(df)
        return f"{len(df)} row(s) returned.\n{shown.to_markdown(index=False)}"

    def _execute_python(self, code: str) -> str:
        dataframes = self.dm.get_dataframes_namespace()
        outcome = run_python(code, dataframes, timeout_seconds=config.SANDBOX_TIMEOUT_SECONDS)
        if outcome["error"]:
            return f"Error while running code: {outcome['error']}\nOutput so far:\n{outcome['stdout']}"

        result = outcome["result"]
        if isinstance(result, pd.DataFrame):
            self.last_dataframe = result
            result_repr = self._maybe_mask(result).to_markdown(index=False)
        elif isinstance(result, pd.Series):
            result_repr = result.to_string()
        else:
            result_repr = repr(result)
        return f"stdout:\n{outcome['stdout']}\nresult:\n{result_repr}"

    def _create_chart(self, query: str, chart_type: str, x: str | None = None,
                       y: str | None = None, color: str | None = None,
                       title: str | None = None) -> str:
        ensure_read_only(query)
        df = self.dm.run_sql(query, max_rows=config.MAX_SQL_ROWS_RETURNED)
        fig = build_chart(df, chart_type, x=x, y=y, color=color, title=title)
        self.last_chart = fig
        self.last_dataframe = df
        return f"Chart rendered from {len(df)} row(s). Describe the key insight in your final answer."
