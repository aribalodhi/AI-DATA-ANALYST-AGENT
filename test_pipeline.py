"""
Test suite: exercises every module against sample data, plus a mocked
LLM response to prove the Agent's tool-calling loop is wired correctly -
all without needing a live Ollama server running. Run with:

    python3 test_pipeline.py
"""
import sys
import traceback
import pandas as pd

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


# ---------- DataManager ----------
from core.data_manager import DataManager

dm = DataManager()


def t_load_csv():
    name = dm.load_csv(open("sample_data/sample_telecom_usage.csv", "rb"), "sample_telecom_usage.csv")
    assert name == "sample_telecom_usage", name
    assert dm.list_tables()[name]["rows"] == 15


check("DataManager.load_csv", t_load_csv)


def t_reupload_replaces():
    before = dm.list_tables()["sample_telecom_usage"]["rows"]
    dm.load_csv(open("sample_data/sample_telecom_usage.csv", "rb"), "sample_telecom_usage.csv")
    tables = dm.list_tables()
    assert len(tables) == 1, f"expected 1 table after re-upload, got {len(tables)}"
    assert tables["sample_telecom_usage"]["rows"] == before


check("DataManager re-upload replaces (no duplicate tables)", t_reupload_replaces)


def t_get_schema():
    schema = dm.get_schema("sample_telecom_usage")
    col_names = {c["column_name"] for c in schema["columns"]}
    assert {"msisdn", "region", "churned", "monthly_revenue"} <= col_names, col_names
    assert len(schema["sample_rows"]) == 3


check("DataManager.get_schema", t_get_schema)


def t_run_sql():
    df = dm.run_sql(
        "SELECT region, COUNT(*) n, AVG(monthly_revenue) avg_rev FROM sample_telecom_usage GROUP BY region ORDER BY n DESC",
        max_rows=500,
    )
    assert len(df) > 0
    assert "avg_rev" in df.columns


check("DataManager.run_sql (aggregation)", t_run_sql)


def t_row_cap():
    df = dm.run_sql("SELECT * FROM sample_telecom_usage", max_rows=3)
    assert len(df) == 3, len(df)


check("DataManager.run_sql (row cap)", t_row_cap)

# ---------- sql_guard ----------
from core.sql_guard import ensure_read_only, UnsafeSQLError


def t_guard_allows_select():
    ensure_read_only("SELECT * FROM sample_telecom_usage WHERE churned = 1")
    ensure_read_only("WITH x AS (SELECT 1 AS a) SELECT * FROM x")


check("sql_guard allows SELECT/WITH", t_guard_allows_select)


def t_guard_blocks_write():
    for bad in [
        "DROP TABLE sample_telecom_usage",
        "DELETE FROM sample_telecom_usage",
        "UPDATE sample_telecom_usage SET churned = 0",
        "SELECT 1; DROP TABLE sample_telecom_usage;",
        "INSERT INTO sample_telecom_usage VALUES (1)",
    ]:
        try:
            ensure_read_only(bad)
            raise AssertionError(f"should have blocked: {bad}")
        except UnsafeSQLError:
            pass


check("sql_guard blocks writes + statement stacking", t_guard_blocks_write)

# ---------- sandbox ----------
from core.sandbox import run_python, SandboxTimeoutError


def t_sandbox_basic():
    dfs = {"sample_telecom_usage": dm.run_sql("SELECT * FROM sample_telecom_usage", 1000)}
    out = run_python(
        "result = sample_telecom_usage.groupby('region')['monthly_revenue'].mean().sort_values(ascending=False)",
        dfs, timeout_seconds=5,
    )
    assert out["error"] is None, out["error"]
    assert isinstance(out["result"], pd.Series)


check("sandbox runs pandas code", t_sandbox_basic)


def t_sandbox_blocks_import():
    out = run_python("import os\nresult = os.getcwd()", {}, timeout_seconds=5)
    assert out["error"] is not None, "expected an error but sandbox allowed `import os`"


check("sandbox blocks disallowed imports", t_sandbox_blocks_import)


def t_sandbox_timeout():
    try:
        run_python("while True:\n    pass", {}, timeout_seconds=2)
        raise AssertionError("expected SandboxTimeoutError")
    except SandboxTimeoutError:
        pass


check("sandbox enforces timeout", t_sandbox_timeout)

# ---------- charts ----------
from core.charts import build_chart


def t_chart():
    df = dm.run_sql("SELECT region, COUNT(*) n FROM sample_telecom_usage GROUP BY region", 500)
    fig = build_chart(df, "bar", x="region", y="n", title="Subscribers by region")
    assert fig is not None


check("charts.build_chart", t_chart)

# ---------- privacy ----------
from utils.privacy import mask_dataframe, is_sensitive_column


def t_privacy():
    assert is_sensitive_column("msisdn")
    assert not is_sensitive_column("region")
    df = dm.run_sql("SELECT msisdn, region FROM sample_telecom_usage LIMIT 3", 500)
    masked = mask_dataframe(df)
    assert all(masked["msisdn"].str.contains(r"\*"))
    assert masked["region"].equals(df["region"])  # non-sensitive column untouched


check("privacy.mask_dataframe masks only sensitive columns", t_privacy)

# ---------- tools.ToolExecutor end-to-end (no LLM needed) ----------
from core.tools import ToolExecutor


def t_tool_executor():
    ex = ToolExecutor(dm, privacy_mode=True)
    out = ex.dispatch("list_tables", {})
    assert "sample_telecom_usage" in out

    out = ex.dispatch("get_schema", {"table_name": "sample_telecom_usage"})
    assert "msisdn" in out

    out = ex.dispatch("execute_sql", {"query": "SELECT region, msisdn FROM sample_telecom_usage LIMIT 2"})
    assert "row(s) returned" in out
    assert "923001234567" not in out  # masked in privacy mode

    out = ex.dispatch("execute_python", {
        "code": "result = sample_telecom_usage['monthly_revenue'].sum()"
    })
    assert "result" in out

    out = ex.dispatch("create_chart", {
        "query": "SELECT region, COUNT(*) n FROM sample_telecom_usage GROUP BY region",
        "chart_type": "bar", "x": "region", "y": "n",
    })
    assert ex.last_chart is not None

    out = ex.dispatch("execute_sql", {"query": "DROP TABLE sample_telecom_usage"})
    assert out.startswith("Error:")


check("ToolExecutor end-to-end dispatch", t_tool_executor)

# ---------- Agent loop with a mocked Ollama response (no live server needed) ----------
import ollama
from ollama._types import Message
from core.agent import Agent


def t_agent_loop_mocked():
    dm2 = DataManager()
    dm2.load_csv(open("sample_data/sample_telecom_usage.csv", "rb"), "sample_telecom_usage.csv")

    call_count = {"n": 0}

    def fake_chat(self, model, messages, tools=None, think=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the model deciding to call execute_sql
            tool_call = Message.ToolCall(
                function=Message.ToolCall.Function(
                    name="execute_sql",
                    arguments={"query": "SELECT region, AVG(monthly_revenue) avg_rev FROM sample_telecom_usage GROUP BY region ORDER BY avg_rev DESC LIMIT 1"},
                )
            )
            msg = Message(role="assistant", content="", tool_calls=[tool_call])
        else:
            # Simulate the model giving a final answer after seeing the tool result
            msg = Message(role="assistant", content="Islamabad has the highest average monthly revenue.")

        class FakeResponse:
            pass
        r = FakeResponse()
        r.message = msg
        return r

    ollama.Client.chat = fake_chat  # monkey-patch for this test only
    agent = Agent(dm2, model="qwen3", privacy_mode=True)
    result = agent.ask("Which region has the highest average revenue?")

    assert call_count["n"] == 2, f"expected 2 round-trips, got {call_count['n']}"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "execute_sql"
    assert "Islamabad" in result["answer"]


check("Agent tool-calling loop (mocked LLM)", t_agent_loop_mocked)

# ---------- report ----------
print("\n" + "=" * 60)
failed = 0
for name, ok, err in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        failed += 1
        print(err)
print("=" * 60)
print(f"{len(results) - failed}/{len(results)} passed")
sys.exit(1 if failed else 0)
