"""Streamlit leaderboard for SSLM benchmark results.

Launch via:  sslm dashboard
         or: streamlit run src/sslm/sslm/dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sslm.playground.benchmarks import TASK_DISPLAY_NAME, parse_lm_eval_results

# Open LLM Leaderboard v2 column order (shown first when present).
OPEN_LLM_V2_DISPLAY = ["IFEval", "BBH", "MATH", "GPQA*", "MuSR", "MMLU-Pro"]

DEFAULT_RESULTS_DIR = Path(".data/sslm/results")


def _load_smoke(results_dir: Path) -> pd.DataFrame:
    rows = []
    for jsonl in sorted(results_dir.rglob("smoke.jsonl")):
        try:
            model_key = jsonl.relative_to(results_dir).parts[0]
        except ValueError:
            model_key = jsonl.parent.name
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                rec["model"] = model_key
                rows.append(rec)
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the latest run per (model, task) by sorting on date descending.
    df = df.sort_values("date", ascending=False)
    df = df.drop_duplicates(subset=["model", "task"], keep="first")

    df["score_pct"] = (df["score"] * 100).round(1)
    pivot = df.pivot_table(
        index="model",
        columns="task_display",
        values="score_pct",
        aggfunc="first",
    )

    # Put Open LLM v2 columns first, then alphabetical remainder.
    v2_cols = [c for c in OPEN_LLM_V2_DISPLAY if c in pivot.columns]
    other_cols = sorted(c for c in pivot.columns if c not in OPEN_LLM_V2_DISPLAY)
    pivot = pivot[v2_cols + other_cols]

    pivot.insert(0, "Avg", pivot.mean(axis=1).round(1))
    return pivot.sort_values("Avg", ascending=False)


def main() -> None:
    st.set_page_config(
        page_title="SSLM Leaderboard",
        page_icon="=",
        layout="wide",
    )

    st.title("SSLM Reasoning Benchmark Leaderboard")

    with st.sidebar:
        st.header("Settings")
        results_dir = Path(
            st.text_input("Results directory", str(DEFAULT_RESULTS_DIR))
        )
        st.markdown("---")
        st.caption(
            "Scores are from lm-evaluation-harness results.json files "
            "found recursively under the results directory."
        )
        st.caption(
            "Run benchmarks with:\n"
            "```\nsslm sequential --suite open_llm_v2\n```"
        )

    if not results_dir.exists():
        st.warning(f"Results directory not found: `{results_dir}`")
        st.info(
            "Run benchmarks first:\n"
            "```bash\nsslm sequential --suite open_llm_v2\n```"
        )
        return

    rows = parse_lm_eval_results(results_dir)
    smoke_df = _load_smoke(results_dir)

    if not rows and smoke_df.empty:
        st.warning("No results found yet.")
        return

    # --- Leaderboard table ---
    if rows:
        df = pd.DataFrame(rows)
        board = _leaderboard(df)

        st.subheader("Leaderboard")
        st.caption(
            "Columns: Open LLM Leaderboard v2 tasks (IFEval, BBH, MATH, GPQA*, MuSR, MMLU-Pro) "
            "+ any additional tasks run. Scores are percentages (0-100). "
            "GPQA* = GPQA Diamond."
        )

        styled = board.style.format("{:.1f}", na_rep="-").background_gradient(
            cmap="RdYlGn", axis=None, subset=board.columns
        )
        st.dataframe(styled, use_container_width=True)

        # Per-task bar chart
        st.subheader("Score breakdown")
        task_cols = [c for c in board.columns if c != "Avg"]
        if task_cols:
            chart_data = board[task_cols].T
            st.bar_chart(chart_data, use_container_width=True)

        # Run history
        with st.expander("Run history (all result files)"):
            history = (
                df[["model", "task", "task_display", "metric", "score", "date", "result_file"]]
                .sort_values(["model", "date"], ascending=[True, False])
                .reset_index(drop=True)
            )
            history["score"] = (history["score"] * 100).round(2)
            st.dataframe(history, use_container_width=True)

    # --- Smoke results ---
    if not smoke_df.empty:
        st.subheader("Smoke tests")
        cols = [c for c in ["model", "prompt_name", "ok", "latency_s", "response_text"] if c in smoke_df.columns]
        st.dataframe(smoke_df[cols], use_container_width=True)


if __name__ == "__main__":
    main()
