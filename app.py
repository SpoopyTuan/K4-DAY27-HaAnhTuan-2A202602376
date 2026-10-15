"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import threading
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path, db_conn
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url, MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
            FROM audit_events GROUP BY thread_id, pr_url
            ORDER BY MAX(timestamp) DESC LIMIT 25
        """) as cur:
            return [dict(row) for row in await cur.fetchall()]


with st.sidebar:
    st.header("Recent sessions")
    sessions = asyncio.run(recent_sessions())
    if sessions:
        table = st.dataframe(
            sessions, hide_index=True, on_select="rerun", selection_mode="single-row",
            key="recent_sessions",
        )
        if table.selection.rows:
            selected = sessions[table.selection.rows[0]]
            st.caption("Selected thread ID")
            st.code(selected["thread_id"], language=None)
            if st.button("Open selected session"):
                st.session_state.pop("graph_error", None)
                st.session_state.thread_id = selected["thread_id"]
                st.session_state.pr_url = selected["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.session_state.restore_session = True
                st.query_params["thread"] = selected["thread_id"]
    else:
        st.caption("No sessions yet.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    st.caption("Edit records feedback without posting; automatic rewriting is not implemented in this lab graph.")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {
            question: st.text_input(
                question, key=f"escalation_{st.session_state.thread_id}_{index}",
            )
            for index, question in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers"):
            if any(not answer.strip() for answer in answers.values()):
                st.warning("Please answer every question before submitting.")
            else:
                return {question: answer.strip() for question, answer in answers.items()}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        saved = await app.aget_state(cfg)
        if saved.values and saved.values.get("pr_url") != pr_url:
            raise ValueError("The saved thread belongs to a different PR URL")
        if saved.values.get("final_action") and not saved.next:
            return saved.values
        if resume_value is not None:
            if not saved.interrupts:
                raise ValueError("This session is not waiting for a response")
            return await app.ainvoke(Command(resume=resume_value), cfg)
        if saved.values:
            if saved.interrupts:
                return {"__interrupt__": saved.interrupts}
            if saved.next:
                return await app.ainvoke(None, cfg)
            return saved.values
        return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


@st.cache_resource
def session_locks():
    return {}, threading.Lock()


def invoke_graph(pr_url: str, thread_id: str, resume_value=None):
    locks, registry_lock = session_locks()
    with registry_lock:
        lock = locks.setdefault(thread_id, threading.Lock())
    if not lock.acquire(blocking=False):
        st.warning("This session is already running. Wait for it to finish, then reload.")
        st.stop()
    try:
        return asyncio.run(run_graph(pr_url, thread_id, resume_value))
    except Exception as error:
        import logging
        logging.getLogger(__name__).exception("Review failed for thread %s", thread_id)
        st.session_state.graph_error = type(error).__name__
        st.rerun()
    finally:
        lock.release()


def store_result(result):
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


requested_thread = st.query_params.get("thread")
if requested_thread and st.session_state.thread_id is None:
    match = next((row for row in sessions if row["thread_id"] == requested_thread), None)
    if match:
        st.session_state.thread_id = requested_thread
        st.session_state.pr_url = match["pr_url"]
        st.session_state.restore_session = True

if st.session_state.get("graph_error") and not submitted:
    st.error(f"Review failed ({st.session_state.graph_error}). Check the server terminal for details.")
    st.caption("Your session is saved. Retry continues from the saved step.")
    if st.button("Retry saved step", key="retry_saved_step"):
        st.session_state.pop("graph_error", None)
        st.session_state.restore_session = True
        st.rerun()
    st.stop()

if st.session_state.get("restore_session"):
    store_result(invoke_graph(st.session_state.pr_url, st.session_state.thread_id))
    st.session_state.restore_session = False
    st.rerun()


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted:
    st.session_state.pop("graph_error", None)
    pr_url = pr_url.strip()
    try:
        parse_pr_url(pr_url)
    except ValueError:
        st.error("Enter a valid GitHub pull request URL.")
        st.stop()
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.pop("approval_feedback", None)
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        result = invoke_graph(pr_url, st.session_state.thread_id)

    store_result(result)
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = invoke_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            )
        store_result(result)
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action in {"committed", "auto_committed", "auto_approved", "committed_after_escalation"}:
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "auto_skipped":
        st.info("Automatic review completed; posting is temporarily disabled.")
    elif action == "commit_failed" or action == "auto_commit_failed":
        st.error("The review could not be posted. Check the server terminal.")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    if final.get("analysis"):
        st.write(f"Confidence: {final['analysis'].confidence:.0%}")
        st.markdown(final["analysis"].summary)
    st.link_button("View PR on GitHub", st.session_state.pr_url)
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
