from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import requests
import streamlit as st

APP_TITLE = "Abysalto · Document QA"
DEFAULT_API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SEC", "60"))
ALLOWED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff"]
DEFAULT_TOP_K = 5
MAX_CHAT_TURNS = 12

STATUS_ICON = {
    "uploaded": "📄",
    "processing": "⏳",
    "indexed": "✅",
    "failed": "❌",
    "error": "⚠️",
    "deleted": "🗑️",
}


def _default_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


@dataclass(slots=True)
class ApiResult:
    ok: bool
    status_code: int
    data: Any | None
    text: str
    headers: dict[str, str]
    error_code: str | None = None
    error_message: str | None = None


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


CSS = """
<style>
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
.ab-card {
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 14px;
    padding: 1rem 1rem 0.9rem 1rem;
    background: rgba(255, 255, 255, 0.02);
    margin-bottom: 0.8rem;
}
.ab-card-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.ab-card-subtle {
    color: rgba(49, 51, 63, 0.7);
    font-size: 0.92rem;
}
.ab-source {
    border-left: 3px solid rgba(49, 51, 63, 0.28);
    padding-left: 0.8rem;
    margin: 0.75rem 0;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.title(APP_TITLE)
st.caption(
    "A professional workspace for uploading, reviewing, and asking questions across your documents."
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "http_session": _default_requests_session(),
        "api_base_url": DEFAULT_API_BASE_URL,
        "request_timeout": DEFAULT_TIMEOUT_SECONDS,
        "auth_token": "",
        "current_user": None,
        "current_identity": None,
        "documents_cache": [],
        "document_detail_cache": {},
        "selected_doc_ids": [],
        "active_doc_id": None,
        "last_upload_response": None,
        "qa_history": [],
        "auth_view": "login",
        "ask_scope_mode": "selected_docs",
        "ask_top_k": DEFAULT_TOP_K,
        "bootstrap_complete": False,
        "needs_refresh": True,
        "hard_refresh": True,
        "flash_message": None,
        "ask_form_seed": 0,
        "scope_doc_multiselect": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


Level = Literal["success", "info", "warning", "error"]


def flash(level: Level, message: str) -> None:
    st.session_state.flash_message = {"level": level, "message": message}


def render_flash() -> None:
    payload = st.session_state.pop("flash_message", None)
    if not payload:
        return
    level = payload.get("level", "info")
    message = payload.get("message", "")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


render_flash()


def get_session() -> requests.Session:
    return st.session_state.http_session


def api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    include_auth: bool = True,
) -> ApiResult:
    url = f"{st.session_state.api_base_url.rstrip('/')}{path}"
    headers: dict[str, str] = {}
    if include_auth and st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token.strip()}"

    try:
        response = get_session().request(
            method=method.upper(),
            url=url,
            json=json_body,
            files=files,
            headers=headers,
            timeout=int(st.session_state.request_timeout),
        )
    except requests.RequestException as exc:
        return ApiResult(
            ok=False,
            status_code=0,
            data=None,
            text=str(exc),
            headers={},
            error_code="connection_error",
            error_message=f"API request failed: {exc}",
        )

    try:
        data = response.json()
    except ValueError:
        data = None

    payload = data if isinstance(data, dict) else {}
    return ApiResult(
        ok=response.ok,
        status_code=response.status_code,
        data=data,
        text=response.text,
        headers=dict(response.headers),
        error_code=payload.get("error_code"),
        error_message=payload.get("message"),
    )


def format_api_error(result: ApiResult) -> str:
    return result.error_message or result.text or "Request failed."


def clear_document_state(
    *, clear_upload_result: bool = True, clear_history: bool = False
) -> None:
    st.session_state.documents_cache = []
    st.session_state.document_detail_cache = {}
    st.session_state.selected_doc_ids = []
    st.session_state.active_doc_id = None
    if clear_upload_result:
        st.session_state.last_upload_response = None
    if clear_history:
        st.session_state.qa_history = []


def reset_anonymous_session() -> None:
    st.session_state.http_session = _default_requests_session()
    st.session_state.auth_token = ""
    st.session_state.current_user = None
    st.session_state.current_identity = None
    clear_document_state(clear_upload_result=True, clear_history=True)
    queue_refresh(hard=True)


def queue_refresh(*, hard: bool = False) -> None:
    st.session_state.needs_refresh = True
    st.session_state.hard_refresh = bool(st.session_state.hard_refresh or hard)


def sync_identity_state() -> None:
    identity_result = api_request(
        "GET",
        "/auth/identity",
        include_auth=bool(st.session_state.auth_token),
    )
    st.session_state.current_identity = (
        identity_result.data if identity_result.ok else None
    )

    if st.session_state.auth_token:
        me_result = api_request("GET", "/auth/me")
        if me_result.ok:
            st.session_state.current_user = me_result.data
        else:
            st.session_state.auth_token = ""
            st.session_state.current_user = None
            st.session_state.current_identity = None
            flash("warning", "Your sign-in session expired. You are continuing as a guest.")
    else:
        st.session_state.current_user = None


def refresh_documents(*, clear_detail_cache: bool = True) -> None:
    result = api_request("GET", "/documents")
    if not result.ok or not isinstance(result.data, dict):
        flash("warning", f"Unable to load documents: {format_api_error(result)}")
        return

    documents = result.data.get("documents", [])
    st.session_state.documents_cache = documents

    allowed_ids = {doc.get("doc_id") for doc in documents if doc.get("doc_id")}
    ready_ids = {
        doc.get("doc_id")
        for doc in documents
        if doc.get("doc_id") and bool(doc.get("ready_to_ask"))
    }
    st.session_state.selected_doc_ids = [
        doc_id for doc_id in st.session_state.selected_doc_ids if doc_id in ready_ids
    ]

    if clear_detail_cache:
        st.session_state.document_detail_cache = {}
    else:
        st.session_state.document_detail_cache = {
            doc_id: detail
            for doc_id, detail in st.session_state.document_detail_cache.items()
            if doc_id in allowed_ids
        }

    if st.session_state.active_doc_id not in allowed_ids:
        st.session_state.active_doc_id = next(iter(allowed_ids), None)


def refresh_document_detail(doc_id: str, *, force: bool = False) -> dict[str, Any] | None:
    if not doc_id:
        return None
    if not force and doc_id in st.session_state.document_detail_cache:
        return st.session_state.document_detail_cache[doc_id]

    result = api_request("GET", f"/documents/{doc_id}")
    if result.ok and isinstance(result.data, dict):
        st.session_state.document_detail_cache[doc_id] = result.data
        return result.data
    flash("warning", f"Unable to load document details: {format_api_error(result)}")
    return None


def ensure_bootstrap() -> None:
    if not st.session_state.bootstrap_complete or st.session_state.needs_refresh:
        sync_identity_state()
        refresh_documents(clear_detail_cache=bool(st.session_state.hard_refresh))
        st.session_state.bootstrap_complete = True
        st.session_state.needs_refresh = False
        st.session_state.hard_refresh = False


ensure_bootstrap()


def status_label(status: str | None) -> str:
    normalized = (status or "unknown").lower()
    return f"{STATUS_ICON.get(normalized, '•')} {normalized}"


def format_datetime(value: str | None) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "—"
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"


def documents() -> list[dict[str, Any]]:
    return st.session_state.documents_cache


def ready_documents() -> list[dict[str, Any]]:
    return [doc for doc in documents() if doc.get("ready_to_ask")]


def ready_doc_ids() -> list[str]:
    return [doc["doc_id"] for doc in ready_documents() if doc.get("doc_id")]


def selected_ready_doc_ids() -> list[str]:
    allowed = set(ready_doc_ids())
    return [doc_id for doc_id in st.session_state.selected_doc_ids if doc_id in allowed]


def selected_ready_documents() -> list[dict[str, Any]]:
    selected = set(selected_ready_doc_ids())
    return [doc for doc in ready_documents() if doc.get("doc_id") in selected]


def document_lookup() -> dict[str, dict[str, Any]]:
    return {doc["doc_id"]: doc for doc in documents() if doc.get("doc_id")}


def ready_document_options() -> dict[str, str]:
    return {
        doc["doc_id"]: f"{doc.get('filename') or doc['doc_id']} · {status_label(doc.get('status'))}"
        for doc in ready_documents()
        if doc.get("doc_id")
    }


def current_identity_label() -> str:
    if st.session_state.current_user:
        return str(st.session_state.current_user.get("email") or "Signed in")
    identity = st.session_state.current_identity or {}
    session_id = identity.get("session_id")
    if session_id:
        return f"Guest session · {str(session_id)[:12]}…"
    return "Guest session"


def render_connection_panel() -> None:
    with st.sidebar.expander("Connection and API", expanded=False):
        st.text_input("API base URL", key="api_base_url")
        st.number_input(
            "Request timeout (sec)",
            min_value=5,
            max_value=300,
            key="request_timeout",
        )
        if st.button("Reload data", use_container_width=True):
            queue_refresh(hard=True)
            st.rerun()


def render_auth_panel() -> None:
    with st.sidebar:
        st.subheader("Account and session")
        if st.session_state.current_user:
            st.success(f"Signed in as {st.session_state.current_user.get('email')}")
        else:
            st.info("You are currently using a guest session.")

        identity = st.session_state.current_identity or {}
        st.caption(f"Mode: {identity.get('kind', 'unknown')}")
        if identity.get("session_id"):
            st.caption(f"Session ID: {identity['session_id'][:12]}…")

        tabs = st.tabs(["Sign in", "Register"])
        with tabs[0]:
            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                login_submitted = st.form_submit_button(
                    "Sign in", use_container_width=True
                )
            if login_submitted:
                result = api_request(
                    "POST",
                    "/auth/login",
                    json_body={"email": login_email, "password": login_password},
                    include_auth=False,
                )
                if result.ok and isinstance(result.data, dict):
                    st.session_state.auth_token = result.data.get("access_token", "")
                    clear_document_state(clear_upload_result=True, clear_history=True)
                    queue_refresh(hard=True)
                    flash(
                        "success",
                        "Sign-in successful. Documents and ownership state were refreshed.",
                    )
                    st.rerun()
                else:
                    flash("error", format_api_error(result))
                    st.rerun()

        with tabs[1]:
            with st.form("register_form", clear_on_submit=False):
                register_email = st.text_input("Email", key="register_email")
                register_password = st.text_input(
                    "Password", type="password", key="register_password"
                )
                register_submitted = st.form_submit_button(
                    "Create account", use_container_width=True
                )
            if register_submitted:
                register_result = api_request(
                    "POST",
                    "/auth/register",
                    json_body={"email": register_email, "password": register_password},
                    include_auth=False,
                )
                if register_result.ok:
                    login_result = api_request(
                        "POST",
                        "/auth/login",
                        json_body={"email": register_email, "password": register_password},
                        include_auth=False,
                    )
                    if login_result.ok and isinstance(login_result.data, dict):
                        st.session_state.auth_token = login_result.data.get(
                            "access_token", ""
                        )
                        clear_document_state(clear_upload_result=True, clear_history=True)
                        queue_refresh(hard=True)
                        flash(
                            "success",
                            "Your account was created and you were signed in automatically. Guest documents are now linked to your account.",
                        )
                    else:
                        flash(
                            "success",
                            "Your account was created. Sign in to continue under your user account.",
                        )
                    st.rerun()
                else:
                    flash("error", format_api_error(register_result))
                    st.rerun()

        action_cols = st.columns(2)
        if action_cols[0].button("Sign out", use_container_width=True):
            st.session_state.auth_token = ""
            clear_document_state(clear_upload_result=False, clear_history=True)
            queue_refresh(hard=True)
            flash("info", "You have signed out. A new guest session is now active.")
            st.rerun()
        if action_cols[1].button("Reset session", use_container_width=True):
            reset_anonymous_session()
            flash("info", "The session was reset and local state was cleared.")
            st.rerun()


def render_workspace_actions() -> None:
    with st.sidebar:
        st.divider()
        st.subheader("Workspace")
        if st.button("Clear last upload", use_container_width=True):
            st.session_state.last_upload_response = None
            flash("info", "The last upload result was cleared from the view.")
            st.rerun()
        if st.button("Clear Q&A history", use_container_width=True):
            st.session_state.qa_history = []
            flash("info", "The question and answer history was cleared.")
            st.rerun()


render_connection_panel()
render_auth_panel()
render_workspace_actions()


def render_summary_bar() -> None:
    docs = documents()
    ready = ready_documents()
    selected = selected_ready_doc_ids()
    metrics = st.columns(4)
    metrics[0].metric("Identity", "User" if st.session_state.current_user else "Session")
    metrics[1].metric("Documents", len(docs))
    metrics[2].metric("Ready for questions", len(ready))
    metrics[3].metric("Selected in scope", len(selected))
    st.caption(f"Active workspace context: {current_identity_label()}")


render_summary_bar()


workspace_tab, documents_tab = st.tabs(["Workspace", "Documents"])


def render_upload_panel() -> None:
    st.subheader("Upload")
    st.caption(
        "Uploads use a form so file-selection changes do not trigger unnecessary full-app reruns."
    )

    with st.form("upload_form", clear_on_submit=True):
        uploads = st.file_uploader(
            "Add one or more PDFs or images",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        upload_submitted = st.form_submit_button(
            "Upload and process", type="primary", use_container_width=True
        )

    if upload_submitted:
        if not uploads:
            flash("warning", "Select at least one document to upload.")
            st.rerun()

        files_payload = [
            ("files", (upload.name, upload.getvalue(), upload.type or "application/octet-stream"))
            for upload in uploads or []
        ]
        with st.status("Uploading and processing documents…", expanded=True) as status:
            st.write(f"Files: {len(files_payload)}")
            result = api_request("POST", "/upload", files=files_payload)
            if result.ok and isinstance(result.data, dict):
                st.session_state.last_upload_response = result.data
                refresh_documents(clear_detail_cache=True)
                ready_ids_from_upload = [
                    item.get("doc_id")
                    for item in result.data.get("documents", [])
                    if item.get("ready_to_ask") and item.get("doc_id")
                ]
                st.session_state.selected_doc_ids = list(
                    dict.fromkeys(selected_ready_doc_ids() + ready_ids_from_upload)
                )
                ready_count = sum(
                    1 for item in result.data.get("documents", []) if item.get("ready_to_ask")
                )
                status.update(label="Upload completed", state="complete")
                flash(
                    "success",
                    f"Upload completed. {ready_count} of {len(result.data.get('documents', []))} documents are ready for questions.",
                )
                st.rerun()
            else:
                status.update(label="Upload failed", state="error")
                flash("error", format_api_error(result))
                st.rerun()

    upload_response = st.session_state.last_upload_response
    if not upload_response:
        st.info("There is no recent upload result.")
        return

    items = upload_response.get("documents", [])
    ready_count = sum(1 for item in items if item.get("ready_to_ask"))
    st.caption(f"Latest upload: {ready_count}/{len(items)} documents ready for questions.")
    for item in items:
        detail = item.get("status_detail") or item.get("error_detail") or ""
        st.markdown(
            f"<div class='ab-card'><div class='ab-card-title'>{item.get('filename', 'file')}</div>"
            f"<div class='ab-card-subtle'>{status_label(item.get('status'))}</div>"
            f"<div class='ab-card-subtle'>{detail}</div></div>",
            unsafe_allow_html=True,
        )


def render_scope_panel() -> None:
    st.subheader("Question scope")
    ready = ready_documents()
    if not ready:
        st.info("There are no indexed documents yet. Upload and process a document first.")
        return

    options = ready_document_options()
    mode = st.radio(
        "Ask questions across",
        options=["selected_docs", "all_ready_docs"],
        key="ask_scope_mode",
        horizontal=False,
        format_func=lambda value: (
            "Only the selected ready documents"
            if value == "selected_docs"
            else "All ready documents"
        ),
    )

    if mode == "selected_docs":
        current_scope_widget_value = st.session_state.get(
            "scope_doc_multiselect", selected_ready_doc_ids()
        )
        st.session_state.scope_doc_multiselect = [
            doc_id for doc_id in current_scope_widget_value if doc_id in options
        ]
        selected = st.multiselect(
            "Select documents for questions",
            options=list(options.keys()),
            format_func=lambda doc_id: options[doc_id],
            key="scope_doc_multiselect",
        )
        st.session_state.selected_doc_ids = list(selected)
        if selected:
            st.caption(f"Active scope: {len(selected)} document(s).")
        else:
            st.warning(
                "No ready documents are selected. The next question cannot be submitted."
            )
    else:
        st.caption(f"Active scope: all {len(ready)} ready documents.")

    st.slider(
        "Number of retrieval sources",
        min_value=1,
        max_value=8,
        value=int(st.session_state.ask_top_k),
        key="ask_top_k",
        help="A higher value provides broader context, but may return more sources than the user needs to review.",
    )


def render_chat_history() -> None:
    history = st.session_state.qa_history
    if not history:
        st.info("There are no questions yet. Ask your first specific question below.")
        return

    for turn in history:
        with st.chat_message("user"):
            st.write(turn.get("question") or "")
            scope_label = turn.get("scope_label")
            if scope_label:
                st.caption(scope_label)

        with st.chat_message("assistant"):
            if turn.get("ok"):
                response = turn.get("response") or {}
                answer = response.get("answer") or ""
                grounded = bool(response.get("grounded", True))
                if grounded:
                    st.write(answer)
                else:
                    st.warning(answer)
                if response.get("message"):
                    st.caption(response["message"])

                meta_cols = st.columns(2)
                meta_cols[0].metric("Grounded", "yes" if grounded else "partial / weak")
                meta_cols[1].metric(
                    "Confidence",
                    response.get("confidence")
                    if response.get("confidence") is not None
                    else "—",
                )

                sources = response.get("sources") or []
                if sources:
                    with st.expander(f"Sources ({len(sources)})", expanded=False):
                        for source in sources:
                            st.markdown(
                                f"<div class='ab-source'><strong>{source.get('filename') or source.get('doc_id')}</strong>"
                                f" · page {source.get('page') or '—'} · score {source.get('score')}<br/>"
                                f"{source.get('text_excerpt') or ''}<br/>"
                                f"<span class='ab-card-subtle'>doc_id={source.get('doc_id')} · chunk_id={source.get('chunk_id')}"
                                f" · semantic={source.get('semantic_score')} · lexical={source.get('lexical_score')}</span></div>",
                                unsafe_allow_html=True,
                            )

                entities = response.get("entities") or []
                if entities:
                    with st.expander("Named entities", expanded=False):
                        st.dataframe(entities, use_container_width=True, hide_index=True)
            else:
                st.error(turn.get("error_message") or "The question failed.")


def ask_scope_payload() -> tuple[dict[str, Any] | None, str | None]:
    ready = ready_documents()
    if not ready:
        return None, "There are no ready documents available for search."

    scope_mode = st.session_state.ask_scope_mode
    payload: dict[str, Any] = {"top_k": int(st.session_state.ask_top_k)}

    if scope_mode == "selected_docs":
        selected = selected_ready_doc_ids()
        if not selected:
            return None, "Select at least one ready document for this scope."
        payload["scope"] = "docs"
        payload["doc_ids"] = selected
        names = [
            doc.get("filename") or doc.get("doc_id") for doc in selected_ready_documents()
        ]
        label = "Scope: " + ", ".join(names[:5]) + (
            f" and {len(names) - 5} more" if len(names) > 5 else ""
        )
        return payload, label

    payload["scope"] = "all"
    return payload, f"Scope: all {len(ready)} ready documents"


def submit_question(question: str) -> None:
    base_payload, scope_label = ask_scope_payload()
    if base_payload is None:
        flash("warning", scope_label or "The question cannot be submitted.")
        st.rerun()

    payload = {**base_payload, "question": question.strip()}
    with st.status("Searching documents and composing an answer…", expanded=True) as status:
        st.write(scope_label)
        result = api_request("POST", "/ask", json_body=payload)
        if result.ok and isinstance(result.data, dict):
            history = st.session_state.qa_history
            history.append(
                {
                    "question": question.strip(),
                    "scope_label": scope_label,
                    "response": result.data,
                    "ok": True,
                    "asked_at": datetime.utcnow().isoformat(),
                }
            )
            st.session_state.qa_history = history[-MAX_CHAT_TURNS:]
            status.update(label="Answer ready", state="complete")
            flash("success", "The answer was added to the question history.")
            st.session_state.ask_form_seed += 1
            st.rerun()
        else:
            st.session_state.qa_history.append(
                {
                    "question": question.strip(),
                    "scope_label": scope_label,
                    "ok": False,
                    "error_message": format_api_error(result),
                    "asked_at": datetime.utcnow().isoformat(),
                }
            )
            st.session_state.qa_history = st.session_state.qa_history[-MAX_CHAT_TURNS:]
            status.update(label="Question failed", state="error")
            flash("error", format_api_error(result))
            st.session_state.ask_form_seed += 1
            st.rerun()


with workspace_tab:
    left_col, right_col = st.columns([1.7, 1.0], gap="large")

    with right_col:
        render_upload_panel()
        st.divider()
        render_scope_panel()

    with left_col:
        st.subheader("Q&A workspace")
        st.caption(
            "Question history remains visible even after changing scope, document selection, or refreshing the library."
        )
        render_chat_history()

        question_help = ask_scope_payload()[1] or ""
        with st.form(f"ask_form_{st.session_state.ask_form_seed}", clear_on_submit=True):
            question = st.text_area(
                "New question",
                height=110,
                placeholder="Ask a specific question based on the document content.",
            )
            ask_submitted = st.form_submit_button(
                "Submit question", type="primary", use_container_width=True
            )
            if question_help:
                st.caption(question_help)

        if ask_submitted:
            if not question.strip():
                flash("warning", "The question cannot be empty.")
                st.rerun()
            submit_question(question)


def render_document_table() -> None:
    docs = documents()
    if not docs:
        st.info("There are no documents yet. Upload one to get started.")
        return

    rows = []
    selected = set(selected_ready_doc_ids())
    for doc in docs:
        rows.append(
            {
                "in_scope": "yes" if doc.get("doc_id") in selected else "no",
                "filename": doc.get("filename"),
                "status": status_label(doc.get("status")),
                "ready": "yes" if doc.get("ready_to_ask") else "no",
                "pages": doc.get("pages") or "—",
                "chunks": doc.get("chunks") or "—",
                "size": format_size(doc.get("size_bytes")),
                "owner": doc.get("owner_type"),
                "created": format_datetime(doc.get("created_at")),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_document_detail_panel() -> None:
    docs = documents()
    if not docs:
        return

    doc_options = {
        doc["doc_id"]: f"{doc.get('filename')} · {status_label(doc.get('status'))}"
        for doc in docs
        if doc.get("doc_id")
    }
    doc_ids = list(doc_options.keys())
    if not doc_ids:
        return

    if st.session_state.active_doc_id not in doc_options:
        st.session_state.active_doc_id = doc_ids[0]

    active_doc_id = st.selectbox(
        "Document details",
        options=doc_ids,
        index=doc_ids.index(st.session_state.active_doc_id),
        format_func=lambda doc_id: doc_options[doc_id],
    )
    st.session_state.active_doc_id = active_doc_id

    detail = refresh_document_detail(active_doc_id, force=False)
    if not detail:
        return

    cols = st.columns([2, 1, 1, 1])
    cols[0].markdown(f"**{detail.get('filename')}**")
    cols[1].metric("Status", detail.get("status") or "—")
    cols[2].metric("Pages", detail.get("page_count") or "—")
    cols[3].metric("Chunks", detail.get("chunk_count") or "—")

    if detail.get("status_detail"):
        st.caption(detail["status_detail"])

    is_ready = bool(detail.get("ready_to_ask"))
    in_scope = active_doc_id in selected_ready_doc_ids()

    action_cols = st.columns(3)
    if action_cols[0].button("Refresh details", use_container_width=True):
        refresh_document_detail(active_doc_id, force=True)
        flash("info", "Document details were refreshed.")
        st.rerun()

    if is_ready and action_cols[1].button(
        "Remove from scope" if in_scope else "Add to scope",
        use_container_width=True,
    ):
        if in_scope:
            st.session_state.selected_doc_ids = [
                doc_id
                for doc_id in st.session_state.selected_doc_ids
                if doc_id != active_doc_id
            ]
            flash("info", "The document was removed from the active question scope.")
        else:
            st.session_state.selected_doc_ids = list(
                dict.fromkeys(st.session_state.selected_doc_ids + [active_doc_id])
            )
            flash("success", "The document was added to the active question scope.")
        st.rerun()

    if action_cols[2].button("Delete document", use_container_width=True):
        result = api_request("DELETE", f"/documents/{active_doc_id}")
        if result.ok:
            st.session_state.document_detail_cache.pop(active_doc_id, None)
            st.session_state.selected_doc_ids = [
                doc_id
                for doc_id in st.session_state.selected_doc_ids
                if doc_id != active_doc_id
            ]
            queue_refresh(hard=True)
            flash("success", "The document was deleted.")
            st.rerun()
        else:
            flash("error", format_api_error(result))
            st.rerun()

    with st.expander("Artifacts", expanded=False):
        st.json(detail.get("artifacts", {}))


with documents_tab:
    top_cols = st.columns([1, 1, 2])
    if top_cols[0].button("Refresh library", use_container_width=True):
        queue_refresh(hard=True)
        flash("info", "The document library was refreshed.")
        st.rerun()
    if top_cols[1].button("Select all ready", use_container_width=True):
        st.session_state.selected_doc_ids = ready_doc_ids()
        flash(
            "success",
            f"Added {len(st.session_state.selected_doc_ids)} ready documents to the active scope.",
        )
        st.rerun()
    top_cols[2].caption(
        "The library is a read-only overview with clear details and actions. Question scope selection is handled separately to keep the workflow stable."
    )

    render_document_table()
    st.divider()
    render_document_detail_panel()