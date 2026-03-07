from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import streamlit as st

APP_TITLE = "Abysalto · DocQA Streamlit Demo"
DEFAULT_API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SEC", "60"))
ALLOWED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff"]

STATUS_EMOJI = {
    "uploaded": "📄",
    "processing": "⏳",
    "indexed": "✅",
    "failed": "❌",
    "error": "⚠️",
    "deleted": "🗑️",
}


@dataclass(slots=True)
class ApiResult:
    ok: bool
    status_code: int
    data: Any | None
    text: str
    headers: dict[str, str]
    error_code: str | None = None
    error_message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.35rem 0 0.2rem 0;
        }
        .chip {
            display: inline-block;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 999px;
            padding: 0.20rem 0.68rem;
            font-size: 0.83rem;
            background: rgba(49, 51, 63, 0.05);
        }
        .answer-box {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 0.95rem;
            padding: 1rem 1rem 0.85rem 1rem;
            background: rgba(49, 51, 63, 0.03);
            margin-bottom: 1rem;
        }
        .muted-box {
            border: 1px dashed rgba(49, 51, 63, 0.18);
            border-radius: 0.9rem;
            padding: 0.85rem 1rem;
            background: rgba(49, 51, 63, 0.02);
        }
        .tiny {
            font-size: 0.84rem;
            color: #6b7280;
        }
        .section-note {
            color: #6b7280;
            margin-top: -0.15rem;
            margin-bottom: 0.85rem;
        }
        .soft-box {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 0.9rem;
            padding: 0.8rem 1rem;
            background: rgba(49, 51, 63, 0.02);
            margin-top: 0.5rem;
            margin-bottom: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "http_session" not in st.session_state:
        st.session_state.http_session = requests.Session()
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL
    if "request_timeout" not in st.session_state:
        st.session_state.request_timeout = DEFAULT_TIMEOUT_SECONDS
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = ""
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "current_identity" not in st.session_state:
        st.session_state.current_identity = None
    if "documents_cache" not in st.session_state:
        st.session_state.documents_cache = []
    if "document_detail_cache" not in st.session_state:
        st.session_state.document_detail_cache = {}
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = []
    if "last_upload_response" not in st.session_state:
        st.session_state.last_upload_response = None
    if "last_ask_response" not in st.session_state:
        st.session_state.last_ask_response = None
    if "last_api_result" not in st.session_state:
        st.session_state.last_api_result = None
    if "show_raw_payloads" not in st.session_state:
        st.session_state.show_raw_payloads = False
    if "auto_refresh_docs" not in st.session_state:
        st.session_state.auto_refresh_docs = True
    if "doc_sort_by" not in st.session_state:
        st.session_state.doc_sort_by = "created_at"
    if "doc_sort_desc" not in st.session_state:
        st.session_state.doc_sort_desc = True
    if "doc_inspector_selection" not in st.session_state:
        st.session_state.doc_inspector_selection = None
    if "doc_cards_per_row" not in st.session_state:
        st.session_state.doc_cards_per_row = 2
    if "doc_live_refresh" not in st.session_state:
        st.session_state.doc_live_refresh = False
    if "doc_live_refresh_sec" not in st.session_state:
        st.session_state.doc_live_refresh_sec = 5
    if "ask_question" not in st.session_state:
        st.session_state.ask_question = ""


init_state()


def get_session() -> requests.Session:
    return st.session_state.http_session


def reset_anonymous_session() -> None:
    st.session_state.http_session = requests.Session()
    st.session_state.current_identity = None
    st.session_state.documents_cache = []
    st.session_state.document_detail_cache = {}
    st.session_state.last_api_result = None
    st.session_state.last_upload_response = None
    st.session_state.last_ask_response = None
    st.session_state.selected_doc_ids = []
    st.session_state.doc_inspector_selection = None


def api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    include_auth: bool = True,
) -> ApiResult:
    base_url = st.session_state.api_base_url.rstrip("/")
    url = f"{base_url}{path}"
    headers: dict[str, str] = {"Accept": "application/json"}

    if include_auth and st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token.strip()}"

    try:
        response = get_session().request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_body,
            files=files,
            timeout=int(st.session_state.request_timeout),
        )
    except requests.RequestException as exc:
        result = ApiResult(
            ok=False,
            status_code=0,
            data=None,
            text=str(exc),
            headers={},
            error_code="connection_error",
            error_message=f"API request failed: {exc}",
        )
        st.session_state.last_api_result = result
        return result

    data: Any | None = None
    text = response.text
    try:
        data = response.json()
    except ValueError:
        data = None

    payload = data if isinstance(data, dict) else {}
    result = ApiResult(
        ok=response.ok,
        status_code=response.status_code,
        data=data,
        text=text,
        headers=dict(response.headers),
        error_code=payload.get("error_code"),
        error_message=payload.get("message"),
        request_id=payload.get("request_id"),
        retry_after=response.headers.get("Retry-After"),
    )
    st.session_state.last_api_result = result
    return result


def sync_identity_state() -> None:
    identity_result = api_request(
        "GET",
        "/auth/identity",
        include_auth=bool(st.session_state.auth_token),
    )
    if identity_result.ok:
        st.session_state.current_identity = identity_result.data
    else:
        st.session_state.current_identity = None

    if st.session_state.auth_token:
        me_result = api_request("GET", "/auth/me")
        if me_result.ok:
            st.session_state.current_user = me_result.data
        else:
            if me_result.status_code == 401:
                st.session_state.auth_token = ""
            st.session_state.current_user = None
    else:
        st.session_state.current_user = None


def refresh_documents(show_feedback: bool = False) -> ApiResult:
    result = api_request("GET", "/documents")
    if result.ok and isinstance(result.data, dict):
        documents = result.data.get("documents", [])
        st.session_state.documents_cache = documents
        allowed_doc_ids = {doc.get("doc_id") for doc in documents if doc.get("doc_id")}
        st.session_state.selected_doc_ids = [
            doc_id
            for doc_id in st.session_state.selected_doc_ids
            if doc_id in allowed_doc_ids
        ]
        if st.session_state.doc_inspector_selection not in allowed_doc_ids:
            st.session_state.doc_inspector_selection = None
        if show_feedback:
            st.toast("Documents refreshed.")
    elif show_feedback:
        render_api_feedback(result)
    return result


def get_document_detail(doc_id: str, *, force_refresh: bool = False) -> ApiResult:
    if not force_refresh and doc_id in st.session_state.document_detail_cache:
        cached = st.session_state.document_detail_cache[doc_id]
        return ApiResult(
            ok=True,
            status_code=200,
            data=cached,
            text=json.dumps(cached),
            headers={},
        )

    result = api_request("GET", f"/documents/{doc_id}")
    if result.ok and isinstance(result.data, dict):
        st.session_state.document_detail_cache[doc_id] = result.data
    return result


def delete_document(doc_id: str) -> ApiResult:
    result = api_request("DELETE", f"/documents/{doc_id}")
    if result.ok:
        st.session_state.document_detail_cache.pop(doc_id, None)
        st.session_state.selected_doc_ids = [
            item for item in st.session_state.selected_doc_ids if item != doc_id
        ]
        if st.session_state.doc_inspector_selection == doc_id:
            st.session_state.doc_inspector_selection = None
        refresh_documents(show_feedback=False)
    return result


def format_datetime(value: str | None) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def format_size(num_bytes: int | None) -> str:
    if num_bytes in (None, 0):
        return "—" if num_bytes is None else "0 B"
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def status_label(status: str | None) -> str:
    value = (status or "unknown").lower()
    return f"{STATUS_EMOJI.get(value, '•')} {value}"


def trim_text(value: Any, limit: int = 120) -> str:
    cleaned = " ".join(str(value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1]}…"


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def datetime_sort_value(value: Any) -> str:
    if not value:
        return ""
    return str(value)


def render_api_feedback(result: ApiResult) -> None:
    if result.ok:
        return

    message = result.error_message or result.text or "Request failed."
    suffix_parts = []
    if result.error_code:
        suffix_parts.append(f"error_code={result.error_code}")
    if result.request_id:
        suffix_parts.append(f"request_id={result.request_id}")
    if result.retry_after:
        suffix_parts.append(f"retry_after={result.retry_after}s")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""

    if result.status_code == 401:
        st.error(f"401 Unauthorized: {message}{suffix}")
    elif result.status_code == 429:
        st.warning(f"429 Too Many Requests: {message}{suffix}")
    elif result.status_code == 0:
        st.error(message)
    else:
        st.error(f"{result.status_code} Error: {message}{suffix}")


def render_entities(entities: list[dict[str, Any]]) -> None:
    if not entities:
        st.caption("No entities extracted.")
        return

    chips: list[str] = []
    for entity in entities:
        text = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "ENTITY")).strip()
        score = entity.get("score")
        suffix = f" · {float(score):.2f}" if isinstance(score, (int, float)) else ""
        chips.append(
            f"<span class='chip'><strong>{label}</strong>: {text}{suffix}</span>"
        )

    st.markdown(
        f"<div class='chip-wrap'>{''.join(chips)}</div>", unsafe_allow_html=True
    )


def render_simple_chips(
    values: list[str], *, empty_text: str = "None selected."
) -> None:
    if not values:
        st.caption(empty_text)
        return
    chips = "".join(f"<span class='chip'>{value}</span>" for value in values)
    st.markdown(f"<div class='chip-wrap'>{chips}</div>", unsafe_allow_html=True)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):
        filename = source.get("filename") or "Unknown file"
        page = source.get("page")
        score = source.get("score")
        chunk_id = source.get("chunk_id", "—")
        doc_id = source.get("doc_id", "—")

        title_parts = [f"{index}. {filename}"]
        if page is not None:
            title_parts.append(f"page {page}")
        if isinstance(score, (int, float)):
            title_parts.append(f"score {score:.3f}")

        with st.expander(" · ".join(title_parts), expanded=index == 1):
            col_a, col_b = st.columns(2)
            col_a.caption(f"doc_id: {doc_id}")
            col_b.caption(f"chunk_id: {chunk_id}")
            st.write(source.get("text_excerpt") or "—")


def show_sidebar() -> None:
    st.sidebar.title("Control panel")
    st.sidebar.text_input(
        "API base URL",
        key="api_base_url",
        help="Point the Streamlit UI to the running FastAPI backend.",
    )
    st.sidebar.number_input(
        "Request timeout (sec)",
        key="request_timeout",
        min_value=5,
        max_value=300,
        step=5,
    )
    st.sidebar.checkbox(
        "Show raw API payloads",
        key="show_raw_payloads",
        help="Useful for debugging and demos.",
    )
    st.sidebar.checkbox(
        "Auto-refresh documents after mutations",
        key="auto_refresh_docs",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick diagnostics")

    if st.sidebar.button("Health check", use_container_width=True):
        result = api_request("GET", "/health", include_auth=False)
        if result.ok:
            st.sidebar.success("Backend healthy.")
            if st.session_state.show_raw_payloads and result.data is not None:
                st.sidebar.json(result.data)
        else:
            st.sidebar.error(result.error_message or result.text)

    if st.sidebar.button("Refresh identity", use_container_width=True):
        sync_identity_state()
        if st.session_state.current_identity:
            st.sidebar.success("Identity refreshed.")
        else:
            st.sidebar.warning("Identity not available.")

    identity = st.session_state.current_identity or {}
    user = st.session_state.current_user or {}

    st.sidebar.markdown(
        f"""
        <div class='muted-box'>
            <div><strong>Auth mode</strong>: {'Bearer token' if st.session_state.auth_token else 'Anonymous session'}</div>
            <div class='tiny'>identity.kind: {identity.get('kind', '—')}</div>
            <div class='tiny'>session_id: {identity.get('session_id', '—')}</div>
            <div class='tiny'>user: {user.get('email', '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        st.session_state.show_raw_payloads
        and st.session_state.last_api_result is not None
    ):
        with st.sidebar.expander("Last API result", expanded=False):
            result: ApiResult = st.session_state.last_api_result
            st.write(
                {
                    "ok": result.ok,
                    "status_code": result.status_code,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                    "request_id": result.request_id,
                    "retry_after": result.retry_after,
                }
            )
            if result.data is not None:
                st.json(result.data)
            else:
                st.code(result.text or "", language="text")


def render_auth_section() -> None:
    st.header("1. Register / Login")
    st.markdown(
        "Anonymous usage is backed by a persistent `requests.Session()` stored in `st.session_state`, so the backend cookie stays stable across upload, listing and ask flows."
    )

    identity = st.session_state.current_identity or {}
    user = st.session_state.current_user or {}

    summary_cols = st.columns(4)
    summary_cols[0].metric(
        "Auth mode",
        "Bearer" if st.session_state.auth_token else "Anonymous",
    )
    summary_cols[1].metric("Identity kind", identity.get("kind", "—"))
    summary_cols[2].metric("Session", trim_text(identity.get("session_id", "—"), 18))
    summary_cols[3].metric("User", user.get("email", "—"))

    auth_tabs = st.tabs(["Register", "Login", "Use existing token", "Session controls"])

    with auth_tabs[0]:
        with st.form("register_form", clear_on_submit=False):
            register_email = st.text_input("Email", key="register_email")
            register_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )
            register_password_confirm = st.text_input(
                "Confirm password",
                type="password",
                key="register_password_confirm",
            )
            auto_login = st.checkbox("Auto-login after register", value=True)
            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )

        if submitted:
            if register_password != register_password_confirm:
                st.error("Passwords do not match.")
            else:
                result = api_request(
                    "POST",
                    "/auth/register",
                    json_body={
                        "email": register_email,
                        "password": register_password,
                    },
                    include_auth=False,
                )
                if result.ok:
                    st.success("Account created successfully.")
                    if auto_login:
                        login_result = api_request(
                            "POST",
                            "/auth/login",
                            json_body={
                                "email": register_email,
                                "password": register_password,
                            },
                            include_auth=False,
                        )
                        if login_result.ok and isinstance(login_result.data, dict):
                            st.session_state.auth_token = login_result.data.get(
                                "access_token",
                                "",
                            )
                            sync_identity_state()
                            if st.session_state.auto_refresh_docs:
                                refresh_documents(show_feedback=False)
                            st.success("Account created and logged in.")
                        else:
                            render_api_feedback(login_result)
                    else:
                        sync_identity_state()
                else:
                    render_api_feedback(result)

    with auth_tabs[1]:
        with st.form("login_form", clear_on_submit=False):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            result = api_request(
                "POST",
                "/auth/login",
                json_body={"email": login_email, "password": login_password},
                include_auth=False,
            )
            if result.ok and isinstance(result.data, dict):
                st.session_state.auth_token = result.data.get("access_token", "")
                sync_identity_state()
                if st.session_state.auto_refresh_docs:
                    refresh_documents(show_feedback=False)
                expires_in = result.data.get("expires_in")
                st.success(
                    f"Logged in successfully. Token expires in {expires_in} seconds."
                )
            else:
                render_api_feedback(result)

    with auth_tabs[2]:
        with st.form("token_form", clear_on_submit=False):
            token_value = st.text_area(
                "Paste bearer token",
                value=st.session_state.auth_token,
                height=120,
                placeholder="eyJhbGciOi...",
            )
            submitted = st.form_submit_button("Save token", use_container_width=True)

        if submitted:
            st.session_state.auth_token = token_value.strip()
            sync_identity_state()
            if st.session_state.auth_token:
                st.success("Bearer token saved.")
            else:
                st.info("Token cleared.")

    with auth_tabs[3]:
        left, right = st.columns(2)

        if left.button("Refresh identity + user", use_container_width=True):
            sync_identity_state()
            st.success("Identity state refreshed.")

        if right.button("Refresh documents", use_container_width=True):
            refresh_documents(show_feedback=True)

        left, right = st.columns(2)

        if left.button("Logout bearer token", use_container_width=True):
            st.session_state.auth_token = ""
            sync_identity_state()
            st.info(
                "Bearer token removed. Session cookie remains available for anonymous mode."
            )

        if right.button("Reset anonymous session", use_container_width=True):
            reset_anonymous_session()
            sync_identity_state()
            st.warning(
                "Anonymous session reset. A new backend cookie will be created on the next request."
            )

        if st.session_state.show_raw_payloads:
            st.markdown("#### Current identity payload")
            st.json(st.session_state.current_identity or {})
            st.markdown("#### Current user payload")
            st.json(st.session_state.current_user or {})


def render_upload_section() -> None:
    st.header("2. Upload")
    st.markdown(
        "Upload one or more documents, then inspect status, ownership, hashes and processing results."
    )

    with st.form("upload_form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Allowed: PDF, PNG, JPG, JPEG, TIF, TIFF",
        )
        submitted = st.form_submit_button("Upload files", use_container_width=True)

    if submitted:
        if not uploaded_files:
            st.warning("Select at least one file.")
        else:
            files_payload = []
            for file in uploaded_files:
                content_type = file.type or "application/octet-stream"
                files_payload.append(
                    ("files", (file.name, file.getvalue(), content_type))
                )

            result = api_request("POST", "/upload", files=files_payload)
            if result.ok:
                st.session_state.last_upload_response = result.data
                documents = (
                    (result.data or {}).get("documents", [])
                    if isinstance(result.data, dict)
                    else []
                )
                success_doc_ids = [
                    doc.get("doc_id") for doc in documents if doc.get("doc_id")
                ]
                for doc_id in success_doc_ids:
                    if doc_id not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.append(doc_id)
                if st.session_state.auto_refresh_docs:
                    refresh_documents(show_feedback=False)
                st.success("Upload request completed.")
            else:
                render_api_feedback(result)

    upload_response = st.session_state.last_upload_response
    if upload_response:
        documents = upload_response.get("documents", [])
        counts = {
            "success": sum(
                1 for item in documents if item.get("status") not in {"error", "failed"}
            ),
            "processing": sum(
                1 for item in documents if item.get("status") == "processing"
            ),
            "errors": sum(
                1 for item in documents if item.get("status") in {"error", "failed"}
            ),
        }
        metric_cols = st.columns(3)
        metric_cols[0].metric("Successful items", counts["success"])
        metric_cols[1].metric("Processing items", counts["processing"])
        metric_cols[2].metric("Errored items", counts["errors"])

        table_rows = []
        for item in documents:
            table_rows.append(
                {
                    "filename": item.get("filename"),
                    "status": status_label(item.get("status")),
                    "doc_id": item.get("doc_id") or "—",
                    "content_type": item.get("content_type") or "—",
                    "size": format_size(item.get("size_bytes")),
                    "owner_type": item.get("owner_type") or "—",
                    "sha256": trim_text(item.get("sha256") or "—", 18),
                    "error_detail": item.get("error_detail") or "—",
                }
            )
        st.dataframe(table_rows, use_container_width=True)

        for item in documents:
            title = (
                f"{status_label(item.get('status'))} · {item.get('filename', 'file')}"
            )
            with st.expander(
                title,
                expanded=item.get("status") in {"error", "failed"},
            ):
                st.json(item)

        if st.session_state.show_raw_payloads:
            with st.expander("Raw upload response"):
                st.json(upload_response)


def get_visible_documents(
    documents: list[dict[str, Any]],
    selected_statuses: list[str],
    sort_by: str,
    sort_desc: bool,
) -> list[dict[str, Any]]:
    visible_documents = []
    for doc in documents:
        status_match = (
            doc.get("status") in selected_statuses if selected_statuses else True
        )
        if status_match:
            visible_documents.append(doc)

    def sort_key(doc: dict[str, Any]) -> Any:
        if sort_by == "filename":
            return str(doc.get("filename") or "").lower()
        if sort_by == "status":
            return str(doc.get("status") or "").lower()
        if sort_by == "size_bytes":
            return safe_int(doc.get("size_bytes"))
        if sort_by == "pages":
            pages = doc.get("pages")
            if pages is None:
                pages = doc.get("page_count")
            return safe_int(pages)
        if sort_by == "chunks":
            chunks = doc.get("chunks")
            if chunks is None:
                chunks = doc.get("chunk_count")
            return safe_int(chunks)
        if sort_by == "indexed_at":
            return datetime_sort_value(doc.get("indexed_at"))
        return datetime_sort_value(doc.get("created_at"))

    return sorted(visible_documents, key=sort_key, reverse=sort_desc)


def get_processing_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processing_statuses = {"processing", "uploaded"}
    return [
        doc
        for doc in documents
        if str(doc.get("status") or "").lower() in processing_statuses
    ]


def toggle_doc_for_ask(doc_id: str) -> None:
    if doc_id in st.session_state.selected_doc_ids:
        st.session_state.selected_doc_ids = [
            item for item in st.session_state.selected_doc_ids if item != doc_id
        ]
    else:
        st.session_state.selected_doc_ids.append(doc_id)


def render_document_cards(visible_documents: list[dict[str, Any]]) -> None:
    if not visible_documents:
        st.info("No documents match the current filters.")
        return

    cards_per_row = int(st.session_state.doc_cards_per_row)
    for start in range(0, len(visible_documents), cards_per_row):
        row_docs = visible_documents[start : start + cards_per_row]
        cols = st.columns(cards_per_row)

        for index, doc in enumerate(row_docs):
            with cols[index]:
                with st.container(border=True):
                    doc_id = doc.get("doc_id") or "—"
                    filename = doc.get("filename") or "Untitled document"
                    status = doc.get("status")
                    size = format_size(doc.get("size_bytes"))

                    pages = doc.get("pages")
                    if pages is None:
                        pages = doc.get("page_count")

                    chunks = doc.get("chunks")
                    if chunks is None:
                        chunks = doc.get("chunk_count")

                    st.markdown(f"**{filename}**")
                    st.caption(doc_id)
                    st.write(status_label(status))
                    st.caption(
                        f"Size: {size} · Pages: {pages if pages is not None else '—'} · Chunks: {chunks if chunks is not None else '—'}"
                    )
                    st.caption(
                        f"Created: {format_datetime(doc.get('created_at'))} · Indexed: {format_datetime(doc.get('indexed_at'))}"
                    )

                    card_flags = []
                    if doc_id in st.session_state.selected_doc_ids:
                        card_flags.append("Selected for Ask")
                    if str(status or "").lower() == "processing":
                        card_flags.append("Live processing")
                    render_simple_chips(card_flags, empty_text="")

                    btn_cols = st.columns(2)
                    if btn_cols[0].button(
                        "Inspect",
                        use_container_width=True,
                        key=f"inspect_{doc_id}",
                    ):
                        st.session_state.doc_inspector_selection = doc_id

                    toggle_label = (
                        "Remove from Ask"
                        if doc_id in st.session_state.selected_doc_ids
                        else "Add to Ask"
                    )
                    if btn_cols[1].button(
                        toggle_label,
                        use_container_width=True,
                        key=f"toggle_ask_{doc_id}",
                    ):
                        toggle_doc_for_ask(doc_id)


def render_documents_section() -> None:
    st.header("3. Documents list")
    st.markdown(
        "<div class='section-note'>Browse all documents owned by the current identity, filter them, inspect details and manage deletion.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.documents_cache:
        refresh_documents(show_feedback=False)

    documents = list(st.session_state.documents_cache)

    live_cols = st.columns([1.4, 1.2, 3.4])
    live_cols[0].checkbox(
        "Live auto-refresh",
        key="doc_live_refresh",
        help="While documents are processing, refresh the list automatically every few seconds.",
    )
    live_cols[1].slider(
        "Refresh every (sec)",
        min_value=3,
        max_value=20,
        step=1,
        key="doc_live_refresh_sec",
    )

    if st.session_state.doc_live_refresh:
        processing_documents = get_processing_documents(documents)
        if processing_documents:
            refresh_documents(show_feedback=False)
            documents = list(st.session_state.documents_cache)
            processing_documents = get_processing_documents(documents)
            if processing_documents:
                processing_names = [
                    str(doc.get("filename") or doc.get("doc_id"))
                    for doc in processing_documents[:4]
                ]
                more_suffix = ""
                if len(processing_documents) > 4:
                    more_suffix = f" +{len(processing_documents) - 4} more"
                st.info(
                    f"Live refresh active every {st.session_state.doc_live_refresh_sec}s. Still processing: {', '.join(processing_names)}{more_suffix}"
                )
            else:
                st.success("All documents finished processing.")
        else:
            st.success("No documents are currently processing.")

    status_options = sorted({doc.get("status", "unknown") for doc in documents})

    control_cols = st.columns([1.1, 1.2, 1.1, 1.6, 2.0])
    if control_cols[0].button(
        "Refresh list", use_container_width=True, key="docs_refresh"
    ):
        refresh_documents(show_feedback=True)
        documents = list(st.session_state.documents_cache)

    if control_cols[1].button(
        "Select all indexed",
        use_container_width=True,
        key="docs_select_indexed",
    ):
        indexed_ids = [
            doc.get("doc_id")
            for doc in documents
            if doc.get("status") == "indexed" and doc.get("doc_id")
        ]
        st.session_state.selected_doc_ids = list(
            dict.fromkeys(st.session_state.selected_doc_ids + indexed_ids)
        )
        st.success(f"Added {len(indexed_ids)} indexed docs to Ask selection.")

    if control_cols[2].button(
        "Clear selection",
        use_container_width=True,
        key="docs_clear_selection",
    ):
        st.session_state.selected_doc_ids = []
        st.info("Selected docs cleared.")

    filter_cols = st.columns([2.0, 1.4, 1.2, 1.2])
    selected_statuses = filter_cols[0].multiselect(
        "Filter by status",
        options=status_options,
        default=status_options,
    )
    sort_by = filter_cols[1].selectbox(
        "Sort by",
        options=[
            "created_at",
            "indexed_at",
            "filename",
            "status",
            "size_bytes",
            "pages",
            "chunks",
        ],
        format_func=lambda value: {
            "created_at": "Created at",
            "indexed_at": "Indexed at",
            "filename": "Filename",
            "status": "Status",
            "size_bytes": "Size",
            "pages": "Pages",
            "chunks": "Chunks",
        }[value],
        key="doc_sort_by",
    )
    filter_cols[2].toggle("Descending", key="doc_sort_desc")
    filter_cols[3].selectbox(
        "Cards / row",
        options=[1, 2, 3],
        key="doc_cards_per_row",
    )

    visible_documents = get_visible_documents(
        documents=documents,
        selected_statuses=selected_statuses,
        sort_by=sort_by,
        sort_desc=st.session_state.doc_sort_desc,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total docs", len(documents))
    metric_cols[1].metric("Visible docs", len(visible_documents))
    metric_cols[2].metric(
        "Indexed docs",
        sum(1 for doc in documents if doc.get("status") == "indexed"),
    )
    metric_cols[3].metric(
        "Processing docs",
        sum(1 for doc in documents if doc.get("status") == "processing"),
    )
    metric_cols[4].metric("Selected docs", len(st.session_state.selected_doc_ids))

    valid_doc_ids = {doc.get("doc_id") for doc in documents if doc.get("doc_id")}
    st.session_state.selected_doc_ids = [
        doc_id
        for doc_id in st.session_state.selected_doc_ids
        if doc_id in valid_doc_ids
    ]

    selectable_options = {
        f"{doc.get('filename')} ({doc.get('doc_id')})": doc.get("doc_id")
        for doc in visible_documents
        if doc.get("doc_id")
    }

    selection_cols = st.columns([4.2, 1.2])
    if selection_cols[1].button(
        "Select visible",
        use_container_width=True,
        key="docs_select_visible",
    ):
        st.session_state.selected_doc_ids = [
            doc_id for doc_id in selectable_options.values() if doc_id
        ]

    default_selected_labels = [
        label
        for label, doc_id in selectable_options.items()
        if doc_id in st.session_state.selected_doc_ids
    ]

    selected_labels = selection_cols[0].multiselect(
        "Docs selected for Ask",
        options=list(selectable_options.keys()),
        default=default_selected_labels,
        help="Use this to scope Ask requests to a curated subset of documents.",
        key=f"docs_selected_multiselect_{len(selectable_options)}_{len(default_selected_labels)}",
    )

    st.session_state.selected_doc_ids = [
        selectable_options[label] for label in selected_labels
    ]

    st.markdown(
        "<div class='soft-box'><strong>Ask selection preview</strong></div>",
        unsafe_allow_html=True,
    )
    doc_id_to_name = {
        doc.get("doc_id"): str(doc.get("filename") or doc.get("doc_id"))
        for doc in documents
        if doc.get("doc_id")
    }
    selected_names = [
        doc_id_to_name.get(doc_id, doc_id)
        for doc_id in st.session_state.selected_doc_ids
    ]
    render_simple_chips(selected_names, empty_text="No docs selected for Ask.")

    st.markdown("#### Card view")
    render_document_cards(visible_documents)

    st.markdown("#### Compact table")
    table_rows = []
    for doc in visible_documents:
        pages = doc.get("pages")
        if pages is None:
            pages = doc.get("page_count")
        chunks = doc.get("chunks")
        if chunks is None:
            chunks = doc.get("chunk_count")
        table_rows.append(
            {
                "doc_id": doc.get("doc_id"),
                "filename": doc.get("filename"),
                "status": status_label(doc.get("status")),
                "size": format_size(doc.get("size_bytes")),
                "pages": pages if pages is not None else "—",
                "chunks": chunks if chunks is not None else "—",
                "created_at": format_datetime(doc.get("created_at")),
                "indexed_at": format_datetime(doc.get("indexed_at")),
            }
        )
    if table_rows:
        with st.expander("Show compact table", expanded=False):
            st.dataframe(table_rows, use_container_width=True)

    st.markdown("#### Document inspector")

    inspector_options = {
        f"{doc.get('filename')} ({doc.get('doc_id')})": doc.get("doc_id")
        for doc in visible_documents
        if doc.get("doc_id")
    }

    if not inspector_options:
        st.info("No visible documents available for inspection.")
        if st.session_state.doc_live_refresh and get_processing_documents(documents):
            time.sleep(int(st.session_state.doc_live_refresh_sec))
            st.rerun()
        return

    available_doc_ids = list(inspector_options.values())
    if st.session_state.doc_inspector_selection not in available_doc_ids:
        st.session_state.doc_inspector_selection = available_doc_ids[0]

    current_label = next(
        (
            label
            for label, doc_id in inspector_options.items()
            if doc_id == st.session_state.doc_inspector_selection
        ),
        list(inspector_options.keys())[0],
    )

    selected_label = st.selectbox(
        "Choose document",
        options=list(inspector_options.keys()),
        index=list(inspector_options.keys()).index(current_label),
        key="doc_inspector_label",
    )
    selected_doc_id = inspector_options[selected_label]
    st.session_state.doc_inspector_selection = selected_doc_id

    inspector_button_cols = st.columns([1.2, 1.2, 3.6])
    if inspector_button_cols[0].button(
        "Refresh detail",
        use_container_width=True,
        key="docs_force_refresh_detail",
    ):
        detail_result = get_document_detail(selected_doc_id, force_refresh=True)
    else:
        detail_result = get_document_detail(selected_doc_id, force_refresh=False)

    if inspector_button_cols[1].button(
        "Toggle Ask selection",
        use_container_width=True,
        key="docs_toggle_inspector_ask",
    ):
        toggle_doc_for_ask(selected_doc_id)

    if not (detail_result.ok and isinstance(detail_result.data, dict)):
        render_api_feedback(detail_result)
        if st.session_state.doc_live_refresh and get_processing_documents(documents):
            time.sleep(int(st.session_state.doc_live_refresh_sec))
            st.rerun()
        return

    detail = detail_result.data

    overview_cols = st.columns(4)
    overview_cols[0].metric("Status", detail.get("status", "—"))
    overview_cols[1].metric("Owner type", detail.get("owner_type", "—"))
    overview_cols[2].metric(
        "Pages",
        detail.get("page_count") if detail.get("page_count") is not None else "—",
    )
    overview_cols[3].metric(
        "Chunks",
        detail.get("chunk_count") if detail.get("chunk_count") is not None else "—",
    )

    meta_left, meta_right = st.columns(2)
    meta_left.write(
        {
            "doc_id": detail.get("doc_id"),
            "filename": detail.get("filename"),
            "content_type": detail.get("content_type"),
            "size": format_size(detail.get("size_bytes")),
        }
    )
    meta_right.write(
        {
            "created_at": format_datetime(detail.get("created_at")),
            "indexed_at": format_datetime(detail.get("indexed_at")),
            "sha256": detail.get("sha256", "—"),
            "owner_id": detail.get("owner_id", "—"),
        }
    )

    st.markdown("**Artifacts**")
    artifacts = detail.get("artifacts", {})
    if artifacts:
        artifact_chips = [
            f"{name}: {'yes' if value else 'no'}" for name, value in artifacts.items()
        ]
        render_simple_chips(artifact_chips, empty_text="No artifact flags.")
    else:
        st.caption("No artifact flags available.")

    with st.expander("Raw document detail payload", expanded=False):
        st.json(detail)

    st.markdown("**Danger zone**")
    danger_cols = st.columns([2.6, 1.2, 3.2])
    confirm_delete = danger_cols[0].checkbox(
        "I understand this will permanently delete the document",
        key=f"delete_confirm_{selected_doc_id}",
    )
    delete_clicked = danger_cols[1].button(
        "Delete document",
        type="secondary",
        use_container_width=True,
        key=f"delete_btn_{selected_doc_id}",
    )

    if delete_clicked:
        if not confirm_delete:
            st.warning("Tick the confirmation checkbox before deleting the document.")
        else:
            delete_result = delete_document(selected_doc_id)
            if delete_result.ok:
                st.success(f"Deleted {selected_doc_id}.")
            else:
                render_api_feedback(delete_result)

    if st.session_state.doc_live_refresh and get_processing_documents(documents):
        time.sleep(int(st.session_state.doc_live_refresh_sec))
        st.rerun()


def render_ask_section() -> None:
    st.header("4. Ask")
    st.markdown(
        "Ask across all indexed docs for the current identity, or narrow the scope to selected documents only."
    )

    example_prompts = [
        "Summarize the main points from my indexed documents.",
        "Which entities, people or organizations appear most often?",
        "What are the important dates, deadlines or commitments mentioned?",
    ]
    quick_cols = st.columns(len(example_prompts))
    for index, prompt in enumerate(example_prompts):
        if quick_cols[index].button(
            prompt,
            use_container_width=True,
            key=f"prompt_{index}",
        ):
            st.session_state.ask_question = prompt

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            key="ask_question",
            height=140,
            placeholder="Ask something grounded in your indexed documents...",
        )
        form_cols = st.columns([2, 2, 1])
        scope_label = form_cols[0].radio(
            "Scope",
            options=["all indexed docs", "selected docs only"],
            horizontal=True,
        )
        top_k = form_cols[1].slider("top_k", min_value=1, max_value=20, value=5)
        selected_docs_snapshot = list(st.session_state.selected_doc_ids)
        form_cols[2].metric("Selected docs", len(selected_docs_snapshot))
        submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted:
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if scope_label == "selected docs only":
            payload["scope"] = "docs"
            payload["doc_ids"] = selected_docs_snapshot
        else:
            payload["scope"] = "all"

        result = api_request("POST", "/ask", json_body=payload)
        if result.ok and isinstance(result.data, dict):
            st.session_state.last_ask_response = result.data
            st.success("Answer generated.")
        else:
            render_api_feedback(result)

    ask_response = st.session_state.last_ask_response
    if ask_response:
        answer = ask_response.get("answer") or "—"
        confidence = ask_response.get("confidence")
        sources = ask_response.get("sources", [])
        entities = ask_response.get("entities", [])

        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Confidence",
            f"{float(confidence):.3f}" if isinstance(confidence, (int, float)) else "—",
        )
        metric_cols[1].metric("Sources", len(sources))
        metric_cols[2].metric("Entities", len(entities))

        st.markdown("#### Entities")
        render_entities(entities)

        st.markdown("#### Sources")
        render_sources(sources)

        st.download_button(
            "Download answer as JSON",
            data=json.dumps(ask_response, indent=2, ensure_ascii=False),
            file_name="ask_response.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.session_state.show_raw_payloads:
            with st.expander("Raw ask response"):
                st.json(ask_response)


show_sidebar()
st.title(APP_TITLE)
st.caption(
    "Professional Streamlit demo for auth, upload, document management and grounded Q&A against the existing FastAPI backend."
)

if st.session_state.current_identity is None:
    sync_identity_state()

render_auth_section()
st.divider()
render_upload_section()
st.divider()
render_documents_section()
st.divider()
render_ask_section()
