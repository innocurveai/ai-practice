import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ──────────────────────────────────────────────
# 경로 및 환경 변수
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOGO_PATH = PROJECT_ROOT / "logo.png"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_BATCH_SIZE = 10
CHATBOT_TITLE = "ENA RAG 챗봇"
PBKDF2_ITERATIONS = 100_000

ANSWER_FORMAT_INSTRUCTION = """
답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요.
주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요.
답변은 서술형으로 작성하되 존대말을 사용하세요.
완전한 문장으로 서술하세요.
구분선(---, ===, ___) 사용 금지.
취소선(~~텍스트~~) 사용 금지.
참조 표시나 출처 문구 사용 금지.
"""

RAG_SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘. "
    + ANSWER_FORMAT_INSTRUCTION
)

DIRECT_LLM_SYSTEM_PROMPT = (
    "당신은 친절하고 유능한 AI 어시스턴트입니다. "
    + ANSWER_FORMAT_INSTRUCTION
)

FOLLOW_UP_SYSTEM_PROMPT = (
    "사용자와 AI의 대화를 바탕으로, 사용자가 이어서 물어볼 만한 질문 3개를 생성하세요. "
    "각 질문은 한 줄로 작성하고, 번호 없이 질문만 줄바꿈으로 구분하세요. "
    "질문만 출력하고 다른 설명은 하지 마세요."
)

TITLE_SYSTEM_PROMPT = (
    "아래 첫 질문과 첫 답변을 바탕으로 세션 제목을 한글로 만들어 주세요. "
    "15자 이내의 짧은 제목만 출력하고, 따옴표나 설명은 넣지 마세요."
)


# ──────────────────────────────────────────────
# 시크릿 로딩 (Streamlit Cloud 우선, 없으면 .env)
# ──────────────────────────────────────────────
def get_secret(key: str, default: str = "") -> str:
    """st.secrets → 환경변수(.env) 순으로 키를 조회한다."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return (os.getenv(key) or default).strip()


OPENAI_API_KEY = ""
SUPABASE_URL = ""
SUPABASE_ANON_KEY = ""


def refresh_secrets() -> None:
    """앱 시작/재실행 시 st.secrets → .env 순으로 키를 다시 읽는다."""
    global OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
    OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
    SUPABASE_URL = get_secret("SUPABASE_URL")
    SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")


refresh_secrets()


# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("multi_user_rag")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    for noisy_logger in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
        "supabase",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger


LOGGER = setup_logging()


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def remove_separators(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"^[\-\=_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_memory_context(memory: list[dict[str, str]], limit: int = 50) -> str:
    recent = memory[-limit:]
    lines: list[str] = []
    for item in recent:
        role = "사용자" if item["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def append_follow_up_section(answer: str, follow_up_questions: list[str]) -> str:
    section_lines = ["### 💡 다음에 물어볼 수 있는 질문들"]
    for idx, question in enumerate(follow_up_questions[:3], start=1):
        section_lines.append(f"{idx}. {question.strip()}")
    return f"{answer.rstrip()}\n\n" + "\n".join(section_lines)


def parse_follow_up_questions(raw_text: str) -> list[str]:
    questions: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned:
            questions.append(cleaned)
    return questions[:3]


def missing_env_messages() -> list[str]:
    messages: list[str] = []
    if not OPENAI_API_KEY:
        messages.append(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "Streamlit secrets 또는 .env 파일에 키를 추가해 주세요."
        )
    if not SUPABASE_URL:
        messages.append(
            "SUPABASE_URL이 설정되지 않았습니다. "
            "Streamlit secrets 또는 .env 파일에 키를 추가해 주세요."
        )
    if not SUPABASE_ANON_KEY:
        messages.append(
            "SUPABASE_ANON_KEY가 설정되지 않았습니다. "
            "Streamlit secrets 또는 .env 파일에 키를 추가해 주세요."
        )
    return messages


def get_supabase_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as exc:
        LOGGER.error("Supabase 클라이언트 생성 실패: %s", exc)
        return None


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def current_user_id() -> str | None:
    return st.session_state.get("user_id")


def require_user_id() -> str:
    user_id = current_user_id()
    if not user_id:
        raise PermissionError("로그인이 필요합니다.")
    return user_id


# ──────────────────────────────────────────────
# 비밀번호 해시 (평문 저장 금지)
# ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    """PBKDF2-SHA256 해시. 형식: salt$hex_digest"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return secrets.compare_digest(check.hex(), digest_hex)


# ──────────────────────────────────────────────
# 사용자 회원가입 / 로그인 (user 테이블 기반)
# ──────────────────────────────────────────────
def register_user(
    supabase: Client,
    login_id: str,
    password: str,
    display_name: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 모두 입력해 주세요."
    if len(login_id) < 3:
        return None, "아이디는 3자 이상이어야 합니다."
    if len(password) < 4:
        return None, "비밀번호는 4자 이상이어야 합니다."

    try:
        existing = (
            supabase.table("user")
            .select("id")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            return None, "이미 사용 중인 아이디입니다."

        payload = {
            "login_id": login_id,
            "password_hash": hash_password(password),
            "display_name": display_name.strip() or login_id,
        }
        response = supabase.table("user").insert(payload).execute()
        if not response.data:
            return None, "회원가입에 실패했습니다. 잠시 후 다시 시도해 주세요."
        return response.data[0], None
    except Exception as exc:
        LOGGER.error("회원가입 실패: %s", exc)
        return None, f"회원가입 중 오류가 발생했습니다: {exc}"


def authenticate_user(
    supabase: Client,
    login_id: str,
    password: str,
) -> tuple[dict[str, Any] | None, str | None]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 입력해 주세요."

    try:
        response = (
            supabase.table("user")
            .select("id, login_id, password_hash, display_name")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None, "아이디 또는 비밀번호가 올바르지 않습니다."

        user = response.data[0]
        if not verify_password(password, user.get("password_hash") or ""):
            return None, "아이디 또는 비밀번호가 올바르지 않습니다."

        return {
            "id": user["id"],
            "login_id": user["login_id"],
            "display_name": user.get("display_name") or user["login_id"],
        }, None
    except Exception as exc:
        LOGGER.error("로그인 실패: %s", exc)
        return None, f"로그인 중 오류가 발생했습니다: {exc}"


def set_logged_in_user(user: dict[str, Any]) -> None:
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.login_id = user["login_id"]
    st.session_state.display_name = user.get("display_name") or user["login_id"]
    reset_local_session()


def logout_user() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ──────────────────────────────────────────────
# Supabase 세션 / 벡터 CRUD (항상 user_id 필터)
# ──────────────────────────────────────────────
def list_sessions(supabase: Client, user_id: str) -> list[dict[str, Any]]:
    response = (
        supabase.table("chat_sessions")
        .select("id, title, processed_files, created_at, updated_at, user_id")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return response.data or []


def session_belongs_to_user(
    supabase: Client,
    session_id: str,
    user_id: str,
) -> bool:
    response = (
        supabase.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(response.data)


def ensure_session_row(
    supabase: Client,
    session_id: str,
    user_id: str,
    title: str = "새 세션",
    processed_files: list[str] | None = None,
) -> None:
    payload = {
        "id": session_id,
        "user_id": user_id,
        "title": title,
        "processed_files": processed_files or [],
        "updated_at": datetime.utcnow().isoformat(),
    }
    existing = (
        supabase.table("chat_sessions")
        .select("id, title, user_id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        update_payload = {
            "processed_files": processed_files or [],
            "updated_at": datetime.utcnow().isoformat(),
        }
        if existing.data[0].get("title") and existing.data[0]["title"] != "새 세션":
            pass
        elif title and title != "새 세션":
            update_payload["title"] = title
        (
            supabase.table("chat_sessions")
            .update(update_payload)
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
    else:
        supabase.table("chat_sessions").insert(payload).execute()


def save_messages(
    supabase: Client,
    session_id: str,
    user_id: str,
    chat_history: list[dict[str, str]],
) -> None:
    (
        supabase.table("chat_messages")
        .delete()
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not chat_history:
        return

    rows = []
    for idx, message in enumerate(chat_history):
        rows.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": message["role"],
                "content": message["content"],
                "message_order": idx,
            }
        )
    supabase.table("chat_messages").insert(rows).execute()


def load_messages(
    supabase: Client,
    session_id: str,
    user_id: str,
) -> list[dict[str, str]]:
    response = (
        supabase.table("chat_messages")
        .select("role, content, message_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("message_order")
        .execute()
    )
    return [
        {"role": row["role"], "content": row["content"]}
        for row in (response.data or [])
    ]


def auto_save_session(supabase: Client | None = None) -> str | None:
    """현재 화면 상태를 Supabase에 자동 저장한다."""
    client = supabase or get_supabase_client()
    if client is None:
        return "Supabase 연결 정보가 없어 자동 저장을 건너뜁니다."

    user_id = current_user_id()
    if not user_id:
        return "로그인이 필요해 자동 저장을 건너뜁니다."

    try:
        session_id = st.session_state.current_session_id
        title = st.session_state.session_title or "새 세션"
        ensure_session_row(
            client,
            session_id=session_id,
            user_id=user_id,
            title=title,
            processed_files=st.session_state.processed_files,
        )
        save_messages(client, session_id, user_id, st.session_state.chat_history)
        return None
    except Exception as exc:
        LOGGER.error("자동 저장 실패: %s", exc)
        return f"자동 저장 중 오류가 발생했습니다: {exc}"


def generate_session_title(user_query: str, answer: str) -> str:
    try:
        llm = get_llm(temperature=0.3)
        messages = [
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"질문:\n{user_query}\n\n답변:\n{answer[:800]}"
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        title = remove_separators(str(content)).replace('"', "").replace("'", "")
        title = re.sub(r"\s+", " ", title).strip()
        return title[:30] if title else "저장된 세션"
    except Exception as exc:
        LOGGER.warning("세션 제목 생성 실패: %s", exc)
        return (user_query[:20] + "...") if len(user_query) > 20 else user_query or "저장된 세션"


def copy_vector_documents(
    supabase: Client,
    source_session_id: str,
    target_session_id: str,
) -> None:
    response = (
        supabase.table("vector_documents")
        .select("content, metadata, embedding, file_name")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return

    payloads = []
    for row in rows:
        payloads.append(
            {
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": row["embedding"],
                "file_name": row["file_name"],
                "session_id": target_session_id,
            }
        )

    for start in range(0, len(payloads), VECTOR_BATCH_SIZE):
        batch = payloads[start : start + VECTOR_BATCH_SIZE]
        supabase.table("vector_documents").insert(batch).execute()


def insert_session_snapshot(supabase: Client) -> tuple[str | None, str | None]:
    """현재 상태를 새 세션으로 INSERT 저장한다."""
    try:
        user_id = require_user_id()
        if not st.session_state.chat_history:
            return None, "저장할 대화가 없습니다. 먼저 질문을 해주세요."

        first_user = next(
            (m["content"] for m in st.session_state.chat_history if m["role"] == "user"),
            "",
        )
        first_assistant = next(
            (
                m["content"]
                for m in st.session_state.chat_history
                if m["role"] == "assistant"
            ),
            "",
        )
        title = generate_session_title(first_user, first_assistant)

        new_session_id = str(uuid.uuid4())
        source_session_id = st.session_state.current_session_id

        auto_save_session(supabase)

        supabase.table("chat_sessions").insert(
            {
                "id": new_session_id,
                "user_id": user_id,
                "title": title,
                "processed_files": st.session_state.processed_files,
            }
        ).execute()

        save_messages(supabase, new_session_id, user_id, st.session_state.chat_history)
        copy_vector_documents(supabase, source_session_id, new_session_id)

        st.session_state.current_session_id = new_session_id
        st.session_state.session_title = title
        st.session_state.last_loaded_label = f"{title}::{new_session_id}"
        queue_session_selectbox(f"{title}::{new_session_id}")
        return title, None
    except Exception as exc:
        LOGGER.error("세션 저장 실패: %s", exc)
        return None, f"세션 저장 중 오류가 발생했습니다: {exc}"


def delete_session(supabase: Client, session_id: str) -> str | None:
    try:
        user_id = require_user_id()
        if not session_belongs_to_user(supabase, session_id, user_id):
            return "본인 소유의 세션만 삭제할 수 있습니다."
        (
            supabase.table("chat_sessions")
            .delete()
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        return None
    except Exception as exc:
        LOGGER.error("세션 삭제 실패: %s", exc)
        return f"세션 삭제 중 오류가 발생했습니다: {exc}"


def get_vector_file_names(
    supabase: Client,
    session_id: str | None = None,
    user_id: str | None = None,
) -> list[str]:
    if session_id:
        query = (
            supabase.table("vector_documents")
            .select("file_name")
            .eq("session_id", session_id)
        )
        response = query.execute()
        return sorted(
            {row["file_name"] for row in (response.data or []) if row.get("file_name")}
        )

    # 세션 미지정 시: 해당 사용자의 세션에 속한 파일만 조회
    if not user_id:
        return []
    sessions = list_sessions(supabase, user_id)
    session_ids = [row["id"] for row in sessions]
    if not session_ids:
        return []

    names: set[str] = set()
    for sid in session_ids:
        response = (
            supabase.table("vector_documents")
            .select("file_name")
            .eq("session_id", sid)
            .execute()
        )
        for row in response.data or []:
            if row.get("file_name"):
                names.add(row["file_name"])
    return sorted(names)


def count_vectors(supabase: Client, session_id: str) -> int:
    response = (
        supabase.table("vector_documents")
        .select("id", count="exact")
        .eq("session_id", session_id)
        .execute()
    )
    if response.count is not None:
        return response.count
    return len(response.data or [])


# ──────────────────────────────────────────────
# PDF / 벡터 저장
# ──────────────────────────────────────────────
def store_documents_to_supabase(
    supabase: Client,
    session_id: str,
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
) -> str | None:
    """file_name/session_id를 명시해서 vector_documents에 직접 저장한다."""
    try:
        rows: list[dict[str, Any]] = []
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        for chunk, vector in zip(chunks, vectors):
            file_name = (
                chunk.metadata.get("file_name")
                or chunk.metadata.get("source_file")
                or "unknown.pdf"
            )
            metadata = dict(chunk.metadata or {})
            metadata["file_name"] = file_name
            metadata["session_id"] = session_id
            rows.append(
                {
                    "content": chunk.page_content,
                    "metadata": metadata,
                    "embedding": vector,
                    "file_name": file_name,
                    "session_id": session_id,
                }
            )

        for start in range(0, len(rows), VECTOR_BATCH_SIZE):
            batch = rows[start : start + VECTOR_BATCH_SIZE]
            supabase.table("vector_documents").insert(batch).execute()
        return None
    except Exception as exc:
        LOGGER.error("벡터 저장 실패: %s", exc)
        return f"벡터 DB 저장 중 오류가 발생했습니다: {exc}"


def process_pdf_files(
    uploaded_files: list[Any],
    session_id: str,
) -> tuple[list[str], str | None]:
    if not OPENAI_API_KEY:
        return [], (
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "Streamlit secrets 또는 .env 파일에 키를 추가해 주세요."
        )

    supabase = get_supabase_client()
    if supabase is None:
        return [], "SUPABASE_URL 또는 SUPABASE_ANON_KEY가 설정되지 않았습니다."

    try:
        user_id = require_user_id()
    except PermissionError:
        return [], "로그인이 필요합니다."

    all_chunks: list[Document] = []
    processed_names: list[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_name"] = uploaded_file.name
                doc.metadata["source_file"] = uploaded_file.name
                doc.metadata["session_id"] = session_id

            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["file_name"] = uploaded_file.name
                chunk.metadata["session_id"] = session_id
            all_chunks.extend(chunks)
            processed_names.append(uploaded_file.name)
        except Exception as exc:
            LOGGER.error("PDF 처리 실패 (%s): %s", uploaded_file.name, exc)
            return [], f"'{uploaded_file.name}' 파일 처리 중 오류가 발생했습니다."
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_chunks:
        return [], "PDF에서 텍스트를 추출하지 못했습니다."

    ensure_session_row(
        supabase,
        session_id=session_id,
        user_id=user_id,
        title=st.session_state.session_title or "새 세션",
        processed_files=list(
            dict.fromkeys(st.session_state.processed_files + processed_names)
        ),
    )

    embeddings = get_embeddings()
    error = store_documents_to_supabase(supabase, session_id, all_chunks, embeddings)
    if error:
        return [], error

    return processed_names, None


def retrieve_documents(
    supabase: Client,
    session_id: str,
    query: str,
    k: int = 10,
) -> list[Document]:
    """match_vector_documents RPC로 세션 필터 검색. 실패 시 테이블 조회로 대체."""
    user_id = current_user_id()
    if user_id and not session_belongs_to_user(supabase, session_id, user_id):
        LOGGER.warning("다른 사용자 세션에 대한 벡터 검색 시도 차단: %s", session_id)
        return []

    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)

    try:
        response = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = response.data or []
        docs: list[Document] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": metadata}
            metadata["file_name"] = row.get("file_name") or metadata.get("file_name")
            metadata["session_id"] = row.get("session_id") or session_id
            docs.append(
                Document(page_content=row.get("content") or "", metadata=metadata)
            )
        return docs
    except Exception as exc:
        LOGGER.warning("RPC 검색 실패, 대체 검색 사용: %s", exc)

    try:
        response = (
            supabase.table("vector_documents")
            .select("content, metadata, file_name, session_id, embedding")
            .eq("session_id", session_id)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return []

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return -1.0
            return dot / (norm_a * norm_b)

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            emb = row.get("embedding")
            if isinstance(emb, str):
                emb = json.loads(emb)
            if not isinstance(emb, list):
                continue
            scored.append((cosine(query_embedding, emb), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        docs = []
        for _, row in scored[:k]:
            metadata = row.get("metadata") or {}
            metadata["file_name"] = row.get("file_name")
            metadata["session_id"] = row.get("session_id")
            docs.append(Document(page_content=row.get("content") or "", metadata=metadata))
        return docs
    except Exception as fallback_exc:
        LOGGER.error("대체 검색도 실패: %s", fallback_exc)
        return []


# ──────────────────────────────────────────────
# LLM 응답
# ──────────────────────────────────────────────
def generate_follow_up_questions(llm: Any, user_query: str, answer: str) -> list[str]:
    try:
        messages = [
            SystemMessage(content=FOLLOW_UP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"사용자 질문:\n{user_query}\n\n"
                    f"AI 답변:\n{answer}\n\n"
                    "위 대화를 바탕으로 후속 질문 3개를 생성하세요."
                )
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        questions = parse_follow_up_questions(str(content))
        while len(questions) < 3:
            questions.append("이 주제에 대해 더 자세히 설명해 주실 수 있나요?")
        return questions[:3]
    except Exception as exc:
        LOGGER.warning("후속 질문 생성 실패: %s", exc)
        return [
            "이 내용을 더 쉽게 설명해 주실 수 있나요?",
            "관련된 다른 주제도 알려 주실 수 있나요?",
            "실생활에서 어떻게 활용할 수 있나요?",
        ]


def stream_llm_response(llm: Any, messages: list[Any], placeholder: Any) -> str:
    full_response = ""
    for chunk in llm.stream(messages):
        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if piece:
            full_response += piece
            placeholder.markdown(remove_separators(full_response))
    return remove_separators(full_response)


def generate_direct_llm_answer(
    llm: Any,
    user_query: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
) -> str:
    memory_context = format_memory_context(conversation_memory)
    messages = [
        SystemMessage(content=DIRECT_LLM_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"현재 질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def generate_rag_answer(
    llm: Any,
    user_query: str,
    conversation_memory: list[dict[str, str]],
    placeholder: Any,
) -> str:
    supabase = get_supabase_client()
    if supabase is None:
        warning = "⚠️ Supabase 연결 정보가 없어 RAG 검색을 할 수 없습니다."
        placeholder.warning(warning)
        return warning

    docs = retrieve_documents(
        supabase,
        st.session_state.current_session_id,
        user_query,
        k=10,
    )
    if not docs:
        warning = (
            "⚠️ 현재 세션의 벡터 DB에서 관련 문서를 찾지 못했습니다. "
            "PDF를 업로드하고 '파일 처리하기'를 먼저 실행해 주세요."
        )
        placeholder.warning(warning)
        return warning

    context = "\n\n".join(doc.page_content for doc in docs)
    memory_context = format_memory_context(conversation_memory)
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"이전 대화:\n{memory_context}\n\n"
                f"참고 문서:\n{context}\n\n"
                f"질문:\n{user_query}"
            )
        ),
    ]
    answer = stream_llm_response(llm, messages, placeholder)
    follow_up = generate_follow_up_questions(llm, user_query, answer)
    final_answer = append_follow_up_section(answer, follow_up)
    placeholder.markdown(final_answer)
    return final_answer


def update_conversation_memory(
    user_query: str,
    assistant_answer: str,
    conversation_memory: list[dict[str, str]],
) -> None:
    conversation_memory.append({"role": "user", "content": user_query})
    conversation_memory.append({"role": "assistant", "content": assistant_answer})
    if len(conversation_memory) > 50:
        del conversation_memory[:-50]


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.9rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.6rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.35rem !important; }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }

        div[data-testid="stChatInput"] textarea {
            font-size: 1.2rem !important;
        }

        div.stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }

        div.stButton > button:hover {
            background-color: #ff85c1 !important;
            color: white !important;
        }

        .ena-header-title {
            text-align: center !important;
            font-size: 2.4rem !important;
            line-height: 1.1 !important;
            font-weight: 700 !important;
            margin: 0.5rem 0 1rem 0 !important;
        }

        .ena-header-title .ena-blue {
            color: #1f77b4 !important;
            font-size: 2.4rem !important;
        }

        .ena-header-title .ena-gold {
            color: #ffd700 !important;
            font-size: 2.4rem !important;
        }

        .ena-auth-box {
            max-width: 480px;
            margin: 1.5rem auto;
            padding: 1.5rem;
            border: 1px solid #eee;
            border-radius: 12px;
            background: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with left_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("## 📚")

    with center_col:
        st.markdown(
            """
            <div class="ena-header-title">
                <span class="ena-blue">ENA</span>
                <span class="ena-gold">RAG 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        if st.session_state.get("logged_in"):
            st.caption(f"👤 {st.session_state.get('display_name', '')}")
            if st.button("로그아웃", key="logout_header"):
                logout_user()
                st.rerun()


def queue_session_selectbox(label: str) -> None:
    """selectbox 위젯 생성 이후에 key를 바꾸면 오류가 나므로,
    다음 렌더 시작 시 적용되도록 pending에 넣는다.
    """
    st.session_state.selected_session_label = label
    st.session_state._pending_session_selectbox = label


def apply_pending_session_selectbox(session_options: list[str]) -> None:
    """selectbox 생성 전에 pending 선택을 반영한다."""
    if "_pending_session_selectbox" in st.session_state:
        pending = st.session_state._pending_session_selectbox
        del st.session_state["_pending_session_selectbox"]
        if pending in session_options:
            st.session_state.session_selectbox = pending
        else:
            st.session_state.session_selectbox = "새로운 작업 세션"
        return

    if "session_selectbox" not in st.session_state:
        label = st.session_state.selected_session_label
        st.session_state.session_selectbox = (
            label if label in session_options else "새로운 작업 세션"
        )
    elif st.session_state.session_selectbox not in session_options:
        st.session_state.session_selectbox = "새로운 작업 세션"


def reset_local_session(new_id: str | None = None) -> None:
    st.session_state.current_session_id = new_id or str(uuid.uuid4())
    st.session_state.session_title = "새 세션"
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.has_vectors = False
    st.session_state.last_loaded_label = "새로운 작업 세션"
    queue_session_selectbox("새로운 작업 세션")


def init_session_state() -> None:
    defaults = {
        "logged_in": False,
        "user_id": None,
        "login_id": None,
        "display_name": None,
        "chat_history": [],
        "conversation_memory": [],
        "processed_files": [],
        "current_session_id": str(uuid.uuid4()),
        "session_title": "새 세션",
        "has_vectors": False,
        "selected_session_label": "새로운 작업 세션",
        "last_loaded_label": "새로운 작업 세션",
        "initialized": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_session_into_ui(
    supabase: Client,
    session_id: str,
    title: str,
) -> str | None:
    try:
        user_id = require_user_id()
        if not session_belongs_to_user(supabase, session_id, user_id):
            return "본인 소유의 세션만 로드할 수 있습니다."

        messages = load_messages(supabase, session_id, user_id)
        session_rows = (
            supabase.table("chat_sessions")
            .select("processed_files, title")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        processed_files: list[str] = []
        if session_rows.data:
            files = session_rows.data[0].get("processed_files") or []
            if isinstance(files, str):
                files = json.loads(files)
            processed_files = list(files)
            title = session_rows.data[0].get("title") or title

        st.session_state.current_session_id = session_id
        st.session_state.session_title = title
        st.session_state.chat_history = messages
        st.session_state.conversation_memory = messages[-50:]
        st.session_state.processed_files = processed_files
        st.session_state.has_vectors = count_vectors(supabase, session_id) > 0
        label = f"{title}::{session_id}"
        st.session_state.last_loaded_label = label
        queue_session_selectbox(label)
        return None
    except Exception as exc:
        LOGGER.error("세션 로드 실패: %s", exc)
        return f"세션 로드 중 오류가 발생했습니다: {exc}"


def render_auth_screen() -> None:
    """로그인 / 회원가입 화면 (Supabase Auth 미사용)."""
    st.subheader("🔐 로그인 / 회원가입")
    st.caption("Supabase Auth가 아닌 DB의 user 테이블로 계정을 관리합니다.")

    for msg in missing_env_messages():
        st.warning(msg)

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        with st.form("login_form"):
            login_id = st.text_input("아이디 (login_id)", key="login_id_input")
            password = st.text_input(
                "비밀번호", type="password", key="login_password_input"
            )
            submitted = st.form_submit_button("로그인")
            if submitted:
                supabase = get_supabase_client()
                if supabase is None:
                    st.error("SUPABASE_URL / SUPABASE_ANON_KEY가 필요합니다.")
                else:
                    user, error = authenticate_user(supabase, login_id, password)
                    if error:
                        st.error(error)
                    elif user:
                        set_logged_in_user(user)
                        st.success(f"{user['display_name']}님, 환영합니다!")
                        st.rerun()

    with tab_signup:
        with st.form("signup_form"):
            login_id = st.text_input("아이디 (login_id)", key="signup_id_input")
            display_name = st.text_input("표시 이름 (선택)", key="signup_name_input")
            password = st.text_input(
                "비밀번호", type="password", key="signup_password_input"
            )
            password2 = st.text_input(
                "비밀번호 확인", type="password", key="signup_password2_input"
            )
            submitted = st.form_submit_button("회원가입")
            if submitted:
                if password != password2:
                    st.error("비밀번호가 일치하지 않습니다.")
                else:
                    supabase = get_supabase_client()
                    if supabase is None:
                        st.error("SUPABASE_URL / SUPABASE_ANON_KEY가 필요합니다.")
                    else:
                        user, error = register_user(
                            supabase, login_id, password, display_name
                        )
                        if error:
                            st.error(error)
                        elif user:
                            set_logged_in_user(
                                {
                                    "id": user["id"],
                                    "login_id": user["login_id"],
                                    "display_name": user.get("display_name")
                                    or user["login_id"],
                                }
                            )
                            st.success("회원가입이 완료되었습니다. 바로 이용할 수 있습니다.")
                            st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])


def render_sidebar() -> str:
    st.sidebar.header("⚙️ 설정")
    st.sidebar.caption(
        f"로그인: {st.session_state.get('display_name') or st.session_state.get('login_id')}"
    )

    model_name = st.sidebar.radio(
        "LLM 모델 선택",
        [MODEL_NAME],
        index=0,
    )

    rag_option = st.sidebar.radio(
        "RAG (PDF 검색) 선택",
        ["사용 안 함", "RAG 사용"],
        index=0,
    )

    uploaded_files = st.sidebar.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("파일 처리하기"):
        if not uploaded_files:
            st.sidebar.warning("업로드할 PDF 파일을 선택해 주세요.")
        else:
            env_errors = missing_env_messages()
            if env_errors:
                for msg in env_errors:
                    st.sidebar.error(msg)
            else:
                with st.spinner("PDF 파일을 처리하고 Supabase에 저장하는 중..."):
                    processed_names, error_message = process_pdf_files(
                        uploaded_files,
                        st.session_state.current_session_id,
                    )
                if error_message:
                    st.sidebar.error(error_message)
                else:
                    st.session_state.processed_files = list(
                        dict.fromkeys(
                            st.session_state.processed_files + processed_names
                        )
                    )
                    st.session_state.has_vectors = True
                    save_error = auto_save_session()
                    if save_error:
                        st.sidebar.warning(save_error)
                    st.sidebar.success(
                        f"{len(processed_names)}개 PDF 파일 처리 및 세션 자동 저장 완료"
                    )

    if st.session_state.processed_files:
        st.sidebar.write("처리된 파일:")
        for file_name in st.session_state.processed_files:
            st.sidebar.write(f"- {file_name}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗂 세션 관리")

    supabase = get_supabase_client()
    user_id = current_user_id()
    session_options = ["새로운 작업 세션"]
    session_map: dict[str, dict[str, Any]] = {}

    if supabase is not None and user_id:
        try:
            for row in list_sessions(supabase, user_id):
                label = f"{row['title']}::{row['id']}"
                session_options.append(label)
                session_map[label] = row
        except Exception as exc:
            st.sidebar.error(f"세션 목록 조회 실패: {exc}")

    apply_pending_session_selectbox(session_options)

    selected_label = st.sidebar.selectbox(
        "세션 선택",
        options=session_options,
        key="session_selectbox",
    )
    st.session_state.selected_session_label = selected_label

    # 풀다운에서 세션을 고르면 자동 로드
    if (
        selected_label != st.session_state.last_loaded_label
        and selected_label != "새로운 작업 세션"
        and supabase is not None
        and selected_label in session_map
    ):
        row = session_map[selected_label]
        error = load_session_into_ui(supabase, row["id"], row["title"])
        if error:
            st.sidebar.error(error)
        else:
            st.sidebar.info(f"세션 자동 로드: {row['title']}")
            st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("세션저장"):
            if supabase is None:
                st.sidebar.error("Supabase 키가 없어 저장할 수 없습니다.")
            else:
                with st.spinner("세션을 INSERT 저장하는 중..."):
                    title, error = insert_session_snapshot(supabase)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"세션 저장 완료: {title}")
                    st.rerun()

    with col2:
        if st.button("세션로드"):
            if selected_label == "새로운 작업 세션":
                st.sidebar.warning("로드할 세션을 선택해 주세요.")
            elif supabase is None:
                st.sidebar.error("Supabase 키가 없어 로드할 수 없습니다.")
            elif selected_label not in session_map:
                st.sidebar.error("선택한 세션을 찾을 수 없습니다.")
            else:
                row = session_map[selected_label]
                error = load_session_into_ui(supabase, row["id"], row["title"])
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"세션 로드 완료: {row['title']}")
                    st.rerun()

    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("세션삭제"):
            target_id = st.session_state.current_session_id
            if selected_label in session_map:
                target_id = session_map[selected_label]["id"]
            if supabase is None:
                st.sidebar.error("Supabase 키가 없어 삭제할 수 없습니다.")
            else:
                error = delete_session(supabase, target_id)
                if error:
                    st.sidebar.error(error)
                else:
                    reset_local_session()
                    st.sidebar.success("선택한 세션을 삭제했습니다.")
                    st.rerun()

    with col4:
        if st.button("화면초기화"):
            reset_local_session()
            st.sidebar.success("화면을 초기화했습니다.")
            st.rerun()

    if st.sidebar.button("vectordb"):
        if supabase is None or not user_id:
            st.sidebar.error("Supabase 키가 없거나 로그인되지 않았습니다.")
        else:
            files = get_vector_file_names(
                supabase,
                session_id=st.session_state.current_session_id,
                user_id=user_id,
            )
            if not files:
                all_files = get_vector_file_names(
                    supabase, session_id=None, user_id=user_id
                )
                if not all_files:
                    st.sidebar.info("vectordb에 저장된 파일이 없습니다.")
                else:
                    st.sidebar.warning("현재 세션에는 파일이 없습니다. 내 전체 목록:")
                    for name in all_files:
                        st.sidebar.write(f"- {name}")
            else:
                st.sidebar.success("현재 세션 vectordb 파일 목록")
                for name in files:
                    st.sidebar.write(f"- {name}")

    if st.sidebar.button("로그아웃", key="logout_sidebar"):
        logout_user()
        st.rerun()

    st.sidebar.subheader("현재 설정")
    st.sidebar.text(f"모델: {model_name}")
    st.sidebar.text(f"RAG: {rag_option}")
    st.sidebar.text(f"사용자: {st.session_state.get('login_id')}")
    st.sidebar.text(f"세션 제목: {st.session_state.session_title}")
    st.sidebar.text(f"세션 ID: {st.session_state.current_session_id[:8]}...")
    st.sidebar.text(f"처리된 파일 수: {len(st.session_state.processed_files)}")
    st.sidebar.text(f"대화 기록 수: {len(st.session_state.chat_history)}")
    st.sidebar.text(f"벡터 준비: {'예' if st.session_state.has_vectors else '아니오'}")

    return rag_option


def handle_user_input(user_query: str, rag_option: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    if not current_user_id():
        error_message = "⚠️ 로그인이 필요합니다."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error_message}
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        return

    if not OPENAI_API_KEY:
        error_message = (
            "⚠️ OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
            "Streamlit secrets 또는 `.env` 파일에 키를 설정한 뒤 다시 시도해 주세요."
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error_message}
        )
        update_conversation_memory(
            user_query, error_message, st.session_state.conversation_memory
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        return

    if rag_option == "RAG 사용" and (not SUPABASE_URL or not SUPABASE_ANON_KEY):
        error_message = "⚠️ RAG 사용을 위해 SUPABASE_URL / SUPABASE_ANON_KEY가 필요합니다."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": error_message}
        )
        update_conversation_memory(
            user_query, error_message, st.session_state.conversation_memory
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            llm = get_llm()
            if rag_option == "RAG 사용":
                final_answer = generate_rag_answer(
                    llm=llm,
                    user_query=user_query,
                    conversation_memory=st.session_state.conversation_memory,
                    placeholder=placeholder,
                )
            else:
                final_answer = generate_direct_llm_answer(
                    llm=llm,
                    user_query=user_query,
                    conversation_memory=st.session_state.conversation_memory,
                    placeholder=placeholder,
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_answer}
            )
            update_conversation_memory(
                user_query, final_answer, st.session_state.conversation_memory
            )

            user_count = sum(
                1 for m in st.session_state.chat_history if m["role"] == "user"
            )
            if user_count == 1 and st.session_state.session_title in ("새 세션", "", None):
                st.session_state.session_title = generate_session_title(
                    user_query, final_answer
                )

            save_error = auto_save_session()
            if save_error and "연결 정보" not in save_error:
                st.warning(save_error)

        except Exception as exc:
            LOGGER.error("답변 생성 중 오류: %s", exc)
            friendly_message = (
                "답변 생성 중 오류가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": friendly_message}
            )
            update_conversation_memory(
                user_query, friendly_message, st.session_state.conversation_memory
            )
            placeholder.error(friendly_message)


def main() -> None:
    st.set_page_config(
        page_title=CHATBOT_TITLE,
        page_icon="📚",
        layout="wide",
    )

    refresh_secrets()
    inject_custom_css()
    init_session_state()
    render_header()

    if not st.session_state.get("logged_in"):
        render_auth_screen()
        return

    for msg in missing_env_messages():
        st.warning(msg)

    rag_option = render_sidebar()
    render_chat_history()

    user_query = st.chat_input("메시지를 입력하세요...")
    if user_query:
        handle_user_input(user_query, rag_option)


if __name__ == "__main__":
    main()
