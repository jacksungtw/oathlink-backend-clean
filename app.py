# app.py — OathLink Backend (MVP+ 搜尋切換 /compose 強化)
import os, time, json, uuid, sqlite3, pathlib
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

APP_NAME     = "OathLink Backend"
APP_VERSION  = "0.4.0"
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# ===== Config =====
DB_PATH        = os.getenv("DB_PATH", "/app/data/memory.db")
AUTH_TOKEN     = os.getenv("AUTH_TOKEN")                     # 若設定才要求驗證
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                 # 若設定才嘗試呼叫
OPENAI_BASE    = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 搜尋模式: like / fts
SEARCH_MODE    = os.getenv("SEARCH_MODE", "like").strip().lower()  # "like" (default) or "fts"

# 人格模板（恆尊稱：願主／師父／您）
BASE_PERSONA = (
    "您是『OathLink 穩定語風人格助手（無蘊）』。"
    "規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；"
    "必要時先標註風險與前置條件。"
)

# 確保 DB 目錄
pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ===== DB Helpers =====
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # 基表
    con.execute("""
        CREATE TABLE IF NOT EXISTS memory(
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT,
            ts REAL NOT NULL
        )
    """)
    # 如需 FTS5，建全文檢索表（獨立存內容，簡單可靠）
    if SEARCH_MODE == "fts":
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(id UNINDEXED, content, tags, tokenize='unicode61');
        """)
        # 對齊欄位，確保 id 唯一（刪除舊重覆）
        con.execute("CREATE TABLE IF NOT EXISTS _fts_guard(id TEXT PRIMARY KEY)")
    return con

def _fts_enabled() -> bool:
    if SEARCH_MODE != "fts":
        return False
    try:
        with _conn() as c:
            # 確認能 select FTS 表
            c.execute("SELECT count(*) FROM memory_fts")
        return True
    except Exception:
        return False

def _fts_upsert(c: sqlite3.Connection, mid: str, content: str, tags_json: str):
    # 先刪再插，避免同 id 重覆
    c.execute("DELETE FROM memory_fts WHERE id = ?", (mid,))
    c.execute("INSERT INTO memory_fts (id, content, tags) VALUES (?,?,?)", (mid, content, tags_json))

def _write_memory(content: str, tags: Optional[List[str]]) -> str:
    mid = str(uuid.uuid4())
    with _conn() as con:
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        con.execute(
            "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
            (mid, content, tags_json, time.time()),
        )
        if _fts_enabled():
            _fts_upsert(con, mid, content, tags_json)
    return mid

def _search_memory_like(q: str, top_k: int) -> List[Dict[str, Any]]:
    like = f"%{q}%"
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, content, tags, ts
            FROM memory
            WHERE content LIKE ? OR COALESCE(tags,'') LIKE ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (like, like, int(top_k)),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            tags = json.loads(r["tags"]) if r["tags"] else []
        except Exception:
            tags = (r["tags"] or "").split(",")
        out.append({"id": r["id"], "content": r["content"], "tags": tags, "ts": r["ts"]})
    return out

def _search_memory_fts(q: str, top_k: int) -> List[Dict[str, Any]]:
    # 基於 FTS5 MATCH + 以時間新到舊做 tie-break
    # 簡單 query：原樣送進 MATCH；若包含空白，FTS 會做 OR 拆詞
    with _conn() as con:
        rows = con.execute(
            """
            SELECT m.id, m.content, m.tags, m.ts
            FROM memory_fts f
            JOIN memory m ON m.id = f.id
            WHERE f.memory_fts MATCH ?
            ORDER BY m.ts DESC
            LIMIT ?
            """,
            (q, int(top_k)),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            tags = json.loads(r["tags"]) if r["tags"] else []
        except Exception:
            tags = (r["tags"] or "").split(",")
        out.append({"id": r["id"], "content": r["content"], "tags": tags, "ts": r["ts"]})
    return out

def _search_memory(q: str, top_k: int) -> List[Dict[str, Any]]:
    q = (q or "").strip()
    if not q:
        return []
    if _fts_enabled():
        try:
            return _search_memory_fts(q, top_k)
        except Exception:
            # FTS 失敗時退回 LIKE
            return _search_memory_like(q, top_k)
    return _search_memory_like(q, top_k)

# ===== Auth =====
def _check_auth(x_auth_token: Optional[str]):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ===== Schemas =====
class MemoryWriteReq(BaseModel):
    content: str
    tags: Optional[List[str]] = None

class ComposeReq(BaseModel):
    input: str
    tags: Optional[List[str]] = None
    top_k: int = 5
    # 新增參數
    tone: Optional[str] = None           # e.g. "正式/精煉/鼓勵/技術"
    output_format: Optional[str] = None  # e.g. "bullets/table/steps/paragraph"

# ===== Utility =====
@app.get("/", summary="Root")
def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "fts_enabled": _fts_enabled(),
        "paths": ["/health", "/memory/write", "/memory/search", "/compose", "/routes", "/debug/peek", "/debug/reindex_fts"],
        "ts": time.time(),
    }

@app.get("/routes", summary="List routes")
def routes():
    from fastapi.routing import APIRoute
    return sorted([f"{list(r.methods)[0]} {r.path}" for r in app.routes if isinstance(r, APIRoute)])

@app.get("/health", summary="Healthcheck")
def health():
    return {"ok": True, "ts": time.time()}

# ===== Memory API =====
@app.post("/memory/write", summary="Memory Write")
def memory_write(
    body: MemoryWriteReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _check_auth(x_auth_token)
    mid = _write_memory(body.content, body.tags)
    return {"ok": True, "id": mid}

@app.get("/memory/search", summary="Memory Search")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _check_auth(x_auth_token)
    results = _search_memory(q, top_k)
    return {"ok": True, "results": results, "ts": time.time()}

# ===== Compose =====
def _build_persona(req: ComposeReq) -> str:
    parts = [BASE_PERSONA]
    if req.tone:
        parts.append(f"語氣：{req.tone}。")
    if req.output_format:
        parts.append(f"輸出格式：{req.output_format}。")
    return "".join(parts)

def _build_prompt(req: ComposeReq, hits: List[Dict[str, Any]]) -> Dict[str, str]:
    ctx = "\n".join([f"- {h['content']}" for h in hits]) or "（無匹配記憶）"
    system = _build_persona(req)
    user = f"【輸入】\n{req.input}\n\n【可用記憶】\n{ctx}\n\n請以固定語風輸出最終回覆。"
    return {"system": system, "user": user}

def _call_openai_chat(messages: List[Dict[str, str]]) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        import requests
        url = f"{OPENAI_BASE.rstrip('/')}/v1/chat/completions"
        payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json; charset=utf-8"}
        r = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or None
    except Exception:
        return None

@app.post("/compose", summary="Compose with persona + memory")
def compose(
    req: ComposeReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _check_auth(x_auth_token)
    # 1) 記憶檢索（用 tags + input 作為查詢）
    query = (" ".join(req.tags or []) + " " + req.input).strip()
    hits = _search_memory(query or req.input, req.top_k)

    # 2) 組 prompt
    p = _build_prompt(req, hits)

    # 3) 嘗試呼叫模型；失敗則本地回覆
    out = _call_openai_chat([
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user"]},
    ])
    if out is None:
        # 本地安全回覆
        out = (
            "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
            "1) 已整合輸入與歷史記憶。\n"
            "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
        )
        # 若指定輸出格式，給最基本模板
        if req.output_format == "bullets":
            out += "- 要點一\n- 要點二\n- 要點三\n"
        elif req.output_format == "steps":
            out += "步驟1：...\n步驟2：...\n步驟3：...\n"

    return {
        "ok": True,
        "prompt": p,
        "context_hits": hits,
        "output": out,
        "model_used": (OPENAI_MODEL if OPENAI_API_KEY else "local-fallback"),
        "search_mode": SEARCH_MODE,
        "ts": time.time(),
    }

# ===== Debug =====
@app.get("/debug/peek", summary="Inspect DB quick view")
def debug_peek():
    out = {
        "db_path": DB_PATH,
        "now": time.time(),
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "fts_enabled": _fts_enabled(),
    }
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    def q(sql: str):
        try:
            cur = con.execute(sql); return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            return {"error": str(e)}
    out["count_memory"] = q("SELECT COUNT(*) AS n FROM memory")
    out["last5_memory"] = q("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC LIMIT 5")
    if _fts_enabled():
        out["count_memory_fts"] = q("SELECT COUNT(*) AS n FROM memory_fts")
    con.close()
    return out

@app.post("/debug/reindex_fts", summary="Rebuild FTS index from memory")
def debug_reindex_fts(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _