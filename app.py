# app.py — OathLink Backend (MVP, fixed)
import os, time, json, uuid, sqlite3, pathlib, traceback
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
import json
from fastapi.responses import JSONResponse

APP_NAME = "OathLink Backend"
class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
    def render(self, content) -> bytes:
        # 確保不逃脫中文，並用 UTF-8 編碼
        return json.dumps(content, ensure_ascii=False).encode("utf-8")
app = FastAPI(title=APP_NAME, version="0.3.0", default_response_class=UTF8JSONResponse)
# --- UTF-8 Middleware ---
@app.middleware("http")
async def force_utf8_json(request, call_next):
    resp = await call_next(request)
    ctype = resp.headers.get("content-type", "")
    if ctype and ctype.startswith("application/json") and "charset=" not in ctype.lower():
        resp.headers["content-type"] = "application/json; charset=utf-8"
    return resp
# --- end ---
# ===== Config =====
DB_PATH        = os.getenv("DB_PATH", "/app/data/memory.db")
AUTH_TOKEN     = os.getenv("AUTH_TOKEN")                     # 若設定才要求驗證
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                 # 若設定才嘗試呼叫
OPENAI_BASE    = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 人格模板（恆尊稱：願主／師父／您）
PERSONA_PROMPT = (
    "您是『OathLink 穩定語風人格助手（無蘊）』。"
    "規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；"
    "必要時先標註風險與前置條件。"
)

# 確保 DB 目錄
pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ===== DB Helpers (FIX: 確保在模組頂層存在) =====
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS memory(
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT,
            ts REAL NOT NULL
        )
    """)
    return con

def _write_memory(content: str, tags: Optional[List[str]]) -> str:
    mid = str(uuid.uuid4())
    with _conn() as con:
        con.execute(
            "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
            (mid, content, json.dumps(tags or []), time.time()),
        )
    return mid

def _search_memory(q: str, top_k: int) -> List[Dict[str, Any]]:
    q = (q or "").strip()
    if not q:
        return []
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

# ===== Utility =====
@app.get("/", summary="Root")
def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "paths": ["/health", "/memory/write", "/memory/search", "/compose", "/routes", "/debug/peek"],
        "ts": time.time(),
    }

@app.get("/routes", summary="List routes")
def routes():
    from fastapi.routing import APIRoute
    return sorted([f"{sorted(r.methods)[0]} {r.path}" for r in app.routes if isinstance(r, APIRoute)])

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
def _build_prompt(req: ComposeReq, hits: List[Dict[str, Any]]) -> Dict[str, str]:
    ctx = "\n".join([f"- {h['content']}" for h in hits]) or "（無匹配記憶）"
    system = PERSONA_PROMPT
    user = f"【輸入】\n{req.input}\n\n【可用記憶】\n{ctx}\n\n請以固定語風輸出最終回覆。"
    return {"system": system, "user": user}

def _call_openai_chat(messages: List[Dict[str, str]]) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        import requests
        url = f"{OPENAI_BASE.rstrip('/')}/v1/chat/completions"
        payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
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
    # 1) 記憶檢索
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
        out = (
            "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
            "1) 已整合輸入與歷史記憶。\n"
            "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
        )

    return {
        "ok": True,
        "prompt": p,
        "context_hits": hits,
        "output": out,
        "model_used": (OPENAI_MODEL if OPENAI_API_KEY else "local-fallback"),
        "ts": time.time(),
    }

# ===== Debug =====
@app.get("/debug/peek", summary="Inspect DB quick view")
def debug_peek():
    out = {"db_path": DB_PATH, "now": time.time()}
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    def q(sql: str):
        try:
            cur = con.execute(sql); return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            return {"error": str(e)}
    out["count_memory"] = q("SELECT COUNT(*) AS n FROM memory")
    out["last5_memory"] = q("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC LIMIT 5")
    con.close()
    return out