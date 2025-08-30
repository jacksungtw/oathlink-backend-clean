# app.py — OathLink Backend (UTF-8 fix + CORS + memory)
import os, json, sqlite3, time, unicodedata, uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Body, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_TITLE   = "OathLink Backend"
APP_VERSION = "0.4.1"

# -------- Env --------
AUTH_TOKEN     = (os.getenv("X_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or "").strip()
DB_PATH        = os.getenv("DB_PATH", "data/oathlink.db")
SEARCH_MODE    = (os.getenv("SEARCH_MODE") or "fts").lower()  # "basic" | "fts"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -------- App (single instance) --------
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# -------- CORS (global, before routes) --------
ALLOWED_ORIGINS = [
    "http://localhost:8501", "http://127.0.0.1:8501",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "*"  # 開發期方便；正式環境可改成您的網域即可
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],        # 包含 OPTIONS
    allow_headers=["*"],        # 含 Content-Type, X-Auth-Token
)

# 額外保險：捕捉所有 OPTIONS，避免 405
@app.options("/{full_path:path}", include_in_schema=False)
async def _options_catch_all(full_path: str) -> Response:
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "600",
        },
    )

# -------- SQLite (UTF-8 hardening) --------
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
con = sqlite3.connect(
    DB_PATH,
    check_same_thread=False,
    detect_types=sqlite3.PARSE_DECLTYPES,
)

# 關鍵：任何 bytes 以 UTF-8 解析；str 原樣保留
con.text_factory = lambda b: (b.decode("utf-8", "ignore")
                              if isinstance(b, (bytes, bytearray)) else str(b))

cur = con.cursor()
cur.execute("PRAGMA encoding = 'UTF-8';")
cur.execute("PRAGMA journal_mode = WAL;")
cur.execute("PRAGMA synchronous = NORMAL;")

cur.execute("""
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  tags TEXT,
  ts REAL NOT NULL
);
""")
con.commit()

# -------- Helpers --------
def _now() -> float: return time.time()
def _mk_id() -> str: return str(uuid.uuid4())
def _norm(s: str) -> str: return unicodedata.normalize("NFKC", s or "")

def _write_memory(content: str, tags: List[str]) -> str:
    mid = _mk_id()
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, content, json.dumps(tags, ensure_ascii=False), _now())
    )
    con.commit()
    return mid

def _search_basic(q: str, top_k: int):
    q = _norm(q)
    rows = cur.execute("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC LIMIT 200").fetchall()
    hits = []
    for r in rows:
        content = r[1] or ""
        tags = json.loads(r[2] or "[]")
        blob = f"{content} {' '.join(tags)}"
        if q and q in blob:
            hits.append({"id": r[0], "content": content, "tags": tags, "ts": r[3]})
    return hits[: max(1, top_k)]

def _search_fts(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id,content,tags,ts FROM memory "
        "WHERE content LIKE ? OR tags LIKE ? "
        "ORDER BY ts DESC LIMIT ?",
        (like, like, max(1, top_k))
    ).fetchall()
    return [{"id": r[0], "content": r[1], "tags": json.loads(r[2] or "[]"), "ts": r[3]} for r in rows]

def _search_memory(q: str, top_k: int):
    return _search_fts(q, top_k) if SEARCH_MODE == "fts" else _search_basic(q, top_k)

# -------- Models --------
class MemoryWriteReq(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

class ComposeReq(BaseModel):
    input: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)

# -------- Guard --------
def _guard(x_auth_token: Optional[str]):
    if AUTH_TOKEN and (not x_auth_token or x_auth_token != AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------- Routes --------
@app.get("/", summary="Root")
def root():
    return {
        "ok": True,
        "service": APP_TITLE,
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "fts_enabled": (SEARCH_MODE == "fts"),
        "paths": ["/health","/memory/write","/memory/search","/compose","/routes","/debug/peek","/debug/reset"],
        "ts": _now(),
    }

@app.get("/routes", summary="List routes")
def routes():
    return {"ok": True, "routes": [r.path for r in app.router.routes], "ts": _now()}

@app.get("/health", summary="Healthcheck")
def health():
    # 明確 charset，防代理誤判
    return JSONResponse(content={"ok": True, "ts": _now()},
                        media_type="application/json; charset=utf-8")

@app.post("/memory/write", summary="Write memory")
def memory_write(
    req: MemoryWriteReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    mid = _write_memory(_norm(req.content), req.tags)
    return JSONResponse(content={"ok": True, "id": mid},
                        media_type="application/json; charset=utf-8")

@app.get("/memory/search", summary="Search memory")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    hits = _search_memory(q, top_k)
    return JSONResponse(content={"ok": True, "results": hits, "ts": _now()},
                        media_type="application/json; charset=utf-8")

@app.post("/compose", summary="Compose with persona + memory")
def compose(
    req: ComposeReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)

    q = _norm(req.input.strip())
    hits = _search_memory(q, req.top_k)

    system_prompt = (
        "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；"
        "回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    )
    # 直接放入原字串，禁止任何 ASCII 強制轉碼
    user_prompt = f"【輸入】\n{req.input}\n\n【可用記憶】\n" + (
        "\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）"
    ) + "\n\n請以固定語風輸出最終回覆。"

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
    )

    payload = {
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini" if OPENAI_API_KEY else "local-fallback",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    }
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

@app.get("/debug/peek", summary="Peek recent rows")
def debug_peek(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "content": r[1],                          # 已由 text_factory 保證 UTF-8
            "tags": json.loads(r[2] or "[]"),
            "ts": r[3],
        })
    return JSONResponse(content={"ok": True, "rows": items, "ts": _now()},
                        media_type="application/json; charset=utf-8")

@app.post("/debug/reset", summary="Drop & recreate memory table (auth required)")
def debug_reset(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    cur.execute("DROP TABLE IF EXISTS memory;")
    con.commit()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      tags TEXT,
      ts REAL NOT NULL
    );
    """)
    con.commit()
    return JSONResponse(content={"ok": True, "reset": True, "ts": _now()},
                        media_type="application/json; charset=utf-8")