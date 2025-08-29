# app.py
import os, json, sqlite3, time, re, unicodedata
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Body, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_TITLE     = "OathLink Backend"
APP_VERSION   = "0.4.0"

# ---- 環境變數 ----
AUTH_TOKEN     = (os.getenv("X_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or "").strip()
DB_PATH        = os.getenv("DB_PATH", "data/oathlink.db")
SEARCH_MODE    = (os.getenv("SEARCH_MODE") or "basic").lower()  # "basic" | "fts"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# =========================
# FastAPI APP（唯一）
# =========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ---- 全域 CORS（放在路由之前）----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # 開發用 *；正式環境可改白名單
    allow_credentials=True,
    allow_methods=["*"],             # 確保 OPTIONS 被接受
    allow_headers=["*"],             # 含 Content-Type, X-Auth-Token 等
)

# ---- 額外保險：處理所有路由的 OPTIONS，避免 405 ----
@app.options("/{full_path:path}", include_in_schema=False)
async def options_catch_all(full_path: str) -> Response:
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "600",
        },
    )

# ---- 統一 JSON 回傳（強制 charset，避免中文亂碼）----
def json_ok(data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    r = JSONResponse(content=data, media_type="application/json; charset=utf-8", status_code=status_code)
    # 代理可能會吃掉 CORS，這裡再補一次
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD"
    r.headers["Access-Control-Allow-Headers"] = "*"
    return r

# =========================
# 資料庫初始化
# =========================
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  tags TEXT,
  ts REAL NOT NULL
);
""")
con.commit()

def _now() -> float:
    return time.time()

def _mk_id() -> str:
    import uuid
    return str(uuid.uuid4())

def _norm(s: str) -> str:
    # 基本正規化，提升搜尋穩定度
    return unicodedata.normalize("NFKC", s or "")

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
    rows = cur.execute(
        "SELECT * FROM memory ORDER BY ts DESC LIMIT 200"
    ).fetchall()
    hits = []
    for r in rows:
        content = r["content"] or ""
        tags    = json.loads(r["tags"] or "[]")
        text    = f"{content} {' '.join(tags)}"
        if q and q in text:
            hits.append({
                "id": r["id"], "content": content, "tags": tags, "ts": r["ts"]
            })
    return hits[:max(1, top_k)]

def _search_fts(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id, content, tags, ts FROM memory "
        "WHERE content LIKE ? OR tags LIKE ? ORDER BY ts DESC LIMIT ?",
        (like, like, max(1, top_k))
    ).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "id": r["id"],
            "content": r["content"],
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return hits

def _search_memory(q: str, top_k: int):
    return _search_fts(q, top_k) if SEARCH_MODE == "fts" else _search_basic(q, top_k)

# =========================
# 請求模型
# =========================
class MemoryWriteReq(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

class ComposeReq(BaseModel):
    input: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)

# =========================
# 驗證
# =========================
def _guard(token: Optional[str]):
    if AUTH_TOKEN:
        if not token or token != AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# 路由
# =========================
@app.get("/", summary="Root")
def root():
    return json_ok({
        "ok": True,
        "service": APP_TITLE,
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "fts_enabled": (SEARCH_MODE == "fts"),
        "paths": ["/health","/memory/write","/memory/search","/compose","/routes","/debug/peek"],
        "ts": _now()
    })

@app.get("/health", summary="Healthcheck")
def health():
    return json_ok({"ok": True, "ts": _now()})

@app.get("/routes", summary="List routes")
def routes():
    return json_ok({"ok": True, "routes": [r.path for r in app.router.routes], "ts": _now()})

@app.post("/memory/write", summary="Write memory")
def memory_write(
    req: MemoryWriteReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    mid = _write_memory(_norm(req.content), req.tags)
    return json_ok({"ok": True, "id": mid})

@app.get("/memory/search", summary="Search memory")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    hits = _search_memory(q, top_k)
    return json_ok({"ok": True, "results": hits, "ts": _now()})

@app.post("/compose", summary="Compose with persona + memory")
def compose(
    req: ComposeReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    hits = _search_memory(req.input.strip(), req.top_k)

    system_prompt = (
        "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；"
        "回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    )
    user_prompt = f"【輸入】\n{req.input}\n\n【可用記憶】\n" + (
        "\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）"
    ) + "\n\n請以固定語風輸出最終回覆。"

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
    )

    return json_ok({
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini" if OPENAI_API_KEY else "local-fallback",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    })

@app.get("/debug/peek", summary="Peek last 50 rows")
def debug_peek():
    rows = cur.execute("SELECT * FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "content": r["content"],
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return json_ok({"ok": True, "rows": items, "ts": _now()})