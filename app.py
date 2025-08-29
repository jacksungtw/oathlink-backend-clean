# app.py
import os, json, sqlite3, time, unicodedata
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, JSONResponse

APP_VERSION   = "0.4.0"
AUTH_TOKEN    = os.getenv("X_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or "abc123"
DB_PATH       = os.getenv("DB_PATH", "/app/data/memory.db")
SEARCH_MODE   = (os.getenv("SEARCH_MODE") or "fts").lower()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")

# —— 全域：預設 JSON 用 UTF-8（避免亂碼）
app = FastAPI(title="OathLink Backend", version=APP_VERSION,
              default_response_class=ORJSONResponse)

# —— 全域 CORS（放在路由前）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 上線可收斂到你的網域 / localhost:8501
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# —— 額外保險：所有路由的 OPTIONS
@app.options("/{full_path:path}", include_in_schema=False)
async def options_all(full_path: str):
    return JSONResponse({}, media_type="application/json; charset=utf-8",
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                        })

# ===== 資料庫 =====
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
)
""")
con.commit()

def _now() -> float: return time.time()

def _mk_id() -> str:
    import uuid; return str(uuid.uuid4())

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _write_memory(content: str, tags: List[str]) -> str:
    mid = _mk_id()
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, content, json.dumps(tags, ensure_ascii=False), _now())
    )
    con.commit()
    return mid

# —— 讓搜尋先穩：僅比對 content（避免 JSON/LIKE 邊角）
def _search_fts(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id, content, tags, ts FROM memory "
        "WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (like, max(1, top_k))
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
    return _search_fts(q, top_k) if SEARCH_MODE == "fts" else _search_fts(q, top_k)

# ===== 路由 =====
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "OathLink Backend",
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "fts_enabled": (SEARCH_MODE == "fts"),
        "paths": ["/health","/memory/write","/memory/search","/compose","/routes","/debug/peek"],
        "ts": _now()
    }

@app.get("/routes")
def routes():
    return {"ok": True, "routes": [r.path for r in app.router.routes], "ts": _now()}

@app.get("/health")
def health():
    # 顯式帶 charset，避免任何代理剝頭時造成亂碼
    return JSONResponse({"ok": True, "ts": _now()},
                        media_type="application/json; charset=utf-8",
                        headers={"Access-Control-Allow-Origin": "*"})

def _guard(token: Optional[str]):
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/memory/write")
def memory_write(
    req: Dict[str, Any] = Body(...),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    content = _norm(str(req.get("content","")))
    if not content: raise HTTPException(400, "content required")
    tags = req.get("tags") or []
    mid = _write_memory(content, tags)
    return {"ok": True, "id": mid}

@app.get("/memory/search")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    hits = _search_memory(q, top_k)
    return {"ok": True, "results": hits, "ts": _now()}

@app.post("/compose")
def compose(
    req: Dict[str, Any] = Body(...),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
    request: Request = None
):
    _guard(x_auth_token)
    q = _norm(str(req.get("input","")))
    top_k = int(req.get("top_k") or 5)
    hits = _search_memory(q, top_k)

    system_prompt = "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    user_prompt = f"【輸入】\n{q}\n\n【可用記憶】\n" + ( "\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）") + "\n\n請以固定語風輸出最終回覆。"

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
    )

    # 明確 UTF-8
    return JSONResponse({
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    }, media_type="application/json; charset=utf-8")
    
@app.get("/debug/peek")
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
    return {"ok": True, "rows": items, "ts": _now()}