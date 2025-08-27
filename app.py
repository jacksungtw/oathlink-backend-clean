mport os, json, sqlite3, time, re, unicodedata
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# app.py 開頭加：
from fastapi.middleware.cors import CORSMiddleware

# —— CORS：務必在註冊任何路由之前加上（含預檢設定）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 開發期先放開，正式可改成您的前端域名
    allow_origin_regex=".*",          # 搭配 *，確保所有來源
    allow_credentials=True,
    allow_methods=["*"],              # 需要讓 OPTIONS 自動通過
    allow_headers=["*"],              # 需要讓自定義標頭（如 X-Auth-Token）通過
    expose_headers=["*"],
    max_age=86400
)

# —— 保險：通用 OPTIONS（某些環境下預檢仍會被路由層擋，這個保證 204）
@app.options("/{rest_of_path:path}")
def preflight(rest_of_path: str):
    return Response(status_code=204)
    
APP_VERSION   = "0.4.0"
AUTH_TOKEN    = os.getenv("X_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or ""
DB_PATH       = os.getenv("DB_PATH", "data/oathlink.db")
SEARCH_MODE   = (os.getenv("SEARCH_MODE") or "basic").lower()  # basic | fts
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="OathLink Backend", version=APP_VERSION)

# -----------------------------
# CORS（← 新增 / 重要）
# -----------------------------
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 前端純檔案開啟（PWA / file://）時，部分瀏覽器不帶 Origin；保留 '*' 便於測試
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 資料庫基礎
# -----------------------------
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
    # 基本正規化，避免亂碼比對失敗
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
        "SELECT * FROM memory ORDER BY ts DESC LIMIT 100"
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
    return hits[:top_k]

# （可選）簡易 FTS：當 SEARCH_MODE=fts 時，用 LIKE 做全字/子字搜尋
def _search_fts(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id, content, tags, ts FROM memory "
        "WHERE content LIKE ? OR tags LIKE ? ORDER BY ts DESC LIMIT ?",
        (like, like, max(top_k, 1))
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
    if SEARCH_MODE == "fts":
        return _search_fts(q, top_k)
    return _search_basic(q, top_k)

# -----------------------------
# 請求 / 回應模型
# -----------------------------
class MemoryWriteReq(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

class ComposeReq(BaseModel):
    input: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)

# -----------------------------
# 驗證 Token
# -----------------------------
def _guard(token: Optional[str]):
    if AUTH_TOKEN:
        if not token or token != AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------
# 路由
# -----------------------------
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
    return {
        "ok": True,
        "routes": [r.path for r in app.router.routes],
        "ts": _now()
    }

@app.get("/health")
def health():
    return {"ok": True, "ts": _now()}

@app.post("/memory/write")
def memory_write(
    req: MemoryWriteReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    mid = _write_memory(_norm(req.content), req.tags)
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
    req: ComposeReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)

    q = req.input.strip()
    hits = _search_memory(q, req.top_k)

    system_prompt = "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    user_prompt = f"【輸入】\n{req.input}\n\n【可用記憶】\n" + (
        "\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）"
    ) + "\n\n請以固定語風輸出最終回覆。"

    # 簡易輸出（本地拼接）；若有 OPENAI_API_KEY，可改為呼叫雲端模型
    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。\n"
    )

    return {
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": ("gpt-4o-mini" if OPENAI_API_KEY else "local-fallback") if False else "gpt-4o-mini",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    }

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