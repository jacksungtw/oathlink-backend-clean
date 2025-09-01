import os, json, sqlite3, time, unicodedata, uuid
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# app.py
import json
from fastapi.responses import Response

def _json_utf8(obj) -> Response:
    return Response(
        content=json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json; charset=utf-8",
    )
APP_TITLE = "OathLink Backend"
APP_VERSION = "0.5.0"

# -----------------------------
# FastAPI App + CORS
# -----------------------------
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",  # 測試用，正式請收斂
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 環境變數
# -----------------------------
AUTH_TOKEN     = os.getenv("AUTH_TOKEN", "").strip()
DB_PATH        = os.getenv("DB_PATH", "data/oathlink.db")
SEARCH_MODE    = (os.getenv("SEARCH_MODE") or "fts").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------------
# 資料庫初始化（確保 UTF-8）
# -----------------------------
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.row_factory = sqlite3.Row
con.text_factory = lambda b: b.decode(errors="ignore") if isinstance(b, bytes) else str(b)
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

# -----------------------------
# 工具函式
# -----------------------------
def _now() -> float:
    return time.time()

def _mk_id() -> str:
    return str(uuid.uuid4())

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _write_memory(content: str, tags: List[str]) -> str:
    mid = _mk_id()
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, _norm(content), json.dumps(tags, ensure_ascii=False), _now())
    )
    con.commit()
    return mid

def _search_basic(q: str, top_k: int):
    q = _norm(q)
    rows = cur.execute("SELECT * FROM memory ORDER BY ts DESC LIMIT 100").fetchall()
    hits = []
    for r in rows:
        text = (r["content"] or "") + " " + " ".join(json.loads(r["tags"] or "[]"))
        if q in text:
            hits.append({
                "id": r["id"], "content": r["content"], "tags": json.loads(r["tags"] or "[]"), "ts": r["ts"]
            })
    return hits[:top_k]

def _search_fts(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? OR tags LIKE ? ORDER BY ts DESC LIMIT ?",
        (like, like, max(top_k, 1))
    ).fetchall()
    return [
        {"id": r["id"], "content": r["content"], "tags": json.loads(r["tags"] or "[]"), "ts": r["ts"]}
        for r in rows
    ]

def _search_memory(q: str, top_k: int):
    return _search_fts(q, top_k) if SEARCH_MODE == "fts" else _search_basic(q, top_k)

# -----------------------------
# 請求模型
# -----------------------------
class MemoryWriteReq(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

class ComposeReq(BaseModel):
    input: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)

# -----------------------------
# Token 驗證
# -----------------------------
def _guard(token: Optional[str]):
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------
# 路由
# -----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": APP_TITLE,
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "paths": ["/health","/memory/write","/memory/search","/compose","/routes","/debug/peek","/debug/reset"],
        "ts": _now()
    }

@app.get("/health")
def health():
    r = JSONResponse({"ok": True, "ts": _now()})
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

@app.get("/routes")
def routes():
    return {"ok": True, "routes": [r.path for r in app.router.routes], "ts": _now()}

@app.post("/memory/write")
def memory_write(req: MemoryWriteReq, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    mid = _write_memory(_norm(req.content), req.tags)  # _write_memory 內部不要 encode/decode
    return _json_utf8({"ok": True, "id": mid})

@app.get("/memory/search")
def memory_search(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=100),
                  x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    hits = _search_memory(q, top_k)
    return _json_utf8({"ok": True, "results": hits, "ts": _now()})

@app.post("/compose")
def compose(req: ComposeReq, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    hits = _search_memory(req.input.strip(), req.top_k)

    system_prompt = "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    user_prompt = f"【輸入】\n{req.input}\n\n【可用記憶】\n" + (
    "\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）"
    ) + "\n\n請以固定語風輸出最終回覆。"
    output = (
        f"願主，以下為基於您輸入與可用記憶所整理之回覆：\n"
        f"1) 已整合輸入：{req.input}\n"
        f"2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )

    return _json_utf8({
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    })
@app.get("/debug/peek")
def debug_peek():
    rows = cur.execute("SELECT * FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    items = [{
        "id": r["id"],
        "content": r["content"],                 # 不做 encode/decode
        "tags": json.loads(r["tags"] or "[]"),
        "ts": r["ts"]
    } for r in rows]
    return _json_utf8({"ok": True, "rows": items, "ts": _now()})

@app.post("/debug/reset")
def debug_reset(x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    cur.execute("DELETE FROM memory")
    con.commit()
    return {"ok": True, "reset": True, "ts": _now()}
 from fastapi import Request

@app.post("/debug/echo", summary="回顯原始請求以檢查編碼")
async def debug_echo(req: Request):
    raw = await req.body()  # 原始 bytes
    try:
        parsed = await req.json()
    except Exception:
        parsed = None
    return _json_utf8({
        "raw_len": len(raw),
        "raw_first_24_bytes": list(raw[:24]),
        "as_text_utf8": raw.decode("utf-8", errors="replace"),
        "parsed": parsed,
    })  
    # ===== Bundle Schemas =====
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BundleItem(BaseModel):
    id: Optional[str] = None
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    ts: float = Field(default_factory=_now)

class BundlePayload(BaseModel):
    bundle_version: str = "1.0"
    persona: str = "無蘊-敬語版"
    memory: List[BundleItem] = Field(default_factory=list)

# —— Helper：確保 UTF-8 正確寫入（關鍵：ensure_ascii=False）
def _json_dumps_utf8(o) -> str:
    return json.dumps(o, ensure_ascii=False)

# ===== Bundle: Export =====
@app.get("/bundle/export")
def bundle_export(
    since_ts: Optional[float] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    if since_ts:
        rows = cur.execute(
            "SELECT id, content, tags, ts FROM memory WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
            (since_ts, limit)
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
    mem = []
    for r in rows:
        mem.append({
            "id": r["id"],
            "content": r["content"],                       # 已用 TEXT，SQLite 內建 UTF-8
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return _json_utf8 (bundle) {
        "ok": True,
        "bundle_version": "1.0",
        "persona": "無蘊-敬語版",
        "memory": mem,
        "count": len(mem),
        "ts": _now()
    }

# ===== Bundle: Import =====
@app.post("/bundle/import")
def bundle_import(
    payload: BundlePayload,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)

    imported, skipped = 0, 0
    for item in payload.memory:
        content = _norm(item.content)
        if not content:
            skipped += 1
            continue
        tags = [t for t in (item.tags or []) if t]
        # 若帶 id 且已存在 → 略過避免覆蓋
        if item.id:
            ex = cur.execute("SELECT 1 FROM memory WHERE id = ?", (item.id,)).fetchone()
            if ex:
                skipped += 1
                continue
            mid = item.id
        else:
            mid = _mk_id()
        cur.execute(
            "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
            (mid, content, _json_dumps_utf8(tags), float(item.ts or _now()))
        )
        imported += 1
    con.commit()
    return {"ok": True, "imported": imported, "skipped": skipped, "ts": _now()}

# ===== Bundle: Preview =====
@app.get("/bundle/preview")
def bundle_preview(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    count = cur.execute("SELECT COUNT(*) c FROM memory").fetchone()["c"]
    sample = cur.execute(
        "SELECT content, tags, ts FROM memory ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    sample_obj = []
    if sample:
        sample_obj.append({
            "content": sample["content"],
            "tags": json.loads(sample["tags"] or "[]"),
            "ts": sample["ts"]
        })
    return {
        "ok": True,
        "persona": "無蘊-敬語版",
        "count_memory": count,
        "latest_ts": sample["ts"] if sample else None,
        "sample": sample_obj
    }