# app.py
import os, json, sqlite3, time, unicodedata
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ==============================
# 基本設定
# ==============================
APP_TITLE   = "OathLink Backend"
APP_VERSION = "0.5.0"

AUTH_TOKEN     = (os.getenv("X_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or "").strip()
DB_PATH        = os.getenv("DB_PATH", "data/oathlink.db")
SEARCH_MODE    = (os.getenv("SEARCH_MODE") or "basic").lower()  # basic | fts（目前預設 basic/LIKE）
DEFAULT_PERSONA= os.getenv("PERSONA_NAME", "無蘊-敬語版")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS（全域）
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",  # 開發期方便測試；正式建議收斂
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 統一 JSON（帶 charset）
def json_utf8(payload: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

def _now() -> float:
    return time.time()

def _mk_id() -> str:
    import uuid
    return str(uuid.uuid4())

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

# ==============================
# Mojibake 偵測/修復（UTF-8 被當 latin-1 解析）
# ==============================
def _looks_mojibake(s: str) -> bool:
    if not s:
        return False
    # 常見跡象：UTF-8 以 Latin-1 解碼後的片段
    return ("Ã" in s) or ("Â" in s) or ("å" in s and "æ" in s) or ("�" in s)

def _fix_mojibake_if_needed(s: str) -> (str, bool):
    if not s:
        return s, False
    if _looks_mojibake(s):
        try:
            return s.encode("latin-1").decode("utf-8"), True
        except Exception:
            return s, False
    return s, False

# ==============================
# DB 初始化（移除客製 text_factory，SQLite 預設 UTF-8 即可）
# ==============================
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

# ==============================
# 資料層
# ==============================
def _write_memory(content: str, tags: List[str], ts: Optional[float] = None) -> (str, bool):
    mid = _mk_id()
    # 正規化 + 自癒（如遇 mojibake）
    content = _norm(content)
    content, patched = _fix_mojibake_if_needed(content)
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, content, json.dumps(tags, ensure_ascii=False), ts if ts else _now())
    )
    con.commit()
    return mid, patched

def _search_like(q: str, top_k: int):
    like = f"%{_norm(q)}%"
    rows = cur.execute(
        "SELECT id, content, tags, ts FROM memory "
        "WHERE content LIKE ? OR tags LIKE ? ORDER BY ts DESC LIMIT ?",
        (like, like, max(1, top_k))
    ).fetchall()
    out = []
    for r in rows:
        content = r["content"] or ""
        # 讀出時也自癒一次（若歷史資料曾壞掉）
        fixed, patched = _fix_mojibake_if_needed(content)
        if patched:
            try:
                cur.execute("UPDATE memory SET content=? WHERE id=?", (fixed, r["id"]))
                con.commit()
            except Exception:
                pass
            content = fixed
        out.append({
            "id": r["id"],
            "content": content,
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return out

def _search_memory(q: str, top_k: int):
    # 先用 LIKE（穩定支援中文）；日後若要 FTS 再擴充
    return _search_like(q, top_k)

# ==============================
# 請求模型
# ==============================
class MemoryWriteReq(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

class ComposeReq(BaseModel):
    input: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)

class BundleItem(BaseModel):
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    ts: Optional[float] = None

class BundleIn(BaseModel):
    bundle_version: str = Field(default="1.0")
    persona: str = Field(default=DEFAULT_PERSONA)
    memory: List[BundleItem] = Field(default_factory=list)

# ==============================
# 權杖驗證
# ==============================
def _guard(token: Optional[str]):
    if AUTH_TOKEN:
        if not token or token != AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

# ==============================
# 路由
# ==============================
@app.get("/", summary="Root")
def root():
    return json_utf8({
        "ok": True,
        "service": APP_TITLE,
        "version": APP_VERSION,
        "search_mode": SEARCH_MODE,
        "paths": [
            "/health",
            "/routes",
            "/memory/write",
            "/memory/search",
            "/compose",
            "/bundle/import",
            "/bundle/export",
            "/bundle/preview",
            "/debug/echo",
            "/debug/peek",
            "/debug/reset",
            "/debug/repair_mojibake",
        ],
        "ts": _now(),
    })

@app.get("/routes", summary="List routes")
def routes():
    return json_utf8({"ok": True, "routes": [r.path for r in app.router.routes], "ts": _now()})

@app.get("/health", summary="Healthcheck")
def health():
    return json_utf8({"ok": True, "ts": _now()})

# ---- Debug：確認後端接到的 JSON（驗證客戶端是否送到正確 UTF-8）----
@app.post("/debug/echo")
def debug_echo(body: Dict[str, Any] = Body(...)):
    # 也嘗試對 body 中的可疑字串做預檢（不修改，只顯示）
    preview = {}
    for k, v in body.items():
        if isinstance(v, str):
            fixed, patched = _fix_mojibake_if_needed(v)
            preview[k] = {"value": v, "looks_mojibake": _looks_mojibake(v), "fixed_suggestion": fixed if patched else None}
        else:
            preview[k] = {"value": v}
    return json_utf8({"ok": True, "body": body, "preview": preview, "ts": _now()})

# ---- Debug：最新 50 筆（讀出時自癒並回寫）----
@app.get("/debug/peek")
def debug_peek(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    out = []
    for r in rows:
        content = r["content"] or ""
        fixed, patched = _fix_mojibake_if_needed(content)
        if patched and fixed != content:
            try:
                cur.execute("UPDATE memory SET content=? WHERE id=?", (fixed, r["id"]))
                con.commit()
            except Exception:
                pass
            content = fixed
        out.append({
            "id": r["id"],
            "content": content,
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return json_utf8({"ok": True, "rows": out, "ts": _now()})

# ---- Debug：清庫 ----
@app.post("/debug/reset")
def debug_reset(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    cur.execute("DELETE FROM memory")
    con.commit()
    return json_utf8({"ok": True, "reset": True, "ts": _now()})

# ---- Debug：批次修復庫內 mojibake ----
@app.post("/debug/repair_mojibake")
def repair_all(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id, content FROM memory").fetchall()
    repaired = 0
    for r in rows:
        content = r["content"] or ""
        fixed, patched = _fix_mojibake_if_needed(content)
        if patched and fixed != content:
            cur.execute("UPDATE memory SET content=? WHERE id=?", (fixed, r["id"]))
            repaired += 1
    con.commit()
    return json_utf8({"ok": True, "repaired": repaired, "ts": _now()})

# ---- 記憶：寫入 ----
@app.post("/memory/write")
def memory_write(
    req: MemoryWriteReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    mid, patched = _write_memory(req.content, req.tags)
    return json_utf8({"ok": True, "id": mid, "fixed_from_mojibake": patched})

# ---- 記憶：搜尋（LIKE）----
@app.get("/memory/search")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    hits = _search_memory(q, top_k)
    return json_utf8({"ok": True, "results": hits, "ts": _now()})

# ---- Compose：組合人格+記憶（本地拼接；可接雲端模型）----
@app.post("/compose")
def compose(
    req: ComposeReq,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    q = req.input  # Pydantic 已是正確 Unicode；不要再 encode/decode
    # 讀出命中（自癒已在 _search_memory 進行）
    hits = _search_memory(q, req.top_k)

    system_prompt = (
        "您是『OathLink 穩定語風人格助手（無蘊）』。"
        "規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；"
        "不說空話；必要時先標註風險與前置條件。"
    )
    user_prompt = "【輸入】\n" + q + "\n\n【可用記憶】\n"
    if hits:
        for h in hits:
            user_prompt += f"- {h['content']}\n"
    else:
        user_prompt += "（無匹配記憶）\n"
    user_prompt += "\n請以固定語風輸出最終回覆。"

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆：\n"
        f"1) 已整合輸入：{q}\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )

    return json_utf8({
        "ok": True,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": SEARCH_MODE,
        "ts": _now(),
    })

# ---- Bundle：匯入 ----
@app.post("/bundle/import")
def bundle_import(
    bundle: BundleIn,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    imported = 0
    skipped = 0
    for item in bundle.memory:
        try:
            _id, _ = _write_memory(item.content, item.tags, ts=item.ts)
            imported += 1
        except Exception:
            skipped += 1
    return json_utf8({"ok": True, "imported": imported, "skipped": skipped, "ts": _now()})

# ---- Bundle：匯出（全量）----
@app.get("/bundle/export")
def bundle_export(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC").fetchall()
    mem = []
    for r in rows:
        content = r["content"] or ""
        fixed, _ = _fix_mojibake_if_needed(content)
        mem.append({
            "id": r["id"],
            "content": fixed,
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return json_utf8({
        "ok": True,
        "bundle_version": "1.0",
        "persona": DEFAULT_PERSONA,
        "memory": mem,
        "count": len(mem),
        "ts": _now()
    })

# ---- Bundle：預覽（少量採樣）----
@app.get("/bundle/preview")
def bundle_preview(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 10").fetchall()
    sample = []
    latest_ts = None
    for r in rows:
        content = r["content"] or ""
        fixed, _ = _fix_mojibake_if_needed(content)
        latest_ts = r["ts"] if latest_ts is None else max(latest_ts, r["ts"])
        sample.append({
            "id": r["id"],
            "content": fixed,
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return json_utf8({
        "ok": True,
        "persona": DEFAULT_PERSONA,
        "count_memory": len(sample),
        "latest_ts": latest_ts,
        "sample": sample
    })  