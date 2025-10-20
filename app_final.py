# app_final.py  —  UTF-8 safe + no orjson dependency
import os
import json
import time
import sqlite3
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

# -----------------------------------------------------------------------------
# 基本設定
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("OATHLINK_DB", "oathlink.db")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "abc123")  # 若不想驗證可設為空字串 ""

PERSONA_DEFAULT = {
    "id": "wuyun-keigo",
    "name": "無蘊-敬語版",
    "system_style": "稱呼願主/師父/您；簡明條列；先列風險與前置條件；不得妄稱你。"
}
GLOSSARY_DEFAULT = [
    {"term": "你", "preferred": "您", "notes": "全程敬語"},
    {"term": "妳", "preferred": "您", "notes": "全程敬語"},
]

# -----------------------------------------------------------------------------
# App & 中介層
# -----------------------------------------------------------------------------
app = FastAPI(title="oathlink-backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def j(data: Dict[str, Any], status_code: int = 200) -> Response:
    """以 UTF-8 回傳 JSON，中文不轉義。"""
    return Response(
        content=json.dumps(data, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )

# -----------------------------------------------------------------------------
# 資料庫
# -----------------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # 預設即 UTF-8，無需改 text_factory
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

CONN = get_conn()

def meta_set(key: str, value: Any):
    CONN.execute(
        "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, json.dumps(value, ensure_ascii=False))
    )
    CONN.commit()

def meta_get(key: str, default: Any):
    cur = CONN.execute("SELECT v FROM meta WHERE k=?", (key,))
    row = cur.fetchone()
    if not row:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return default

# 初始化 persona / glossary（若尚未設定）
if meta_get("persona", None) is None:
    meta_set("persona", PERSONA_DEFAULT)
if meta_get("glossary", None) is None:
    meta_set("glossary", GLOSSARY_DEFAULT)

# -----------------------------------------------------------------------------
# 驗證
# -----------------------------------------------------------------------------
def check_auth(x_auth_token: Optional[str]) -> Optional[Response]:
    """若設定了 AUTH_TOKEN，則要求 Header 相符；否則略過。"""
    if AUTH_TOKEN and (x_auth_token or "") != AUTH_TOKEN:
        return j({"ok": False, "error": "unauthorized"}, 401)
    return None

# -----------------------------------------------------------------------------
# 工具
# -----------------------------------------------------------------------------
def new_uuid() -> str:
    import uuid
    return str(uuid.uuid4())

def now_ts() -> float:
    return time.time()

# -----------------------------------------------------------------------------
# 路由
# -----------------------------------------------------------------------------
from pydantic import BaseModel

class MemoryDeleteIn(BaseModel):
    id: str

@app.post("/memory/delete")
def memory_delete(inp: MemoryDeleteIn, req: Request):
    if not check_auth(req):
        raise HTTPException(status_code=401, detail="Unauthorized")
    with CONN:
        cur = CONN.execute("DELETE FROM memory WHERE id=?", (inp.id,))
        deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
    return {"ok": True, "deleted": int(deleted)}

@app.get("/health")
def health():
    return j({"ok": True, "service": "oathlink-backend", "ts": now_ts()})

@app.get("/routes")
def routes():
    return j({
        "ok": True,
        "routes": [
            "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc",
            "/health", "/routes",
            "/debug/reset", "/debug/peek",
            "/memory/write", "/memory/search",
            "/compose",
            "/bundle/export", "/bundle/preview", "/bundle/import", "/bundle/reset",
        ],
        "ts": now_ts()
    })

# --------------------- Debug ---------------------
@app.post("/debug/reset")
def debug_reset(x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    CONN.execute("DELETE FROM memory")
    meta_set("persona", PERSONA_DEFAULT)
    meta_set("glossary", GLOSSARY_DEFAULT)
    return j({"ok": True, "reset": True, "ts": now_ts()})

@app.get("/debug/peek")
def debug_peek(x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    cur = CONN.execute("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC")
    rows = []
    for rid, content, tags, ts_ in cur.fetchall():
        try:
            tags_list = json.loads(tags)
        except Exception:
            tags_list = []
        rows.append({"id": rid, "content": content, "tags": tags_list, "ts": ts_})
    return j({"ok": True, "rows": rows, "ts": now_ts()})

# --------------------- Memory ---------------------
@app.post("/memory/write")
async def memory_write(req: Request, x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    payload = await req.json()
    content = str(payload.get("content", "")).strip()
    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    if not content:
        return j({"ok": False, "error": "content is empty"}, 400)

    rid = new_uuid()
    CONN.execute(
        "INSERT INTO memory(id, content, tags, ts) VALUES(?,?,?,?)",
        (rid, content, json.dumps(tags, ensure_ascii=False), now_ts())
    )
    CONN.commit()
    return j({"ok": True, "id": rid})

@app.get("/memory/search")
def memory_search(q: Optional[str] = None, top_k: int = 5,
                  x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    q = (q or "").strip()
    results = []
    if q:
        cur = CONN.execute(
            "SELECT id,content,tags,ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{q}%", top_k)
        )
        for rid, content, tags, ts_ in cur.fetchall():
            try:
                tags_list = json.loads(tags)
            except Exception:
                tags_list = []
            results.append({"id": rid, "content": content, "tags": tags_list, "ts": ts_})
    return j({"ok": True, "results": results, "ts": now_ts()})

# --------------------- Compose ---------------------
@app.post("/compose")
async def compose(req: Request, x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    payload = await req.json()
    input_text = str(payload.get("input", payload.get("text", "")))
    tags = payload.get("tags", [])
    top_k = int(payload.get("top_k", 5) or 5)

    persona = meta_get("persona", PERSONA_DEFAULT)

    # 取記憶 hits（LIKE 模式：用 input 的前 20 字當關鍵）
    needle = input_text[:20]
    hits = []
    if needle.strip():
        cur = CONN.execute(
            "SELECT id,content,tags,ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{needle}%", top_k)
        )
        for rid, content, tags_, ts_ in cur.fetchall():
            try:
                tags_list = json.loads(tags_)
            except Exception:
                tags_list = []
            hits.append({"id": rid, "content": content, "tags": tags_list, "ts": ts_})

    # 組 prompt（顯示用）
    if hits:
        ctx = "\n".join([f"- {h['content']}" for h in hits])
    else:
        ctx = "（無匹配記憶）"

    prompt = {
        "system": persona["system_style"],
        "user": f"【輸入】\n{input_text}\n\n【可用記憶】\n{ctx}\n\n請以固定語風輸出最終回覆。"
    }

    # 這裡示範本地拼接（未呼叫外部模型）
    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )

    return j({
        "ok": True,
        "prompt": prompt,
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": "like",
        "ts": now_ts()
    })

# --------------------- Bundle ---------------------
@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    persona = meta_get("persona", PERSONA_DEFAULT)
    glossary = meta_get("glossary", GLOSSARY_DEFAULT)

    # 匯出時不把所有 memory 帶出（依您需求可更動：這裡保留空）
    return j({
        "bundle_version": "1.0",
        "persona": persona,
        "memory": [],
        "glossary": glossary,
    })

@app.get("/bundle/preview")
def bundle_preview(x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    persona = meta_get("persona", PERSONA_DEFAULT)
    glossary = meta_get("glossary", GLOSSARY_DEFAULT)

    cur = CONN.execute("SELECT id,content,tags,ts FROM memory ORDER BY ts DESC LIMIT 5")
    sample = []
    for rid, content, tags, ts_ in cur.fetchall():
        try:
            tags_list = json.loads(tags)
        except Exception:
            tags_list = []
        sample.append({"id": rid, "content": content, "tags": tags_list, "ts": ts_})
    return j({
        "ok": True,
        "persona": persona,
        "memory_count": _count_memory(),
        "glossary_count": len(glossary),
        "memory_sample": sample,
        "ts": now_ts()
    })

def _count_memory() -> int:
    cur = CONN.execute("SELECT COUNT(1) FROM memory")
    return int(cur.fetchone()[0])

@app.post("/bundle/import")
async def bundle_import(req: Request, x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    payload = await req.json()
    mode = payload.get("mode", "replace")

    persona = payload.get("persona")
    glossary = payload.get("glossary")
    memory = payload.get("memory", [])

    if mode not in ("replace", "merge"):
        return j({"ok": False, "error": "mode must be 'replace' or 'merge'"}, 400)

    if persona:
        if mode == "replace":
            meta_set("persona", persona)
        else:
            old = meta_get("persona", {})
            old.update(persona or {})
            meta_set("persona", old)

    if glossary:
        if mode == "replace":
            meta_set("glossary", glossary)
        else:
            old = meta_get("glossary", [])
            # 簡單合併：直接接在後面（若需去重可再強化）
            meta_set("glossary", old + glossary)

    # 可選：匯入記憶
    if isinstance(memory, list):
        for m in memory:
            content = str(m.get("content", "")).strip()
            tags = m.get("tags", [])
            if not content:
                continue
            rid = new_uuid()
            CONN.execute(
                "INSERT INTO memory(id, content, tags, ts) VALUES(?,?,?,?)",
                (rid, content, json.dumps(tags, ensure_ascii=False), now_ts())
            )
        CONN.commit()

    return j({"ok": True, "mode": mode, "ts": now_ts()})

@app.post("/bundle/reset")
def bundle_reset(x_auth_token: Optional[str] = Header(None)):
    unauthorized = check_auth(x_auth_token)
    if unauthorized: return unauthorized

    meta_set("persona", PERSONA_DEFAULT)
    meta_set("glossary", GLOSSARY_DEFAULT)
    return j({"ok": True, "ts": now_ts()})
