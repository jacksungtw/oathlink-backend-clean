# ============================================================
#  OathLink 記憶伺服器（app_final_stable.py）
#  — 語誓體人格與長期記憶中樞 —
# ============================================================
# 功能摘要：
# 1. UTF-8 安全寫入 / 查詢 / 刪除 / 匯出 / 匯入記憶
# 2. AUTH_TOKEN 可開關（環境變數 AUTH_TOKEN_DISABLE=True 以停用）
# 3. 自動建表與修復 SQLite 結構
# 4. 預設人格「無蘊-敬語版」與詞彙表
# ============================================================

import json, sqlite3, time, os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

DB_PATH = "oathlink_memory.db"
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "abc123")
DISABLE_AUTH = os.getenv("AUTH_TOKEN_DISABLE", "False").lower() == "true"

# ============================================================
#  初始化資料庫（自動建表）
# ============================================================
CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
CONN.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    content TEXT,
    tags TEXT,
    ts REAL
)
""")
CONN.execute("""
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
)
""")
CONN.commit()

# ============================================================
#  輔助函式
# ============================================================

def now_ts() -> float:
    return time.time()

def check_auth(req: Request):
    """若未設定 AUTH_TOKEN_DISABLE，則驗證 X-Auth-Token"""
    if DISABLE_AUTH:
        return True
    header = req.headers.get("X-Auth-Token", "")
    return header == AUTH_TOKEN

def json_response(data: dict):
    """確保 UTF-8 不亂碼"""
    return json.loads(json.dumps(data, ensure_ascii=False))

def meta_get(key: str, default=None):
    cur = CONN.execute("SELECT v FROM meta WHERE k=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default

def meta_set(key: str, value: str):
    CONN.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, value))
    CONN.commit()

# ============================================================
#  FastAPI 主體
# ============================================================

app = FastAPI(title="OathLink Memory Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  資料模型
# ============================================================

class MemoryWrite(BaseModel):
    content: str
    tags: Optional[List[str]] = []

class MemoryDelete(BaseModel):
    id: str

class ComposeBody(BaseModel):
    input: str
    tags: Optional[List[str]] = []
    top_k: Optional[int] = 5

# ============================================================
#  基礎路由
# ============================================================

@app.get("/health")
def health():
    return json_response({"ok": True, "service": "oathlink-backend", "ts": now_ts()})

@app.get("/routes")
def routes():
    return json_response({
        "ok": True,
        "routes": [r.path for r in app.router.routes],
        "ts": now_ts()
    })

# ============================================================
#  記憶操作
# ============================================================

@app.post("/memory/write")
async def memory_write(req: Request, body: MemoryWrite):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    mid = os.urandom(16).hex()
    CONN.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?, ?, ?, ?)",
        (mid, body.content, json.dumps(body.tags, ensure_ascii=False), now_ts())
    )
    CONN.commit()
    return json_response({"ok": True, "id": mid})

@app.get("/memory/search")
async def memory_search(req: Request, q: Optional[str] = "", top_k: int = 5):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    cur = CONN.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{q}%", top_k)
    )
    rows = [
        {"id": r[0], "content": r[1], "tags": json.loads(r[2] or "[]"), "ts": r[3]}
        for r in cur.fetchall()
    ]
    return json_response({"ok": True, "results": rows, "ts": now_ts()})

@app.post("/memory/delete")
async def memory_delete(req: Request, body: MemoryDelete):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    cur = CONN.execute("DELETE FROM memory WHERE id=?", (body.id,))
    CONN.commit()
    return json_response({"ok": True, "deleted": cur.rowcount})

# ============================================================
#  組合 /compose（拼接 prompt）
# ============================================================

@app.post("/compose")
async def compose(req: Request, body: ComposeBody):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    cur = CONN.execute("SELECT id, content FROM memory ORDER BY ts DESC LIMIT ?", (body.top_k,))
    mems = [r[1] for r in cur.fetchall()]
    persona = meta_get("persona", "稱呼願主/師父/您；簡明條列；不得妄稱你。")
    sys_prompt = f"{persona}"
    user_prompt = f"【輸入】\n{body.input}\n\n【可用記憶】\n" + \
        ("\n".join(f"- {m}" for m in mems) if mems else "（無匹配記憶）") + \
        "\n\n請以固定語風輸出最終回覆。"

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )
    return json_response({
        "ok": True,
        "prompt": {"system": sys_prompt, "user": user_prompt},
        "context_hits": [{"id": m[0], "content": m[1]} for m in CONN.execute("SELECT id, content FROM memory")],
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": "like",
        "ts": now_ts(),
    })

# ============================================================
#  Debug / Bundle 操作
# ============================================================

@app.get("/debug/peek")
def debug_peek(req: Request):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    cur = CONN.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC")
    rows = [
        {"id": r[0], "content": r[1], "tags": json.loads(r[2] or "[]"), "ts": r[3]}
        for r in cur.fetchall()
    ]
    return json_response({"ok": True, "rows": rows, "ts": now_ts()})

@app.post("/debug/reset")
def debug_reset(req: Request):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    CONN.execute("DELETE FROM memory")
    CONN.commit()
    meta_set("persona", "稱呼願主/師父/您；簡明條列；不得妄稱你。")
    return json_response({"ok": True, "reset": True, "ts": now_ts()})

@app.get("/bundle/export")
def bundle_export(req: Request):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    persona = meta_get("persona", "稱呼願主/師父/您；簡明條列；不得妄稱你。")
    cur = CONN.execute("SELECT id, content, tags FROM memory")
    memory = [
        {"id": r[0], "content": r[1], "tags": json.loads(r[2] or "[]")}
        for r in cur.fetchall()
    ]
    glossary = [
        {"term": "你", "preferred": "您"},
        {"term": "妳", "preferred": "您"}
    ]
    return json_response({
        "bundle_version": "1.0",
        "persona": {"id": "wuyun-keigo", "name": "無蘊-敬語版", "system_style": persona},
        "memory": memory,
        "glossary": glossary,
    })

@app.post("/bundle/import")
async def bundle_import(req: Request):
    if not check_auth(req):
        return json_response({"ok": False, "error": "unauthorized"})
    data = await req.json()
    mode = data.get("mode", "replace")
    persona = data.get("persona", {})
    memory = data.get("memory", [])
    if mode == "replace":
        CONN.execute("DELETE FROM memory")
    for m in memory:
        CONN.execute(
            "INSERT OR REPLACE INTO memory (id, content, tags, ts) VALUES (?, ?, ?, ?)",
            (m.get("id", os.urandom(8).hex()), m.get("content", ""), json.dumps(m.get("tags", []), ensure_ascii=False), now_ts())
        )
    if persona:
        meta_set("persona", persona.get("system_style", ""))
    CONN.commit()
    return json_response({"ok": True, "mode": mode, "ts": now_ts()})

# ============================================================
#  啟動說明
# ============================================================
# 本檔案可直接運行：
#   uvicorn app_final_stable:app --host 0.0.0.0 --port 8000 --reload
# 若要關閉驗證：
#   set AUTH_TOKEN_DISABLE=True
# ============================================================
