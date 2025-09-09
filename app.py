# app.py
from __future__ import annotations
import json, os, sqlite3, time, uuid
from typing import List, Optional
from fastapi import FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "oathlink.db")
AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "abc123")

app = FastAPI(title="OathLink Backend (UTF-8 fixed)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row           # 以欄位名讀取
    # 不自訂 text_factory，SQLite 預設就是 UTF-8，避免亂碼來源
    return con

def _now() -> float:
    return float(time.time())

def _json(data: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=data,
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )

def _guard(x_auth_token: Optional[str]):
    if x_auth_token != AUTH_TOKEN:
        raise _json({"ok": False, "error": "unauthorized"}, 401)

def _init_db():
    con = _conn()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    con.commit()
    con.close()

_init_db()

@app.get("/health")
def health():
    return _json({"ok": True, "service": "oathlink-backend", "ts": _now()})

@app.get("/routes")
def routes():
    return _json({
        "ok": True,
        "routes": [
            "/health", "/routes",
            "/memory/write", "/memory/search",
            "/debug/peek", "/debug/reset",
            "/compose"
        ],
        "ts": _now()
    })

@app.post("/memory/write")
def memory_write(
    body: dict,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _guard(x_auth_token)
    content = (body or {}).get("content", "")
    tags: List[str] = (body or {}).get("tags", []) or []
    mid = str(uuid.uuid4())

    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, str(content), json.dumps(tags, ensure_ascii=False), _now()),
    )
    con.commit()
    con.close()
    return _json({"ok": True, "id": mid})

@app.get("/debug/peek")
def debug_peek(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _guard(x_auth_token)
    con = _conn()
    rows = con.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "content": r["content"],                       # 直接原文，避免再編碼
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"],
        })
    return _json({"ok": True, "rows": out, "ts": _now()})

@app.post("/debug/reset")
def debug_reset(
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _guard(x_auth_token)
    con = _conn()
    con.execute("DELETE FROM memory")
    con.commit()
    con.close()
    return _json({"ok": True, "reset": True, "ts": _now()})

@app.get("/memory/search")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=100),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _guard(x_auth_token)
    con = _conn()
    # 先用簡單 LIKE（SQLite 預設 UTF-8 沒問題）；之後再換 FTS 也行
    rows = con.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{q}%", top_k),
    ).fetchall()
    con.close()

    hits = []
    for r in rows:
        hits.append({
            "id": r["id"],
            "content": r["content"],
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"],
        })
    return _json({"ok": True, "results": hits, "ts": _now()})

@app.post("/compose")
def compose(
    body: dict,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
):
    _guard(x_auth_token)
    input_text: str = (body or {}).get("input", "") or ""
    top_k: int = int((body or {}).get("top_k", 1) or 1)

    # 取上下文
    con = _conn()
    rows = con.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{input_text}%", top_k),
    ).fetchall()
    con.close()

    hits = [{
        "id": r["id"],
        "content": r["content"],
        "tags": json.loads(r["tags"] or "[]"),
        "ts": r["ts"],
    } for r in rows]

    prompt_user = (
        f"【輸入】\n{input_text}\n\n"
        "【可用記憶】\n" +
        ("\n".join(f"- {h['content']}" for h in hits) if hits else "（無匹配記憶）") +
        "\n\n請以固定語風輸出最終回覆。"
    )

    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆：\n"
        f"1) 已整合輸入：{input_text}\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )

    return _json({
        "ok": True,
        "prompt": {
            "system": "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。",
            "user": prompt_user,
        },
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": "like",
        "ts": _now(),
    })
    # ==== bundle endpoints (add this block) ====
import os, json, time, sqlite3, uuid
from typing import Optional, List
from fastapi import Request, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "abc123")

def _guard(x_auth_token: Optional[str]):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

DB_PATH = os.getenv("DB_PATH", "data/oathlink.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_con = sqlite3.connect(DB_PATH, check_same_thread=False)
_con.row_factory = sqlite3.Row
_cur = _con.cursor()
# memory 表 & meta 表（儲存 persona）
_cur.execute("""
CREATE TABLE IF NOT EXISTS memory(
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  tags TEXT NOT NULL,
  ts REAL NOT NULL
)""")
_cur.execute("""
CREATE TABLE IF NOT EXISTS meta(
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
)""")
_con.commit()

def _now() -> float:
    return time.time()

def _mk_id() -> str:
    return str(uuid.uuid4())

def _json_utf8(data: dict) -> JSONResponse:
    return JSONResponse(content=data, media_type="application/json; charset=utf-8")

class MemItem(BaseModel):
    content: str
    tags: List[str] = []
    ts: float

class Bundle(BaseModel):
    bundle_version: str = "1.0"
    persona: Optional[str] = None
    memory: List[MemItem] = []

@app.get("/routes")
def list_routes():
    from fastapi.routing import APIRoute
    items = []
    for r in app.routes:
        if isinstance(r, APIRoute):
            items.append({"path": r.path, "methods": sorted(r.methods)})
    return _json_utf8({"ok": True, "routes": items, "ts": _now()})

@app.post("/bundle/import")
async def bundle_import(
    request: Request,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
    file: Optional[UploadFile] = File(default=None)
):
    _guard(x_auth_token)

    # 1) 讀取 JSON：multipart 檔案優先，其次直接讀取 body
    try:
        if file is not None:
            raw = await file.read()
            data = json.loads(raw.decode("utf-8"))
        else:
            data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        bundle = Bundle(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid bundle schema: {e}")

    imported = 0
    for m in bundle.memory:
        # 去重：相同 content+tags 視為重覆
        tags_json = json.dumps(m.tags, ensure_ascii=False)
        exists = _cur.execute(
            "SELECT id FROM memory WHERE content=? AND tags=?",
            (m.content, tags_json)
        ).fetchone()
        if exists:
            continue
        _cur.execute(
            "INSERT INTO memory(id, content, tags, ts) VALUES(?,?,?,?)",
            (_mk_id(), m.content, tags_json, float(m.ts))
        )
        imported += 1

    # 保存 persona（若提供）
    persona_saved = False
    if bundle.persona:
        _cur.execute("INSERT INTO meta(k, v) VALUES('persona', ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (bundle.persona,))
        persona_saved = True

    _con.commit()
    return _json_utf8({
        "ok": True,
        "bundle_version": bundle.bundle_version,
        "persona_saved": persona_saved,
        "imported": imported,
        "skipped": len(bundle.memory) - imported,
        "ts": _now()
    })

@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    persona_row = _cur.execute("SELECT v FROM meta WHERE k='persona'").fetchone()
    persona = persona_row["v"] if persona_row else None

    rows = _cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC").fetchall()
    mem = []
    for r in rows:
        mem.append({
            "id": r["id"],
            "content": r["content"],
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    return _json_utf8({
        "ok": True,
        "bundle_version": "1.0",
        "persona": persona,
        "memory": mem,
        "count": len(mem),
        "ts": _now()
    })

@app.get("/bundle/preview")
def bundle_preview(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    persona_row = _cur.execute("SELECT v FROM meta WHERE k='persona'").fetchone()
    persona = persona_row["v"] if persona_row else None

    rows = _cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 5").fetchall()
    sample = []
    for r in rows:
        sample.append({
            "id": r["id"],
            "content": r["content"],
            "tags": json.loads(r["tags"] or "[]"),
            "ts": r["ts"]
        })
    latest_ts = sample[0]["ts"] if sample else None
    return _json_utf8({
        "ok": True,
        "persona": persona,
        "count_memory": _cur.execute("SELECT COUNT(*) AS c FROM memory").fetchone()["c"],
        "latest_ts": latest_ts,
        "sample": sample
    })
# ==== end bundle endpoints ====