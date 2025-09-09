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
    # ==== Bundle Import / Export / Preview ======================================
from fastapi import File, UploadFile, Request, Header
from typing import Optional, List, Dict, Any
import json, time

# 假設你已有這些工具函式/變數：_guard(x_auth_token), _now(), _json_utf8(),
# 以及寫入/讀取記憶的 DB 游標 cur、連線 con、正規化 _norm 等。
# 若名稱不同，對應改掉即可。

def _insert_memory(mid: str, content: str, tags: List[str], ts: float):
    cur.execute(
        "INSERT INTO memory (id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, content, json.dumps(tags, ensure_ascii=False), ts)
    )
    con.commit()

def _exists_same(content: str, tags: List[str]) -> bool:
    row = cur.execute(
        "SELECT id FROM memory WHERE content=? AND tags=? LIMIT 1",
        (content, json.dumps(tags, ensure_ascii=False))
    ).fetchone()
    return row is not None

@app.post("/bundle/import")
async def bundle_import(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)

    # 1) 取得 bundle 物件
    if file is not None:
        raw = await file.read()
        try:
            bundle = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return _json_utf8({"ok": False, "error": f"invalid_json_file: {e}", "ts": _now()})
    else:
        try:
            bundle = await request.json()
        except Exception as e:
            return _json_utf8({"ok": False, "error": f"invalid_json_body: {e}", "ts": _now()})

    # 2) 驗證基本欄位
    bundle_version = str(bundle.get("bundle_version", "1.0"))
    persona = str(bundle.get("persona", "")) if bundle.get("persona") is not None else ""
    memory_list = bundle.get("memory", [])
    if not isinstance(memory_list, list):
        return _json_utf8({"ok": False, "error": "memory_must_be_list", "ts": _now()})

    # 3) 寫入記憶（略重複去重邏輯）
    imported = 0
    skipped = 0
    for item in memory_list:
        try:
            content = _norm(item.get("content", ""))
            tags = item.get("tags", [])
            ts = float(item.get("ts", _now()))
            if not content:
                skipped += 1
                continue
            if not isinstance(tags, list):
                tags = []
            if _exists_same(content, tags):
                skipped += 1
                continue
            _insert_memory(_mk_id(), content, tags, ts)
            imported += 1
        except Exception:
            skipped += 1

    # 4) （選擇性）保存 persona，可放 settings 表
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            k TEXT PRIMARY KEY,
            v TEXT
        )
        """)
        con.commit()
        cur.execute("INSERT OR REPLACE INTO settings (k, v) VALUES (?, ?)", ("persona", persona))
        con.commit()
    except Exception:
        pass

    return _json_utf8({
        "ok": True,
        "bundle_version": bundle_version,
        "persona_saved": bool(persona),
        "imported": imported,
        "skipped": skipped,
        "ts": _now()
    })

@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC").fetchall()
    mem = [{
        "id": r["id"],
        "content": r["content"],
        "tags": json.loads(r["tags"] or "[]"),
        "ts": r["ts"],
    } for r in rows]
    persona = ""
    try:
        row = cur.execute("SELECT v FROM settings WHERE k='persona'").fetchone()
        if row: persona = row[0]
    except Exception:
        pass
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
    row = cur.execute("SELECT v FROM settings WHERE k='persona'").fetchone()
    persona = row[0] if row else ""
    latest = cur.execute("SELECT MAX(ts) FROM memory").fetchone()
    latest_ts = latest[0] if latest and latest[0] is not None else None
    sample = cur.execute(
        "SELECT content FROM memory ORDER BY ts DESC LIMIT 3"
    ).fetchall()
    return _json_utf8({
        "ok": True,
        "persona": persona,
        "count_memory": cur.execute("SELECT COUNT(*) FROM memory").fetchone()[0],
        "latest_ts": latest_ts,
        "sample": [s[0] for s in sample],
        "ts": _now()
    })
# ============================================================================