# ==== UTF-8 helpers & guard (ADD TO main.py, once) ====
import os, json, sqlite3, time, uuid
from typing import Optional, List
from fastapi import UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse

AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "abc123")
DB_PATH    = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "oathlink.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 如果 main.py 已經有全域連線，沿用既有的；否則建立一個輕量連線
try:
    _con  # type: ignore
except NameError:
    _con = sqlite3.connect(DB_PATH, check_same_thread=False)
    _con.row_factory = sqlite3.Row

def _now() -> float:
    return time.time()

def _guard(x_auth_token: Optional[str]):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _json_utf8(payload: dict) -> JSONResponse:
    # 確保回傳頭帶 charset，避免 PowerShell 或代理端誤判編碼
    return JSONResponse(payload, media_type="application/json; charset=utf-8")

# 確保表存在（與您後端 app.py 一致：content/tags/ts 皆為 UTF-8）
_con.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    ts REAL NOT NULL
)
""")
_con.execute("""
CREATE TABLE IF NOT EXISTS persona (
    id INTEGER PRIMARY KEY CHECK (id=1),
    name TEXT,
    system_style TEXT,
    updated_ts REAL
)
""")
_con.commit()

# ==== Persona endpoints (NEW) ====
@app.get("/persona/get")
def persona_get(x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    row = _con.execute("SELECT name, system_style, updated_ts FROM persona WHERE id=1").fetchone()
    if not row:
        return _json_utf8({"ok": True, "persona": None, "ts": _now()})
    return _json_utf8({"ok": True, "persona": dict(row), "ts": _now()})

@app.post("/persona/put")
def persona_put(data: dict, x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    name = data.get("name")
    system_style = data.get("system_style")
    if name is None and system_style is None:
        raise HTTPException(status_code=400, detail="name or system_style required")
    _con.execute(
        "INSERT INTO persona(id,name,system_style,updated_ts) VALUES(1,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET name=COALESCE(excluded.name, persona.name), "
        "system_style=COALESCE(excluded.system_style, persona.system_style), updated_ts=?",
        (name, system_style, _now(), _now()),
    )
    _con.commit()
    return _json_utf8({"ok": True, "ts": _now()})

# ==== Bundle export/preview/import (NEW) ====
@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = [dict(r) for r in _con.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC")]
    persona_row = _con.execute("SELECT name, system_style FROM persona WHERE id=1").fetchone()
    bundle = {
        "ok": True,
        "bundle_version": "1.0",
        "persona": persona_row["name"] if persona_row else None,
        "memory": rows,
        "count": len(rows),
        "ts": _now(),
    }
    return _json_utf8(bundle)

@app.get("/bundle/preview")
def bundle_preview(x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    persona_row = _con.execute("SELECT name FROM persona WHERE id=1").fetchone()
    cnt_row = _con.execute("SELECT COUNT(*) AS c FROM memory").fetchone()
    sample = [dict(r) for r in _con.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 5")]
    latest = _con.execute("SELECT MAX(ts) AS m FROM memory").fetchone()
    return _json_utf8({
        "ok": True,
        "persona": persona_row["name"] if persona_row else None,
        "count_memory": cnt_row["c"],
        "latest_ts": latest["m"],
        "sample": sample
    })

# 1) multipart 版本：PowerShell/.NET HttpClient 上傳檔案用
@app.post("/bundle/import")
async def bundle_import_file(
    file: UploadFile = File(...),
    x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    raw = await file.read()
    # 防 BOM：utf-8-sig 會自動剝除 BOM；無 BOM 也可正常解
    text = raw.decode("utf-8-sig")
    data = json.loads(text)
    return _bundle_import_core(data)

# 2) application/json 版本：直接 POST JSON 用
@app.post("/bundle/import-json")
def bundle_import_json(
    data: dict,
    x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    return _bundle_import_core(data)

def _bundle_import_core(data: dict) -> JSONResponse:
    # 接受 {"bundle_version":"1.0","persona": "...", "memory":[...]}
    persona = data.get("persona")
    mem_list = data.get("memory", [])
    imported, skipped = 0, 0

    # persona 可選
    if persona:
        _con.execute(
            "INSERT INTO persona(id,name,system_style,updated_ts) VALUES(1,?,NULL,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, updated_ts=excluded.updated_ts",
            (persona, _now()),
        )

    for m in mem_list:
        mid = m.get("id") or str(uuid.uuid4())
        content = m.get("content", "")
        tags = m.get("tags", [])
        ts = float(m.get("ts", _now()))
        try:
            _con.execute(
                "INSERT OR IGNORE INTO memory(id, content, tags, ts) VALUES(?,?,?,?)",
                (mid, content, json.dumps(tags, ensure_ascii=False), ts),
            )
            if _con.total_changes:
                imported += 1
            else:
                skipped += 1
        except sqlite3.IntegrityError:
            skipped += 1

    _con.commit()
    return _json_utf8({
        "ok": True,
        "bundle_version": "1.0",
        "persona_saved": bool(persona),
        "imported": imported,
        "skipped": skipped,
        "ts": _now(),
    })
# ==== end new endpoints ====