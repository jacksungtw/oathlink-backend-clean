# app.py — OathLink Backend (DB Sync Minimal)
# FastAPI + SQLite for portable persona/memory/glossary sync.
# New endpoints:
#   - GET  /sync/pull?since_ts=0      -> return persona, glossary, memory >= since_ts
#   - POST /sync/push                  -> upsert persona/glossary/memory; return server_ts
#
# Tables:
#   meta(key PRIMARY KEY, value TEXT)  -- persona/glossary stored as JSON strings
#   memory(id TEXT PRIMARY KEY, content TEXT, tags TEXT(JSON), ts REAL)

from __future__ import annotations
import json, time, sqlite3, os
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "oathlink.db"

app = FastAPI(title="OathLink Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS meta(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS memory(
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        tags TEXT,           -- JSON array of strings
        ts REAL NOT NULL
    )""")
    con.commit()
    con.close()
init_db()

def get_meta(key: str) -> Optional[Any]:
    con = db(); cur = con.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    con.close()
    if not row: return None
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]

def set_meta(key: str, value: Any):
    val = json.dumps(value, ensure_ascii=False)
    con = db(); cur = con.cursor()
    cur.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, val))
    con.commit(); con.close()

@app.get("/health")
def health():
    return {"ok": True, "service": "oathlink-backend", "ts": time.time()}

@app.get("/routes")
def routes():
    return {"ok": True, "routes": [r.path for r in app.routes], "ts": time.time()}

@app.get("/sync/pull")
def sync_pull(since_ts: float = Query(0.0, description="Return memory with ts >= since_ts")):
    con = db(); cur = con.cursor()
    cur.execute("SELECT id, content, tags, ts FROM memory WHERE ts >= ? ORDER BY ts ASC LIMIT 2000", (since_ts,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    # parse tags JSON if present
    for r in rows:
        if r.get("tags"):
            try: r["tags"] = json.loads(r["tags"])
            except Exception: r["tags"] = []

    persona  = get_meta("persona")  or {}
    glossary = get_meta("glossary") or []

    return {
        "ok": True,
        "server_ts": time.time(),
        "persona": persona,
        "glossary": glossary,
        "memory": rows
    }

@app.post("/sync/push")
def sync_push(
    body: Dict = Body(..., example={
        "persona": {"id":"wuyun-keigo","name":"無蘊-敬語版","system_style":"..."},
        "glossary":[{"term":"你","preferred":"您"}],
        "memory":[
            {"id":"m1","content":"偏好A","tags":["style"],"ts": 1757038551.0},
            {"id":"m2","content":"術語B","tags":["gloss"],"ts": 1757038552.0}
        ]
    })
):
    persona  = body.get("persona")
    glossary = body.get("glossary")
    memory   = body.get("memory") or []

    if persona is not None:
        set_meta("persona", persona)
    if glossary is not None:
        set_meta("glossary", glossary)

    con = db(); cur = con.cursor()
    for m in memory:
        mid = m.get("id")
        if not mid: 
            continue
        content = m.get("content","")
        tags    = json.dumps(m.get("tags", []), ensure_ascii=False)
        ts      = float(m.get("ts") or time.time())
        cur.execute("""
            INSERT INTO memory(id, content, tags, ts)
            VALUES(?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET content=excluded.content, tags=excluded.tags, ts=excluded.ts
        """, (mid, content, tags, ts))
    con.commit(); con.close()

    return {"ok": True, "server_ts": time.time()}

# Optional: raw memory search (LIKE)
@app.get("/memory/search")
def memory_search(q: str = Query("", description="LIKE search on content"), limit: int = 20):
    con = db(); cur = con.cursor()
    cur.execute("SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?", (f"%{q}%", limit))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    for r in rows:
        if r.get("tags"):
            try: r["tags"] = json.loads(r["tags"])
            except Exception: r["tags"] = []
    return {"ok": True, "rows": rows}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "43111"))  # run on 43111 to avoid clashing with 43110
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
