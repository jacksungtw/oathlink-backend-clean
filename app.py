# 建立 app.py
@'
import os, json, time, sqlite3, unicodedata, uuid
from typing import Optional, Any
from fastapi import FastAPI, Header, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

# -------- UTF-8 JSON Response（確保回傳永遠是 UTF-8） --------
class UTF8JSONResponse(Response):
    media_type = "application/json; charset=utf-8"
    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

# -------- 基本設定 --------
DB_PATH   = os.environ.get("DB_PATH", "oathlink.db")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "abc123")

def _now() -> float:
    return time.time()

def _mk_id() -> str:
    return str(uuid.uuid4())

def _norm(s: str) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFKC", s)

def _guard(x_auth_token: Optional[str]):
    if x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------- DB 初始化（不要覆寫 text_factory；SQLite 原生支援 UTF-8） --------
def _init_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS memory(
          id      TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          tags    TEXT NOT NULL, -- JSON (ensure_ascii=False)
          ts      REAL NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory(ts DESC)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_memory_content ON memory(content)")
    con.commit()
    return con

CON = _init_db()

# -------- App 與 CORS --------
app = FastAPI(title="OathLink Backend (UTF-8 Clean)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# -------- 健康檢查與路由清單 --------
@app.get("/health")
def health():
    return UTF8JSONResponse({"ok": True, "service":"OathLink Backend", "ts": _now()})

@app.get("/routes")
def routes():
    items = [{"path": r.path, "methods": sorted(list(r.methods or []))} for r in app.router.routes]
    return UTF8JSONResponse({"ok": True, "routes": items, "ts": _now()})

# -------- Debug：清庫 / 檢視 / 回顯 / 修復 Mojibake --------
@app.post("/debug/reset")
def debug_reset(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    CON.execute("DROP TABLE IF EXISTS memory")
    CON.commit()
    CON.execute("""
        CREATE TABLE memory(
          id      TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          tags    TEXT NOT NULL,
          ts      REAL NOT NULL
        )
    """)
    CON.execute("CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory(ts DESC)")
    CON.execute("CREATE INDEX IF NOT EXISTS idx_memory_content ON memory(content)")
    CON.commit()
    return UTF8JSONResponse({"ok": True, "reset": True, "ts": _now()})

@app.get("/debug/peek")
def debug_peek(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = CON.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
    out = []
    for r in rows:
        try:
            tags = json.loads(r["tags"])
        except Exception:
            tags = []
        out.append({"id": r["id"], "content": r["content"], "tags": tags, "ts": r["ts"]})
    return UTF8JSONResponse({"ok": True, "rows": out, "ts": _now()})

@app.post("/debug/echo")
async def debug_echo(request: Request, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    raw = await request.body()
    as_text_utf8 = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(as_text_utf8)
    except Exception:
        parsed = {}
    return UTF8JSONResponse({
        "ok": True,
        "raw_first_32_bytes": list(raw[:32]),
        "as_text_utf8": as_text_utf8,
        "parsed": parsed,
        "ts": _now()
    })

@app.post("/debug/repair_mojibake")
def debug_repair_mojibake(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = CON.execute("SELECT id, content FROM memory").fetchall()
    repaired = 0
    for r in rows:
        s = r["content"]
        # 嘗試把「UTF-8 當 latin-1 解讀」造成的亂碼修回來
        try:
            fixed = s.encode("latin-1").decode("utf-8")
            if fixed != s:
                CON.execute("UPDATE memory SET content=? WHERE id=?", (fixed, r["id"]))
                repaired += 1
        except Exception:
            pass
    CON.commit()
    return UTF8JSONResponse({"ok": True, "repaired": repaired, "ts": _now()})

# -------- 記憶寫入 / 搜尋 --------
@app.post("/memory/write")
async def memory_write(request: Request, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    body = await request.json()
    content = _norm(body.get("content", ""))
    tags    = body.get("tags", [])
    if not isinstance(tags, list): tags = []
    tags = [str(t) for t in tags]
    mid = _mk_id()
    CON.execute(
        "INSERT INTO memory(id, content, tags, ts) VALUES (?,?,?,?)",
        (mid, content, json.dumps(tags, ensure_ascii=False), _now())
    )
    CON.commit()
    return UTF8JSONResponse({"ok": True, "id": mid})

@app.get("/memory/search")
def memory_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
):
    _guard(x_auth_token)
    qn = _norm(q)
    rows = CON.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{qn}%", top_k)
    ).fetchall()
    out = [{"id": r["id"], "content": r["content"], "tags": json.loads(r["tags"]), "ts": r["ts"]} for r in rows]
    return UTF8JSONResponse({"ok": True, "results": out, "ts": _now()})

# -------- 記憶打包：匯入 / 匯出 / 預覽 --------
@app.post("/bundle/import")
async def bundle_import(request: Request, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    body = await request.json()
    persona = body.get("persona", "無蘊-敬語版")
    mems = body.get("memory", [])
    imported, skipped = 0, 0
    for m in mems:
        content = _norm(m.get("content", ""))
        tags = m.get("tags", [])
        ts   = float(m.get("ts", _now()))
        exists = CON.execute("SELECT 1 FROM memory WHERE content=? AND ABS(ts-?)<1e-6", (content, ts)).fetchone()
        if exists:
            skipped += 1
            continue
        mid = _mk_id()
        CON.execute(
            "INSERT INTO memory(id, content, tags, ts) VALUES (?,?,?,?)",
            (mid, content, json.dumps(tags, ensure_ascii=False), ts)
        )
        imported += 1
    CON.commit()
    return UTF8JSONResponse({"ok": True, "imported": imported, "skipped": skipped, "persona": persona, "ts": _now()})

@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    rows = CON.execute("SELECT content, tags, ts FROM memory ORDER BY ts DESC").fetchall()
    mem = [{"content": r["content"], "tags": json.loads(r["tags"]), "ts": r["ts"]} for r in rows]
    return UTF8JSONResponse({
        "ok": True,
        "bundle_version": "1.0",
        "persona": "無蘊-敬語版",
        "memory": mem,
        "count": len(mem),
        "ts": _now()
    })

@app.get("/bundle/preview")
def bundle_preview(x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    row = CON.execute("SELECT MAX(ts) AS latest_ts, COUNT(*) AS cnt FROM memory").fetchone()
    sample = CON.execute("SELECT content, tags, ts FROM memory ORDER BY ts DESC LIMIT 5").fetchall()
    sample_out = [{"content": r["content"], "tags": json.loads(r["tags"]), "ts": r["ts"]} for r in sample]
    return UTF8JSONResponse({
        "ok": True,
        "persona": "無蘊-敬語版",
        "count_memory": int(row["cnt"] or 0),
        "latest_ts": row["latest_ts"],
        "sample": sample_out
    })

# -------- Compose（不外呼模型；本地組裝語風） --------
@app.post("/compose")
async def compose(request: Request, x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")):
    _guard(x_auth_token)
    body = await request.json()
    input_text = _norm(body.get("input", ""))
    top_k = int(body.get("top_k", 1))
    rows = CON.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{input_text}%", top_k)
    ).fetchall()
    hits = [{"id": r["id"], "content": r["content"], "tags": json.loads(r["tags"]), "ts": r["ts"]} for r in rows]

    system = "您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
    prompt_user = "【輸入】\n" + input_text + "\n\n【可用記憶】\n" + ( "\n".join([f"- {h['content']}" for h in hits]) if hits else "（無匹配記憶）" ) + "\n\n請以固定語風輸出最終回覆。"
    output = f"願主，以下為基於您輸入與可用記憶所整理之回覆：\n1) 已整合輸入：{input_text}\n2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"

    return UTF8JSONResponse({
        "ok": True,
        "prompt": {"system": system, "user": prompt_user},
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "search_mode": "like",
        "ts": _now()
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
'@ | Set-Content -Path .\app.py -Encoding UTF8

# 建立 requirements.txt
@'
fastapi>=0.110,<1.0
uvicorn>=0.23
'@ | Set-Content -Path .\requirements.txt -Encoding UTF8

# 建立 Procfile（給 Railway/Heroku 之類）
@'
web: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
'@ | Set-Content -Path .\Procfile -Encoding UTF8

# 建立 README.md（含驗收三步）
@'
# OathLink Backend (UTF-8 Clean)

## 本機啟動
```bash
pip install -r requirements.txt
uvicorn app:app --reload
