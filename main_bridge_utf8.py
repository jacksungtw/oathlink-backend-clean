
# main_bridge_utf8.py — OathLink 橋接器（強制 UTF-8 回傳）
import os, json, time
from pathlib import Path
from typing import Dict, Any

import uvicorn, httpx
from fastapi import FastAPI, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE = os.environ.get("BASE", "https://oathlink-backend-clean-production.up.railway.app").rstrip("/")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "abc123")
PORT = int(os.environ.get("PORT", "43112"))

app = FastAPI(title="OathLink Bridge (UTF-8)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def j(obj, status=200):
    return JSONResponse(content=obj, status_code=status, media_type="application/json; charset=utf-8")

# 本地 persona/glossary（UTF-8 JSON）
DATA_DIR = Path(__file__).parent.resolve() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = DATA_DIR / "bundle_store.json"

DEFAULT_BUNDLE = {"persona": {}, "memory": [], "glossary": []}
if not BUNDLE_PATH.exists():
    BUNDLE_PATH.write_text(json.dumps(DEFAULT_BUNDLE, ensure_ascii=False, indent=2), encoding="utf-8")

def load_bundle() -> Dict[str, Any]:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

def save_bundle(d: Dict[str, Any]):
    BUNDLE_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def hdr():
    h = {"Content-Type": "application/json; charset=utf-8"}
    if AUTH_TOKEN:
        h["X-Auth-Token"] = AUTH_TOKEN
    return h

@app.get("/health")
async def health():
    return j({"ok": True, "bridge": True, "upstream": BASE, "ts": time.time()})

@app.get("/routes")
async def routes():
    return j({"ok": True, "routes": [r.path for r in app.routes], "ts": time.time()})

# 轉送輔助
async def forward(method: str, path: str, *, json_body=None, params=None):
    url = f"{BASE}{path}"
    async with httpx.AsyncClient() as client:
        r = await client.request(method, url, headers=hdr(), json=json_body, params=params, timeout=60)
        data = r.json()
        return j(data, status=r.status_code)

# Compose / Memory（允許 {text} 或 {input}）
@app.post("/compose")
async def compose(req: Request):
    body = await req.json()
    if "input" not in body and "text" in body:
        body = {"input": body["text"], "tags": body.get("tags", []), "top_k": body.get("top_k", 5)}
    return await forward("POST", "/compose", json_body=body)

@app.post("/memory/write")
async def memory_write(req: Request):
    return await forward("POST", "/memory/write", json_body=await req.json())

@app.get("/memory/search")
async def memory_search(q: str, top_k: int = 5):
    return await forward("GET", "/memory/search", params={"q": q, "top_k": top_k})

# Bundle（本地保存 persona/glossary；memory 走上游）
@app.get("/bundle/export")
async def bundle_export():
    b = load_bundle()
    return j({**b, "memory": []})

@app.get("/bundle/preview")
async def bundle_preview():
    b = load_bundle()
    return j({"ok": True, "persona": b.get("persona", {}), "memory_count": None, "glossary_count": len(b.get("glossary", [])), "memory_sample": [], "ts": time.time()})

@app.post("/bundle/import")
async def bundle_import(req: Request):
    body = await req.json()
    mode = (body.get("mode") or "merge").lower()
    persona = body.get("persona")
    glossary = body.get("glossary")
    memory = body.get("memory")

    cur = load_bundle()
    if mode == "replace":
        cur = {"persona": persona or {}, "memory": [], "glossary": glossary or []}
    else:
        if persona is not None:
            cur["persona"] = persona
        if isinstance(glossary, list):
            cur["glossary"] = glossary
    save_bundle(cur)

    if isinstance(memory, list) and memory:
        async with httpx.AsyncClient() as client:
            for m in memory:
                await client.post(f"{BASE}/memory/write", headers=hdr(), json={"content": m.get("content",""), "tags": m.get("tags", [])}, timeout=30)

    return j({"ok": True, "mode": mode, "ts": time.time()})

if __name__ == "__main__":
    uvicorn.run("main_bridge_utf8:app", host="127.0.0.1", port=PORT, reload=False)
'@ | Set-Content -Path .\main_bridge_utf8.py -Encoding utf8