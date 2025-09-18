# main_bridge.py
# OathLink 橋接器：轉送 /compose 與記憶 API 到雲端 BASE，避免本地亂碼問題。
# 本地僅保存 persona / glossary。

import os
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
PORT = int(os.environ.get("PORT", "43112"))

app = FastAPI(title="OathLink Bridge")

# ---- Local persona/glossary storage ----
BUNDLE_PATH = "bundle_store.json"
if not os.path.exists(BUNDLE_PATH):
    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        json.dump({"persona": {}, "memory": [], "glossary": []}, f, ensure_ascii=False, indent=2)

def load_bundle():
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_bundle(data):
    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---- Health ----
@app.get("/health")
async def health():
    return {"ok": True, "bridge": True, "upstream": BASE}

# ---- Bundle API ----
@app.get("/bundle/export")
async def bundle_export():
    return load_bundle()

@app.get("/bundle/preview")
async def bundle_preview():
    data = load_bundle()
    return {
        "ok": True,
        "persona": data.get("persona", {}),
        "memory_count": len(data.get("memory", [])),
        "glossary_count": len(data.get("glossary", [])),
        "memory_sample": data.get("memory", [])[:3]
    }

@app.post("/bundle/import")
async def bundle_import(req: Request):
    body = await req.json()
    mode = body.get("mode", "merge")
    new_data = {k: body.get(k, []) for k in ["persona", "memory", "glossary"]}
    data = load_bundle()
    if mode == "replace":
        data = new_data
    else:  # merge
        if "persona" in new_data and new_data["persona"]:
            data["persona"].update(new_data["persona"])
        data["memory"].extend(new_data.get("memory", []))
        data["glossary"].extend(new_data.get("glossary", []))
    save_bundle(data)
    return {"ok": True, "mode": mode}

# ---- Proxy helper ----
async def proxy(method: str, path: str, body=None, params=None):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if AUTH_TOKEN:
        headers["X-Auth-Token"] = AUTH_TOKEN
    async with httpx.AsyncClient() as client:
        r = await client.request(method, url, headers=headers, json=body, params=params, timeout=30)
        return JSONResponse(status_code=r.status_code, content=r.json())

# ---- Proxy endpoints ----
@app.post("/compose")
async def compose(req: Request):
    body = await req.json()
    return await proxy("POST", "/compose", body=body)

@app.post("/memory/write")
async def memory_write(req: Request):
    body = await req.json()
    return await proxy("POST", "/memory/write", body=body)

@app.get("/memory/search")
async def memory_search(q: str, top_k: int = 5):
    return await proxy("GET", "/memory/search", params={"q": q, "top_k": top_k})

# ---- Main ----
if __name__ == "__main__":
    uvicorn.run("main_bridge:app", host="127.0.0.1", port=PORT)
