# main.py — Integrated wuyun-agent (UTF-8 safe, /compose + bundle APIs + reset + optional sync)
from __future__ import annotations

import os, json, re, time, traceback
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_NAME = "wuyun-agent"
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = DATA_DIR / "bundle.json"

DEFAULT_BUNDLE: Dict = {
    "bundle_version": "1.0",
    "persona": {
        "id": "wuyun-keigo",
        "name": "無蘊-敬語版",
        "system_style": "稱呼願主/師父/您；簡明條列；先列風險與前置條件；不得妄稱你；不以第2人稱非敬語稱呼。回覆格式可執行、可複製。"
    },
    "memory": [],
    "glossary": [
        {"term": "你", "preferred": "您", "notes": "全程敬語"},
        {"term": "妳", "preferred": "您", "notes": "全程敬語"}
    ]
}

def load_bundle() -> Dict:
    if not BUNDLE_PATH.exists():
        save_bundle(DEFAULT_BUNDLE)
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_bundle(bundle: Dict) -> None:
    with open(BUNDLE_PATH, "w", encoding="utf-8", newline="") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

def utf8_json(obj, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=obj, status_code=status_code, media_type="application/json; charset=utf-8")

TOKEN_PAT = re.compile(r"[\u4e00-\u9fff\w]+")
CODE_FENCE = re.compile(r"```.*?```", re.S)
URL_OR_MAIL = re.compile(r"(https?://\S+|\b[\w\.-]+@[\w\.-]+\.\w+\b)")
YOU = re.compile(r"(?<![A-Za-z0-9_])([你妳])(?![A-Za-z0-9_])")

def hit_memories(bundle: Dict, text: str, k: int = 6):
    tokens = set(TOKEN_PAT.findall(text or ""))
    hits: List[Dict] = []
    for m in bundle.get("memory", []):
        c = (m or {}).get("content", "")
        if not c: continue
        if tokens & set(TOKEN_PAT.findall(c)):
            hits.append(m)
            if len(hits) >= k: break
    return hits

def keigo_postprocess(text: str) -> str:
    blocks: List[str] = []
    def stash(m):
        blocks.append(m.group(0)); return f"[[BLOCK:{len(blocks)-1}]]"
    text = CODE_FENCE.sub(stash, text)
    text = URL_OR_MAIL.sub(stash, text)
    text = YOU.sub("您", text)
    def restore(m): return blocks[int(m.group(1))]
    return re.sub(r"\[\[BLOCK:(\d+)\]\]", restore, text)

def call_model(adapter: str, model_id: str, system: str, user_text: str, params: Dict) -> str:
    return (
        "（本地模板）\n"
        "前置條件/風險/邊界：\n"
        "- 無網或未設金鑰→模板\n"
        "- 僅示範敬語替換與結構\n\n"
        f"[System]\n{system}\n\n"
        f"[User]\n{user_text}\n\n"
        "回覆：這裡應由真實模型回覆。"
    )

app = FastAPI(title=APP_NAME, version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return utf8_json({"ok": True, "service": APP_NAME, "ts": time.time()})

@app.get("/routes")
def routes(): return utf8_json({"ok": True, "routes": [r.path for r in app.routes], "ts": time.time()})

@app.post("/compose")
def compose(body: Dict = Body(..., example={
    "text": "請將這段內容改寫為敬語並條列重點。",
    "adapter": "openai",
    "model_id": "gpt-4o",
    "params": {"temperature": 0.2}
})):
    try:
        text    = (body.get("text") or "")
        adapter = (body.get("adapter") or "openai").lower()
        model   = (body.get("model_id") or "gpt-4o")
        params  = (body.get("params") or {})

        bundle  = load_bundle()
        persona = (bundle.get("persona") or {}).get("system_style", "")
        hits    = hit_memories(bundle, text)
        mem_ctx = "\n".join(f"- {m.get('content','')}" for m in hits) if hits else ""

        glossary_lines = [f"{g.get('term','')} -> {g.get('preferred','')}" for g in bundle.get("glossary", [])]
        system = f"[Persona]\n{persona}\n\n[Glossary]\n" + "\n".join(glossary_lines)
        if mem_ctx: system += f"\n\n[Memory]\n{mem_ctx}"

        result = call_model(adapter, model, system, text, params)
        out    = keigo_postprocess(result)

        return utf8_json({"ok": True, "ts": time.time(), "hits": hits, "system_used": system[:8000], "text": out})
    except Exception as e:
        return utf8_json({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/bundle/export")
def bundle_export():
    return utf8_json(load_bundle())

@app.get("/bundle/preview")
def bundle_preview():
    b = load_bundle()
    return utf8_json({
        "ok": True,
        "persona": b.get("persona", {}),
        "memory_count": len(b.get("memory", [])),
        "glossary_count": len(b.get("glossary", [])),
        "memory_sample": b.get("memory", [])[:5],
        "ts": time.time()
    })

@app.post("/bundle/import")
def bundle_import(body: Dict = Body(...)):
    mode = (body.get("mode") or "merge").lower()
    persona = body.get("persona")
    memory  = body.get("memory")
    glossary= body.get("glossary")

    cur = load_bundle()

    if mode == "replace":
        newb = {
            "bundle_version": cur.get("bundle_version", "1.0"),
            "persona": persona if persona is not None else cur.get("persona", {}),
            "memory":  memory  if memory  is not None else [],
            "glossary":glossary if glossary is not None else []
        }
        save_bundle(newb)
        return utf8_json({"ok": True, "mode": "replace", "ts": time.time()})

    if persona is not None:
        cur["persona"] = persona

    if isinstance(memory, list):
        seen = {m.get("id") for m in cur.get("memory", []) if isinstance(m, dict)}
        for m in memory:
            if not isinstance(m, dict): continue
            mid = m.get("id")
            if mid and mid in seen:
                cur["memory"] = [m if x.get("id")==mid else x for x in cur["memory"]]
            else:
                cur.setdefault("memory", []).append(m)
                if mid: seen.add(mid)

    if isinstance(glossary, list):
        idx = {g.get("term"): i for i, g in enumerate(cur.get("glossary", [])) if isinstance(g, dict)}
        for g in glossary:
            if not isinstance(g, dict): continue
            term = g.get("term")
            if term in idx:
                cur["glossary"][idx[term]] = g
            else:
                cur.setdefault("glossary", []).append(g)

    save_bundle(cur)
    return utf8_json({"ok": True, "mode": "merge", "ts": time.time()})

@app.post("/bundle/reset")
def bundle_reset(keep_memory: bool = Query(False, description="是否保留現有記憶")):
    cur = load_bundle()
    newb = DEFAULT_BUNDLE.copy()
    if keep_memory:
        newb["memory"] = cur.get("memory", [])
    save_bundle(newb)
    return utf8_json({"ok": True, "reset": True, "kept_memory": keep_memory, "persona": newb["persona"], "ts": time.time()})

@app.post("/echo")
def echo(body: Dict = Body(...)):
    return utf8_json({"echo": body})

def _have_requests():
    try:
        import requests  # noqa
        return True
    except Exception:
        return False

@app.post("/sync/pull_from")
def sync_pull_from(body: Dict = Body(..., example={"url":"http://127.0.0.1:43111","since_ts":0,"mode":"replace"})):
    if not _have_requests():
        return utf8_json({"ok": False, "error": "requests not installed"}, status_code=400)
    import requests
    url = body.get("url"); since_ts = float(body.get("since_ts") or 0); mode = (body.get("mode") or "replace").lower()
    if not url: return utf8_json({"ok": False, "error":"missing url"}, status_code=400)
    r = requests.get(f"{url.rstrip('/')}/sync/pull", params={"since_ts": since_ts}, timeout=30)
    r.raise_for_status()
    data = r.json()
    payload = {"mode": mode, "persona": data.get("persona"), "glossary": data.get("glossary"), "memory": data.get("memory")}
    return bundle_import(payload)  # type: ignore

@app.post("/sync/push_to")
def sync_push_to(body: Dict = Body(..., example={"url":"http://127.0.0.1:43111"})):
    if not _have_requests():
        return utf8_json({"ok": False, "error": "requests not installed"}, status_code=400)
    import requests
    url = body.get("url")
    if not url: return utf8_json({"ok": False, "error":"missing url"}, status_code=400)
    b = load_bundle()
    payload = {"persona": b.get("persona"), "glossary": b.get("glossary", []), "memory": b.get("memory", [])}
    r = requests.post(f"{url.rstrip('/')}/sync/push", json=payload, timeout=30)
    r.raise_for_status()
    return utf8_json({"ok": True, "server_res": r.json(), "ts": time.time()})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "43110"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
