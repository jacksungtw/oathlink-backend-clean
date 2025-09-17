# main.py — wuyun-agent (UTF-8 explicit)
from __future__ import annotations

import json, os, re, time, traceback
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = DATA_DIR / "bundle.json"

DEFAULT_BUNDLE = {
    "bundle_version": "1.0",
    "persona": {
        "id": "wuyun-keigo",
        "name": "無蘊-敬語版",
        "system_style": "稱呼願主/師父/您；簡明條列；先列風險與前置條件；不得妄稱你；不以第2人稱非敬語稱呼。回覆格式可執行、可複製。"
    },
    "memory": [],
    "glossary": [
        {"term": "你", "preferred": "您", "notes": "全程敬語，不得稱『你』"},
        {"term": "妳", "preferred": "您", "notes": "全程敬語，不得稱『妳』"}
    ]
}

def load_bundle() -> Dict:
    if not BUNDLE_PATH.exists():
        save_bundle(DEFAULT_BUNDLE)
    try:
        return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_BUNDLE

def save_bundle(b: Dict) -> None:
    BUNDLE_PATH.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")

BUNDLE = load_bundle()

TOKEN_PAT = re.compile(r"[\u4e00-\u9fff\w]+")

def hit_memories(text: str, k: int = 6) -> List[Dict]:
    tokens = set(TOKEN_PAT.findall(text or ""))
    hits: List[Dict] = []
    for m in BUNDLE.get("memory", []):
        content = m.get("content", "")
        if not content: continue
        if tokens & set(TOKEN_PAT.findall(content)):
            hits.append(m)
            if len(hits) >= k: break
    return hits

CODE_FENCE = re.compile(r"```.*?```", re.S)
URL_OR_MAIL = re.compile(r"(https?://\S+|\b[\w\.-]+@[\w\.-]+\.\w+\b)")
YOU = re.compile(r"(?<![A-Za-z0-9_])([你妳])(?![A-Za-z0-9_])")

def keigo_postprocess(text: str) -> str:
    blocks: List[str] = []
    def stash(m):
        blocks.append(m.group(0)); return f"[[BLOCK:{len(blocks)-1}]]"
    text = CODE_FENCE.sub(stash, text)
    text = URL_OR_MAIL.sub(stash, text)
    text = YOU.sub("您", text)
    def restore(m): return blocks[int(m.group(1))]
    text = re.sub(r"\[\[BLOCK:(\d+)\]\]", restore, text)
    return text

def call_model(adapter: str, model_id: str, system: str, user_text: str, params: Dict) -> str:
    if os.getenv("OFFLINE","1") == "1":
        return (
            "（本地模板降級）\n"
            "前置條件/風險/邊界：\n"
            "- 無網路或未設定金鑰→使用模板\n"
            "- 僅示範敬語替換與結構\n\n"
            f"依願主所輸入：{user_text}\n"
            "回覆：這裡應由真實模型回覆。"
        )
    return "未知 adapter 或線上模式未配置，請設定 OFFLINE=0 與相應金鑰。"

app = FastAPI(title="wuyun-agent", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def J(content: Dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=content, status_code=status_code, media_type="application/json; charset=utf-8")

@app.get("/health")
def health():
    return J({"ok": True, "service": "wuyun-agent", "ts": time.time()})

@app.get("/routes")
def routes():
    return J({"ok": True, "routes": [r.path for r in app.routes], "ts": time.time()})

@app.post("/compose")
def compose(
    body: Dict = Body(..., example={
        "text": "請將這段內容改寫為敬語並條列重點。",
        "adapter": "openai",
        "model_id": "gpt-4o",
        "params": {"temperature": 0.2}
    })
):
    try:
        text    = body.get("text", "") or ""
        adapter = (body.get("adapter") or "openai").lower()
        model   = body.get("model_id") or "gpt-4o"
        params  = body.get("params") or {}

        persona = BUNDLE.get("persona", {}).get("system_style", "")
        hits    = hit_memories(text)
        mem_ctx = "\n".join([f"- {m.get('content','')}" for m in hits]) if hits else ""

        glossary_lines = [f"{g.get('term','')} -> {g.get('preferred','')}" for g in BUNDLE.get("glossary", [])]

        system_prompt = f"[Persona]\n{persona}\n\n[Glossary]\n" + "\n".join(glossary_lines)
        if mem_ctx: system_prompt += f"\n\n[Memory]\n{mem_ctx}"

        result_text = call_model(adapter, model, system_prompt, text, params)
        out = keigo_postprocess(result_text)

        return J({
            "ok": True,
            "ts": time.time(),
            "hits": hits,
            "system_used": system_prompt[:8000],
            "text": out
        })
    except Exception as e:
        return J({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/bundle/export")
def bundle_export():
    return J(BUNDLE)

@app.get("/bundle/preview")
def bundle_preview():
    mem = BUNDLE.get("memory", [])
    gloss = BUNDLE.get("glossary", [])
    return J({
        "ok": True,
        "persona": BUNDLE.get("persona", {}),
        "memory_count": len(mem),
        "glossary_count": len(gloss),
        "memory_sample": mem[:5],
        "ts": time.time()
    })

@app.post("/bundle/import")
def bundle_import(
    body: Dict = Body(...)
):
    mode = (body.get("mode") or "merge").lower()
    persona = body.get("persona")
    memory  = body.get("memory")
    glossary= body.get("glossary")

    global BUNDLE
    cur = load_bundle()

    if mode == "replace":
        new_bundle = {
            "bundle_version": cur.get("bundle_version","1.0"),
            "persona": persona if persona is not None else cur.get("persona", {}),
            "memory":  memory  if memory  is not None else [],
            "glossary":glossary if glossary is not None else []
        }
        BUNDLE = new_bundle
        save_bundle(BUNDLE)
        return J({"ok": True, "mode": "replace", "ts": time.time()})

    if persona:
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
        term_index = {g.get("term"): i for i, g in enumerate(cur.get("glossary", [])) if isinstance(g, dict)}
        for g in glossary:
            if not isinstance(g, dict): continue
            term = g.get("term")
            if term and term in term_index:
                cur["glossary"][term_index[term]] = g
            else:
                cur.setdefault("glossary", []).append(g)

    BUNDLE = cur
    save_bundle(BUNDLE)
    return J({"ok": True, "mode": "merge", "ts": time.time()})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "43110"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
