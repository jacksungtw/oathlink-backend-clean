# main.py — wuyun-agent (Local Prompt Composer)
# FastAPI app that composes Persona + Memory into prompts and (optionally) calls a model.
# - /compose : build prompt using bundle.json (persona/glossary/memory) + keigo postprocess
# - /bundle/import : merge/replace bundle items
# - /bundle/export : dump bundle.json
# - /bundle/preview: show current persona/glossary sizes & memory sample

from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- Paths & Bundle IO ----------

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
        # fallback to default
        return DEFAULT_BUNDLE

def save_bundle(b: Dict) -> None:
    BUNDLE_PATH.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")

BUNDLE = load_bundle()

# ---------- Simple Memory Hit (LIKE/FTS-lite) ----------

TOKEN_PAT = re.compile(r"[\u4e00-\u9fff\w]+")

def hit_memories(text: str, k: int = 6) -> List[Dict]:
    """Trivial token overlap against memory.content"""
    tokens = set(TOKEN_PAT.findall(text or ""))
    hits: List[Dict] = []
    for m in BUNDLE.get("memory", []):
        content = m.get("content", "")
        if not content:
            continue
        if tokens & set(TOKEN_PAT.findall(content)):
            hits.append(m)
            if len(hits) >= k:
                break
    return hits

# ---------- Keigo Post-processor (safe replace) ----------

CODE_FENCE = re.compile(r"```.*?```", re.S)
URL_OR_MAIL = re.compile(r"(https?://\S+|\b[\w\.-]+@[\w\.-]+\.\w+\b)")
YOU = re.compile(r"(?<![A-Za-z0-9_])([你妳])(?![A-Za-z0-9_])")

def keigo_postprocess(text: str) -> str:
    blocks: List[str] = []

    def stash(m):
        blocks.append(m.group(0))
        return f"[[BLOCK:{len(blocks)-1}]]"

    # stash code blocks and urls/emails
    text = CODE_FENCE.sub(stash, text)
    text = URL_OR_MAIL.sub(stash, text)

    # replace in remaining
    text = YOU.sub("您", text)

    # restore
    def restore(m):
        return blocks[int(m.group(1))]
    text = re.sub(r"\[\[BLOCK:(\d+)\]\]", restore, text)
    return text

# ---------- Model Adapter (stub) ----------

def call_model(adapter: str, model_id: str, system: str, user_text: str, params: Dict) -> str:
    """
    Replace this stub with real OpenAI/Ollama clients.
    For now, if OFFLINE=1 or no keys, returns a local template response.
    """
    offline = os.getenv("OFFLINE", "1") == "1"
    if offline:
        return (
            "（本地模板降級）\n"
            "前置條件/風險/邊界：\n"
            "- 無網路或未設定金鑰→使用模板\n"
            "- 僅示範敬語替換與結構\n\n"
            f"依願主所輸入：{user_text}\n"
            "回覆：這裡應由真實模型回覆。"
        )
    # Example (pseudocode):
    # if adapter == "openai":
    #     from openai import OpenAI
    #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #     rsp = client.chat.completions.create(
    #         model=model_id,
    #         messages=[
    #             {"role": "system", "content": system},
    #             {"role": "user", "content": user_text}
    #         ],
    #         **params
    #     )
    #     return rsp.choices[0].message.content
    # elif adapter == "ollama":
    #     import requests, json as _json
    #     r = requests.post("http://127.0.0.1:11434/api/chat",
    #         json={
    #             "model": model_id,
    #             "messages": [
    #                 {"role":"system", "content": system},
    #                 {"role":"user", "content": user_text}
    #             ]
    #         }, timeout=60)
    #     return r.json().get("message",{}).get("content","")
    # else:
    #     return "未知 adapter，請檢查設定。"
    return "未知 adapter 或線上模式未配置，請設定 OFFLINE=0 與相應金鑰。"

# ---------- FastAPI App ----------

app = FastAPI(title="wuyun-agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "wuyun-agent", "ts": time.time()}

@app.get("/routes")
def routes():
    return {
        "ok": True,
        "routes": [r.path for r in app.routes],
        "ts": time.time()
    }

# ---------- /compose ----------

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

        glossary_lines = [
            f"{g.get('term','')} -> {g.get('preferred','')}"
            for g in BUNDLE.get("glossary", [])
        ]

        system_prompt = (
            f"[Persona]\n{persona}\n\n"
            f"[Glossary]\n" + "\n".join(glossary_lines)
        )
        if mem_ctx:
            system_prompt += f"\n\n[Memory]\n{mem_ctx}"

        result_text = call_model(adapter, model, system_prompt, text, params)
        out = keigo_postprocess(result_text)

        return JSONResponse({
            "ok": True,
            "ts": time.time(),
            "hits": hits,
            "system_used": system_prompt[:8000],
            "text": out
        })
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }, status_code=500)

# ---------- Bundle APIs ----------

@app.get("/bundle/export")
def bundle_export():
    return JSONResponse(BUNDLE)

@app.get("/bundle/preview")
def bundle_preview():
    mem = BUNDLE.get("memory", [])
    gloss = BUNDLE.get("glossary", [])
    return {
        "ok": True,
        "persona": BUNDLE.get("persona", {}),
        "memory_count": len(mem),
        "glossary_count": len(gloss),
        "memory_sample": mem[:5],
        "ts": time.time()
    }

@app.post("/bundle/import")
def bundle_import(
    body: Dict = Body(
        ...,
        example={
            "mode": "merge",  # or "replace"
            "persona": {"id":"wuyun-keigo","name":"無蘊-敬語版","system_style":"..."},
            "memory": [{"id":"m1","content":"關鍵偏好","tags":["style"],"ts": 1757038551.0}],
            "glossary": [{"term":"你","preferred":"您"}]
        }
    )
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
        return {"ok": True, "mode": "replace", "ts": time.time()}

    # merge mode
    if persona:
        cur["persona"] = persona

    if isinstance(memory, list):
        # append unique by id if provided; otherwise append
        seen = {m.get("id") for m in cur.get("memory", []) if isinstance(m, dict)}
        for m in memory:
            mid = None
            if isinstance(m, dict):
                mid = m.get("id")
                if mid and mid in seen:
                    # replace by id
                    cur["memory"] = [m if x.get("id")==mid else x for x in cur["memory"]]
                else:
                    cur.setdefault("memory", []).append(m)
                    if mid: seen.add(mid)

    if isinstance(glossary, list):
        # merge by term
        term_index = {g.get("term"): i for i, g in enumerate(cur.get("glossary", [])) if isinstance(g, dict)}
        for g in glossary:
            term = g.get("term") if isinstance(g, dict) else None
            if term and term in term_index:
                cur["glossary"][term_index[term]] = g
            else:
                cur.setdefault("glossary", []).append(g)

    BUNDLE = cur
    save_bundle(BUNDLE)
    return {"ok": True, "mode": "merge", "ts": time.time()}

# ---------- Dev entry ----------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "43110"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
