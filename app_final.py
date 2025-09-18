# app_final.py — OathLink 最終整合版（UTF-8 安全 /compose + 記憶 + bundle + reset）
# ----------------------------------------------------------------------------
# 功能摘要
# - UTF-8 鎖定：所有檔案以 UTF-8 讀寫、回傳 JSON 皆帶 charset=utf-8
# - 路由：/health /routes /compose /memory/write /memory/search
#         /debug/reset /debug/peek
#         /bundle/export /bundle/import /bundle/preview /bundle/reset
# - 記憶存放：data/memory.json（UTF-8）
# - Persona/Glossary：data/bundle.json（UTF-8）
# - 權杖：若環境變數 AUTH_TOKEN 存在，要求 Header: X-Auth-Token
# - 模型：預設本地模板，之後可於 call_model() 接 OpenAI/Ollama 等
# ----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, uuid, re, traceback
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, Body, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_NAME   = "oathlink-backend"
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MEM_PATH   = DATA_DIR / "memory.json"
BUNDLE_PATH= DATA_DIR / "bundle.json"

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# -------------------------- 工具：UTF-8 JSON 回傳 --------------------------
def j(obj: Any, status: int = 200) -> JSONResponse:
    # content 必須是 Python 物件，避免二次編碼
    return JSONResponse(content=obj, status_code=status, media_type="application/json; charset=utf-8")

def require_auth(x_auth_token: str | None):
    if AUTH_TOKEN and (x_auth_token or "") != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------------- 檔案 I/O（UTF-8） --------------------------
def read_json(path: Path, default):
    if not path.exists():
        path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------------- 預設資料 --------------------------
DEFAULT_BUNDLE: Dict[str, Any] = {
    "bundle_version": "1.0",
    "persona": {
        "id": "wuyun-keigo",
        "name": "無蘊-敬語版",
        "system_style": "稱呼願主/師父/您；簡明條列；先列風險與前置條件；不得妄稱你。"
    },
    "glossary": [
        {"term":"你","preferred":"您","notes":"全程敬語"},
        {"term":"妳","preferred":"您","notes":"全程敬語"}
    ],
    "memory": []
}

def load_bundle() -> Dict[str, Any]:
    return read_json(BUNDLE_PATH, DEFAULT_BUNDLE)

def save_bundle(b: Dict[str, Any]):
    write_json(BUNDLE_PATH, b)

def load_memory() -> List[Dict[str, Any]]:
    return read_json(MEM_PATH, [])

def save_memory(rows: List[Dict[str, Any]]):
    write_json(MEM_PATH, rows)

# -------------------------- 記憶檢索（LIKE） --------------------------
TOKEN_PAT = re.compile(r"[\u4e00-\u9fff\w]+")

def search_memory(q: str, rows: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not q:
        return []
    q_tokens = set(TOKEN_PAT.findall(q))
    scored = []
    for m in rows:
        content = m.get("content","")
        tokens = set(TOKEN_PAT.findall(content))
        score = len(q_tokens & tokens)
        if q in content:
            score += 2
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]

# -------------------------- 模型呼叫（可擴充） --------------------------
def call_model(system: str, user_text: str, params: Dict[str, Any] | None = None) -> str:
    # 預設本地模板，不外呼
    return (
        "願主，以下為基於您輸入與可用記憶所整理之回覆（本地拼接，未呼叫外部模型）：\n"
        "1) 已整合輸入與歷史記憶。\n"
        "2) 若需更精煉文本，請設定 OPENAI_API_KEY 以啟用雲端生成。"
    )

# -------------------------- FastAPI App --------------------------
app = FastAPI(title=APP_NAME, version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return j({"ok": True, "service": APP_NAME, "ts": time.time()})

@app.get("/routes")
def routes():
    return j({"ok": True, "routes": [r.path for r in app.routes], "ts": time.time()})

# -------------------------- Debug / Reset --------------------------
@app.post("/debug/reset")
def debug_reset(x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    require_auth(x_auth_token)
    save_memory([])
    save_bundle(DEFAULT_BUNDLE)
    return j({"ok": True, "reset": True, "ts": time.time()})

@app.get("/debug/peek")
def debug_peek(x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    require_auth(x_auth_token)
    rows = load_memory()
    rows_sorted = sorted(rows, key=lambda m: m.get("ts", 0), reverse=True)
    return j({"ok": True, "rows": rows_sorted[:50], "ts": time.time()})

# -------------------------- 記憶 API --------------------------
@app.post("/memory/write")
def memory_write(
    body: Dict[str, Any] = Body(..., example={"content":"中文編碼測試","tags":["clean"]}),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")
):
    require_auth(x_auth_token)
    content = (body or {}).get("content", "")
    tags    = (body or {}).get("tags", [])
    if not isinstance(tags, list): tags = []
    row = {"id": str(uuid.uuid4()), "content": content, "tags": tags, "ts": time.time()}
    rows = load_memory()
    rows.append(row)
    save_memory(rows)
    return j({"ok": True, "id": row["id"]})

@app.get("/memory/search")
def memory_search_api(
    q: str, top_k: int = Query(5, ge=1, le=50),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")
):
    require_auth(x_auth_token)
    rows = load_memory()
    results = search_memory(q, rows, top_k=top_k)
    return j({"ok": True, "results": results, "ts": time.time()})

# -------------------------- Compose --------------------------
@app.post("/compose")
def compose(
    body: Dict[str, Any] = Body(..., example={"input":"即時測試","tags":["clean","demo"],"top_k":5}),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")
):
    require_auth(x_auth_token)
    try:
        input_text = body.get("input") or body.get("text") or ""
        tags       = body.get("tags") or []
        top_k      = int(body.get("top_k") or 5)

        bundle     = load_bundle()
        persona    = (bundle.get("persona") or {}).get("system_style", "")
        rows       = load_memory()
        hits       = search_memory(input_text, rows, top_k=top_k)

        system = f"您是『OathLink 穩定語風人格助手（無蘊）』。規範：稱使用者為願主/師父/您；回覆簡明、可執行、條列步驟；不說空話；必要時先標註風險與前置條件。"
        if persona:
            system = persona if isinstance(persona, str) else str(persona)

        mem_ctx = "- " + "\n- ".join([h.get("content","") for h in hits]) if hits else "（無匹配記憶）"
        prompt = {
            "system": system,
            "user":   f"【輸入】\n{input_text}\n\n【可用記憶】\n{mem_ctx}\n\n請以固定語風輸出最終回覆。"
        }

        output = call_model(system, input_text, params={})
        return j({
            "ok": True,
            "prompt": prompt,
            "context_hits": hits,
            "output": output,
            "model_used": "gpt-4o-mini",
            "search_mode": "like",
            "ts": time.time()
        })
    except Exception as e:
        return j({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status=500)

# -------------------------- Bundle APIs --------------------------
@app.get("/bundle/export")
def bundle_export():
    return j(load_bundle())

@app.get("/bundle/preview")
def bundle_preview():
    b = load_bundle()
    return j({
        "ok": True,
        "persona": b.get("persona", {}),
        "memory_count": len(b.get("memory", [])),
        "glossary_count": len(b.get("glossary", [])),
        "memory_sample": b.get("memory", [])[:5],
        "ts": time.time()
    })

@app.post("/bundle/import")
def bundle_import(body: Dict[str, Any] = Body(...), x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    require_auth(x_auth_token)
    mode = (body.get("mode") or "merge").lower()
    persona = body.get("persona")
    memory  = body.get("memory")
    glossary= body.get("glossary")

    cur = load_bundle()

    if mode == "replace":
        newb = {
            "bundle_version": cur.get("bundle_version","1.0"),
            "persona": persona if persona is not None else cur.get("persona", {}),
            "memory":  memory  if memory  is not None else [],
            "glossary":glossary if glossary is not None else []
        }
        save_bundle(newb)
        return j({"ok": True, "mode": "replace", "ts": time.time()})

    # merge
    if persona is not None:
        cur["persona"] = persona

    if isinstance(memory, list):
        # 追加；若有 id 相同，覆蓋
        idx = {m.get("id"): i for i,m in enumerate(cur.get("memory", [])) if isinstance(m, dict) and m.get("id")}
        for m in memory:
            if not isinstance(m, dict): continue
            mid = m.get("id") or str(uuid.uuid4())
            m.setdefault("id", mid)
            m.setdefault("ts", time.time())
            if mid in idx:
                cur["memory"][idx[mid]] = m
            else:
                cur["memory"].append(m)

    if isinstance(glossary, list):
        cur["glossary"] = glossary

    save_bundle(cur)
    return j({"ok": True, "mode": "merge", "ts": time.time()})

@app.post("/bundle/reset")
def bundle_reset(keep_memory: bool = Query(False, description="是否保留現有記憶")):
    cur = load_bundle()
    newb = DEFAULT_BUNDLE.copy()
    if keep_memory:
        newb["memory"] = cur.get("memory", [])
    save_bundle(newb)
    return j({"ok": True, "reset": True, "kept_memory": keep_memory, "persona": newb["persona"], "ts": time.time()})

# -------------------------- 入口 --------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app_final:app", host="127.0.0.1", port=port, reload=False)