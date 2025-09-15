# D:\OathLink_MVP\main.py
import os, json, time, uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────────────
# 1) 建立 FastAPI 物件（要在所有 @app.route 之前）
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="wuyun-agent")

# ─────────────────────────────────────────────────────────────────────
# 2) 簡單的「端上可攜記憶」記憶體存放（可改成 SQLite）
#    以及 persona 預設（敬語版）
# ─────────────────────────────────────────────────────────────────────
PERSONA_KEY = "persona_name"
STORE: Dict[str, Any] = {
    PERSONA_KEY: "無蘊-敬語版",
    "memory": []  # list of {id, content, tags, ts}
}

def now_ts() -> float:
    return float(time.time())

def coerce_mem(rec: Dict[str, Any]) -> Dict[str, Any]:
    # 正規化欄位
    return {
        "id": rec.get("id") or str(uuid.uuid4()),
        "content": str(rec.get("content", "")),
        "tags": list(rec.get("tags", [])),
        "ts": float(rec.get("ts") or now_ts())
    }

# ─────────────────────────────────────────────────────────────────────
# 3) 基本工具：UTF-8 / UTF-8-BOM 自動解碼
# ─────────────────────────────────────────────────────────────────────
def decode_utf8_or_bom(b: bytes) -> str:
    # 先試 UTF-8-SIG（可吃 BOM），失敗再退回 UTF-8
    try:
        return b.decode("utf-8-sig")
    except Exception:
        return b.decode("utf-8")

# ─────────────────────────────────────────────────────────────────────
# 4) 健康/路由
# ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "service": "wuyun-agent", "ts": now_ts()}

@app.get("/routes")
def routes():
    return {
        "ok": True,
        "routes": [
            "/health", "/routes",
            "/compose",
            "/bundle/export", "/bundle/preview",
            "/bundle/import", "/bundle/import-json",
            "/persona/get", "/persona/put"
        ],
        "ts": now_ts()
    }

# ─────────────────────────────────────────────────────────────────────
# 5) compose（最小版：把 persona 與 hits 拼進去；有 OPENAI_API_KEY 再調用）
# ─────────────────────────────────────────────────────────────────────
@app.post("/compose")
async def compose(req: Request):
    """
    Body: { input: str, top_k?: int, system_override?: str }
    """
    body = await req.json()
    text: str = body.get("input", "")
    top_k: int = int(body.get("top_k", 2))
    system_override: Optional[str] = body.get("system_override")

    # 取 hits（很簡單的 like）
    hits = []
    kw = text.strip()
    if kw:
        for r in STORE["memory"]:
            if kw in r["content"]:
                hits.append(r)
            if len(hits) >= top_k:
                break

    # Persona（敬語規則）
    persona = system_override or "稱呼願主/師父/您；回覆簡明、可執行、條列步驟；先標註風險與前置條件；不得妄稱你。"

    # 若有 OPENAI_API_KEY 可在此串 OpenAI；這裡為保險先用本地模板輸出
    output = (
        "願主，以下為基於您輸入與可用記憶所整理之回覆：\n"
        f"1) 已整合輸入：{text or '（空）'}\n"
        "2) 若需雲端生成，請設定 OPENAI_API_KEY。"
    )

    prompt = {
        "system": persona,
        "user": f"【輸入】\n{text}\n\n【可用記憶】\n"
                + "\n".join(f"- {h['content']}" for h in hits) if hits else "- （無匹配記憶）"
    }

    return {
        "ok": True,
        "prompt": prompt,
        "context_hits": hits,
        "output": output,
        "model_used": "gpt-4o-mini",
        "provider": "openai",
        "ts": now_ts()
    }

# ─────────────────────────────────────────────────────────────────────
# 6) persona 讀寫
# ─────────────────────────────────────────────────────────────────────
@app.get("/persona/get")
def persona_get():
    return {"ok": True, "persona": STORE.get(PERSONA_KEY, "無蘊-敬語版"), "ts": now_ts()}

@app.post("/persona/put")
async def persona_put(req: Request):
    data = await req.json()
    name = data.get("persona") or data.get("name")
    if not name:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "persona is required"})
    STORE[PERSONA_KEY] = str(name)
    return {"ok": True, "saved": True, "persona": STORE[PERSONA_KEY], "ts": now_ts()}

# ─────────────────────────────────────────────────────────────────────
# 7) bundle：export / preview / import（multipart 與 raw-json 皆可）
#    統一資料形制：
#    {
#      "bundle_version":"1.0",
#      "persona":"無蘊-敬語版",
#      "memory":[{content,tags,ts,id?}, ...]
#    }
# ─────────────────────────────────────────────────────────────────────
def bundle_dict() -> Dict[str, Any]:
    return {
        "ok": True,
        "bundle_version": "1.0",
        "persona": STORE.get(PERSONA_KEY, "無蘊-敬語版"),
        "memory": STORE["memory"],
        "count": len(STORE["memory"]),
        "ts": now_ts()
    }

@app.get("/bundle/export")
def bundle_export():
    return bundle_dict()

@app.get("/bundle/preview")
def bundle_preview():
    mem = STORE["memory"]
    sample = mem[:5]
    latest_ts = max((m["ts"] for m in mem), default=0.0)
    return {
        "ok": True,
        "persona": STORE.get(PERSONA_KEY, "無蘊-敬語版"),
        "count_memory": len(mem),
        "latest_ts": latest_ts,
        "sample": sample
    }

@app.post("/bundle/import")
async def bundle_import(file: UploadFile = File(...)):
    # 接受 multipart/form-data 的檔案欄位名 "file"
    raw = await file.read()
    text = decode_utf8_or_bom(raw)
    try:
        data = json.loads(text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {e}"})
    return _bundle_apply(data)

@app.post("/bundle/import-json")
async def bundle_import_json(req: Request):
    # 接受 Content-Type: application/json 直接 body
    raw = await req.body()
    text = decode_utf8_or_bom(raw)
    try:
        data = json.loads(text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {e}"})
    return _bundle_apply(data)

def _bundle_apply(data: Dict[str, Any]):
    # persona
    persona = data.get("persona")
    if isinstance(persona, str) and persona.strip():
        STORE[PERSONA_KEY] = persona.strip()

    # memory
    imported = 0
    for rec in data.get("memory", []):
        m = coerce_mem(rec)
        # 以 id 去重；若沒有 id 就直接新增
        exists = next((x for x in STORE["memory"] if x["id"] == m["id"]), None)
        if exists:
            # 若內容一樣就跳過
            if exists["content"] == m["content"] and exists["tags"] == m["tags"]:
                continue
            # 否則覆蓋（或您可改成保留舊的）
            exists.update(m)
        else:
            STORE["memory"].append(m)
            imported += 1

    return {
        "ok": True,
        "bundle_version": "1.0",
        "persona_saved": True,
        "imported": imported,
        "ts": now_ts()
    }