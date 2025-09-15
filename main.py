import os
import time
import uuid
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# 基本設定
# -------------------------
APP_NAME = "wuyun-agent"
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BUNDLE_PATH = os.path.join(DATA_DIR, "bundle.json")  # 用於本地保存 persona + memory

# 預設 persona（避免空值亂碼）
DEFAULT_PERSONA = "無蘊-敬語版"

# -------------------------
# FastAPI 準備
# -------------------------
app = FastAPI(title=APP_NAME, docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------------------------
# 共用：UTF-8 JSON Response（避免亂碼）
# -------------------------
def utf8_json(data: Dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(
        content=json.loads(json.dumps(data, ensure_ascii=False)),
        status_code=status,
        media_type="application/json; charset=utf-8",
    )

# -------------------------
# 本地 Bundle 存取（UTF-8 / 無 BOM）
# -------------------------
def load_bundle() -> Dict[str, Any]:
    if not os.path.exists(BUNDLE_PATH):
        # 初次啟動：建立空 bundle
        bundle = {
            "bundle_version": "1.0",
            "persona": DEFAULT_PERSONA,
            "memory": [],
        }
        save_bundle(bundle)
        return bundle
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            # 檔案損毀時回退
            return {"bundle_version": "1.0", "persona": DEFAULT_PERSONA, "memory": []}

def save_bundle(bundle: Dict[str, Any]) -> None:
    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        # ensure_ascii=False => 正確輸出中文
        json.dump(bundle, f, ensure_ascii=False, indent=2)

# -------------------------
# 簡易記憶檢索：LIKE（之後可換 FTS/向量）
# -------------------------
def memory_search(haystack: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    scored = []
    for row in haystack:
        c = row.get("content", "") or ""
        # 最簡單：子字串命中長度
        score = c.count(q) if q in c else 0
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], -float(x[1].get("ts", 0.0))))
    return [r for _, r in scored[:top_k]]

# -------------------------
# Pydantic 請求模型
# -------------------------
class ComposeBody(BaseModel):
    input: str = Field(..., description="使用者輸入")
    top_k: int = Field(3, description="檢索記憶數量（上限）")
    system_override: Optional[str] = Field(None, description="臨時覆寫系統語風")

# -------------------------
# 路由：健康檢查 / 可用路由
# -------------------------
@app.get("/health")
def health():
    return utf8_json({"ok": True, "service": APP_NAME, "ts": time.time()})

@app.get("/routes")
def routes():
    return utf8_json({
        "ok": True,
        "routes": {
            "/openapi.json",
            "/docs",
            "/redoc",
            "/health",
            "/routes",
            "/compose",
            "/bundle/import",
            "/bundle/export",
            "/bundle/preview",
        },
    })

# -------------------------
# 路由：compose（可離線回應；有 OPENAI_API_KEY 也先走模板，之後您可接真模型）
# -------------------------
@app.post("/compose")
async def compose(body: ComposeBody):
    bundle = load_bundle()
    persona = bundle.get("persona") or DEFAULT_PERSONA
    memory = bundle.get("memory", [])

    hits = memory_search(memory, body.input, top_k=body.top_k)

    # 系統語風（可被覆寫）
    system_text = body.system_override or (
        "稱呼願主/師父/您；回覆簡明、可執行、條列步驟；"
        "必要時先標註風險與前置條件；不得妄稱你。"
    )

    # 拼 Prompt（確保中文）
    user_block = f"【輸入】\n{body.input}\n\n【可用記憶】"
    if hits:
        for h in hits:
            user_block += f"\n- {h.get('content','')}"
    else:
        user_block += "\n- （無匹配記憶）"

    # 提供一個穩定的本地模板輸出（如需接模型，可在此處接 OpenAI/Anthropic/Ollama）
    output = [
        "願主，以下為基於您輸入與可用記憶所整理之回覆：",
        f"1) 已整合輸入：{body.input}",
    ]
    if hits:
        output.append(f"2) 命中記憶：{len(hits)} 條；已併入語境。")
    else:
        output.append("2) 當前無命中記憶；已以通用模板回覆。")

    # 如您要接 OpenAI，可在此判斷 OPENAI_API_KEY 並改為真推理
    # 這裡先固定 model/provider 字樣，確保與您之前腳本的解析相容
    resp = {
        "ok": True,
        "prompt": {
            "system": system_text,
            "user": user_block,
        },
        "context_hits": hits,
        "output": "\n".join(output),
        "model_used": "gpt-4o-mini",
        "provider": os.environ.get("LLM_PROVIDER", "local"),
        "ts": time.time(),
    }
    return utf8_json(resp)

# -------------------------
# 路由：bundle/export（導出完整 bundle）
# -------------------------
@app.get("/bundle/export")
def bundle_export():
    bundle = load_bundle()
    resp = {
        "ok": True,
        "bundle_version": bundle.get("bundle_version", "1.0"),
        "persona": bundle.get("persona", DEFAULT_PERSONA),
        "memory": bundle.get("memory", []),
        "count": len(bundle.get("memory", [])),
    }
    return utf8_json(resp)

# -------------------------
# 路由：bundle/preview（摘要預覽）
# -------------------------
@app.get("/bundle/preview")
def bundle_preview():
    bundle = load_bundle()
    mem = bundle.get("memory", [])
    latest_ts = 0.0
    if mem:
        latest_ts = max(float(m.get("ts", 0.0)) for m in mem)
    sample = mem[:5]
    resp = {
        "ok": True,
        "persona": bundle.get("persona", DEFAULT_PERSONA),
        "count_memory": len(mem),
        "latest_ts": latest_ts,
        "sample": sample,
    }
    return utf8_json(resp)

# -------------------------
# 路由：bundle/import
# 1) multipart/form-data: field name = "file"（支援您 PowerShell 的做法）
# 2) application/json  : 直接傳 JSON
# -------------------------
@app.post("/bundle/import")
async def bundle_import(request: Request, file: UploadFile = File(None)):
    content_type = request.headers.get("content-type", "")

    # 情境 A：multipart/form-data，上傳檔案欄位 "file"
    if "multipart/form-data" in content_type:
        if not file:
            raise HTTPException(status_code=400, detail="Field 'file' is required in multipart")
        try:
            raw = await file.read()
            # 嚴格用 UTF-8 解析（不吃 BOM）
            payload = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid UTF-8 file: {e}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # 情境 B：application/json，直接讀 body
    elif "application/json" in content_type:
        try:
            raw = await request.body()
            payload = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid UTF-8 body: {e}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    # 驗證/合併
    bundle = load_bundle()
    bundle["bundle_version"] = payload.get("bundle_version", "1.0")
    persona_in = payload.get("persona")
    if persona_in:
        bundle["persona"] = persona_in

    imported = 0
    incoming = payload.get("memory", []) or []
    if incoming:
        mem = bundle.get("memory", [])
        # 去重邏輯：用 (content, ts) 當 key；若無 id，自動補 id
        seen = {(m.get("content",""), float(m.get("ts",0.0))) for m in mem}
        for m in incoming:
            content = m.get("content", "")
            ts = float(m.get("ts", 0.0))
            key = (content, ts)
            if key not in seen:
                if "id" not in m or not m["id"]:
                    m["id"] = str(uuid.uuid4())
                if "tags" in m and isinstance(m["tags"], list):
                    # 確認 tags 都是 str
                    m["tags"] = [str(t) for t in m["tags"]]
                mem.append(m)
                seen.add(key)
                imported += 1
        bundle["memory"] = mem

    save_bundle(bundle)
    return utf8_json({
        "ok": True,
        "bundle_version": bundle.get("bundle_version","1.0"),
        "persona_saved": bool(persona_in),
        "imported": imported,
        "ts": time.time(),
    })