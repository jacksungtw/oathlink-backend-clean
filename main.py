# === 放在檔案最上方 ===
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import ORJSONResponse
import orjson

# 讓 orjson 用 UTF-8 並且不要把中文轉成 \uXXXX
def _orjson_dumps(v, *, default):
    return orjson.dumps(
        v,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z | orjson.OPT_INDENT_2
    )

app = FastAPI(default_response_class=ORJSONResponse)
ORJSONResponse.render = staticmethod(lambda v: orjson.dumps(v, option=orjson.OPT_INDENT_2))

# 你原本的 in-memory 存儲（示例）
MEM = []
PERSONA = "無蘊-敬語版"

@app.get("/routes")
def routes():
    return {"ok": True, "routes": [r.path for r in app.router.routes]}

# 匯入：支援 multipart 上傳 .json（UTF-8 / 無 BOM）
@app.post("/bundle/import")
async def bundle_import(file: UploadFile = File(...)):
    raw = await file.read()
    # 這行很關鍵：一定用 utf-8 解碼，再交給 json.loads
    data = json.loads(raw.decode("utf-8"))
    global PERSONA, MEM
    PERSONA = data.get("persona") or PERSONA

    imported = 0
    for item in data.get("memory", []):
        # 這裡不做任何 encode/decode，保持 Python 字串（Unicode）
        MEM.append({
            "id": item.get("id") or "",
            "content": item.get("content") or "",
            "tags": item.get("tags") or [],
            "ts": item.get("ts") or 0.0,
        })
        imported += 1

    return {
        "ok": True,
        "bundle_version": data.get("bundle_version", "1.0"),
        "persona_saved": True,
        "imported": imported,
    }

# 預覽：直接把 Python 字串原樣回傳（不要再手動 encode/decode）
@app.get("/bundle/preview")
def bundle_preview():
    latest_ts = max((m.get("ts") or 0.0) for m in MEM) if MEM else 0.0
    sample = MEM[:2]
    return {
        "ok": True,
        "persona": PERSONA,           # 這裡不要 .encode() / .decode()
        "count_memory": len(MEM),
        "latest_ts": latest_ts,
        "sample": sample,
    }

# 匯出：同樣保持字串
@app.get("/bundle/export")
def bundle_export():
    return {
        "ok": True,
        "bundle_version": "1.0",
        "persona": PERSONA,
        "memory": MEM,
        "count": len(MEM),
    }