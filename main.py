# --- paste begin: persona & bundle utilities (UTF-8 safe) ---
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import io, json, time

app: FastAPI  # 假設您已有 app

# 內存暫存（或換成本地 sqlite）
_PERSONA: Dict[str, Any] = {
    "id": "wuyun-keigo",
    "name": "無蘊-敬語版",
    "system_style": "稱呼願主/師父/您；回覆簡明、可執行、條列步驟；先標註風險與前置條件；不得妄稱你。",
}
_MEMORY: List[Dict[str, Any]] = []

def _utf8_loads(b: bytes) -> Any:
    # 嘗試去掉 BOM
    if len(b) >= 3 and b[:3] == b'\xef\xbb\xbf':
        b = b[3:]
    return json.loads(b.decode("utf-8"))

def _now() -> float:
    return time.time()

@app.get("/persona/get")
def persona_get():
    return {"ok": True, "persona": _PERSONA, "ts": _now()}

@app.post("/persona/put")
def persona_put(data: Dict[str, Any]):
    # 允許以 JSON 設定 persona
    global _PERSONA
    _PERSONA = {
        "id": data.get("id", "wuyun-keigo"),
        "name": data.get("name", "無蘊-敬語版"),
        "system_style": data.get("system_style", "稱呼願主/師父/您；回覆簡明、可執行、條列步驟；先標註風險與前置條件；不得妄稱你。"),
    }
    return {"ok": True, "saved": True, "ts": _now()}

@app.get("/bundle/export")
def bundle_export():
    return {
        "ok": True,
        "bundle_version": "1.0",
        "persona": _PERSONA.get("name") or _PERSONA,
        "memory": _MEMORY,
        "count": len(_MEMORY),
        "ts": _now(),
    }

@app.post("/bundle/import")
async def bundle_import(file: UploadFile = File(...)):
    # 僅 multipart/form-data: file
    b = await file.read()
    try:
        data = _utf8_loads(b)
    except Exception as e:
        return JSONResponse({"ok": False, "detail": f"Invalid JSON: {e}"}, status_code=400)

    persona = data.get("persona")
    if persona:
        # 兼容字串或物件
        if isinstance(persona, str):
            _PERSONA["name"] = persona
        elif isinstance(persona, dict):
            _PERSONA.update(persona)

    imported = 0
    for m in data.get("memory", []):
        # 去重（以 content+ts 粗略判定）
        key = (m.get("content"), m.get("ts"))
        if not any((x.get("content"), x.get("ts")) == key for x in _MEMORY):
            _MEMORY.append({
                "id": m.get("id") or "",
                "content": m.get("content") or "",
                "tags": m.get("tags") or [],
                "ts": float(m.get("ts") or _now()),
            })
            imported += 1

    return {"ok": True, "bundle_version": data.get("bundle_version", "1.0"),
            "persona_saved": True, "imported": imported, "skipped": max(0, len(data.get("memory", [])) - imported),
            "ts": _now()}
# --- paste end ---