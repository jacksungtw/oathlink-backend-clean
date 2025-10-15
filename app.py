# -*- coding: utf-8 -*-
# OathLink Backend: GPT整合 + 人格注入 + 記憶回寫
# 依賴: fastapi, uvicorn, pydantic, python-multipart(如要上傳), requests
# pip install fastapi uvicorn pydantic requests

import os, json, sqlite3, time, re, unicodedata
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Body, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ====== 環境設定 ======
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE      = os.getenv("OPENAI_BASE", "https://api.openai.com/v1").strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com").strip()

AUTH_TOKEN       = os.getenv("AUTH_TOKEN", "abc123").strip()  # 可改為空字串表示不驗證
DB_PATH          = os.getenv("DB_PATH", "oathlink.db")

# ====== FastAPI ======
app = FastAPI(title="oathlink-backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== DB 初始化 ======
def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            content TEXT,
            tags TEXT,     -- 以JSON字串儲存陣列
            ts REAL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    # 預設人格（可自行修改）
    if not get_meta("persona_system"):
        set_meta("persona_system",
            "稱呼願主；簡明條列；先列風險與前置條件；"
            "保持一致語風；必要時提示缺少參數；不得妄稱你。"
        )
    if not get_meta("persona_name"):
        set_meta("persona_name", "OathLink 隨侍")
    conn.commit()
    conn.close()

def get_meta(key: str) -> Optional[str]:
    conn = db(); cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def set_meta(key: str, value: str):
    conn = db(); cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, value))
    conn.commit(); conn.close()

def write_memory(content: str, tags: Optional[List[str]] = None) -> str:
    _id = gen_id()
    conn = db(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO memory(id, content, tags, ts) VALUES (?,?,?,?)",
        (_id, content, json.dumps(tags or [] , ensure_ascii=False), time.time())
    )
    conn.commit(); conn.close()
    return _id

def search_memory_like(q: str, top_k: int = 5) -> List[Dict[str, Any]]:
    conn = db(); cur = conn.cursor()
    cur.execute(
        "SELECT id, content, tags, ts FROM memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
        (f"%{q}%", top_k)
    )
    rows = cur.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "id": r[0], "content": r[1],
            "tags": json.loads(r[2] or "[]"),
            "ts": r[3]
        })
    return res

def peek_memory(limit: int = 50) -> List[Dict[str, Any]]:
    conn = db(); cur = conn.cursor()
    cur.execute("SELECT id, content, tags, ts FROM memory ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); conn.close()
    return [{
        "id": r[0], "content": r[1],
        "tags": json.loads(r[2] or "[]"),
        "ts": r[3]
    } for r in rows]

def reset_db():
    conn = db(); cur = conn.cursor()
    cur.execute("DELETE FROM memory;")
    cur.execute("DELETE FROM meta;")
    conn.commit(); conn.close()
    # 重置預設人格
    set_meta("persona_system",
        "稱呼願主；簡明條列；先列風險與前置條件；保持一致語風；不得妄稱你。"
    )
    set_meta("persona_name", "OathLink 隨侍")

def gen_id() -> str:
    import uuid
    return str(uuid.uuid4())

def require_auth(x_auth_token: Optional[str]):
    if not AUTH_TOKEN:
        return
    if x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ====== 請求資料模型 ======
class WriteReq(BaseModel):
    content: str
    tags: Optional[List[str]] = []

class ComposeReq(BaseModel):
    input: str
    tags: Optional[List[str]] = []
    top_k: Optional[int] = 5
    model: Optional[str] = None      # 可覆寫模型
    provider: Optional[str] = None   # "openai" / "deepseek"

class BundleReq(BaseModel):
    memories: List[WriteReq] = []
    persona_system: Optional[str] = None
    persona_name: Optional[str] = None

class SettingsReq(BaseModel):
    persona_system: Optional[str] = None
    persona_name: Optional[str] = None

# ====== 基礎與除錯 ======
@app.get("/health")
def health():
    return {"ok": True, "service": "oathlink-backend", "ts": time.time()}

@app.post("/debug/reset")
def debug_reset(x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    reset_db()
    return {"ok": True, "reset": True, "ts": time.time()}

@app.get("/debug/peek")
def debug_peek(x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    return {"ok": True, "rows": peek_memory(), "ts": time.time()}

# ====== 記憶 ======
@app.post("/memory/write")
def memory_write(body: WriteReq, x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    _id = write_memory(body.content.strip(), body.tags or [])
    return {"ok": True, "id": _id}

@app.get("/memory/search")
def memory_search(q: str, top_k: int = 5, x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    return {"ok": True, "results": search_memory_like(q, top_k), "ts": time.time()}

# ====== 人格 / 設定 ======
@app.post("/settings")
def settings_update(body: SettingsReq, x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    if body.persona_system is not None:
        set_meta("persona_system", body.persona_system)
    if body.persona_name is not None:
        set_meta("persona_name", body.persona_name)
    return {
        "ok": True,
        "persona_system": get_meta("persona_system"),
        "persona_name": get_meta("persona_name"),
    }

@app.get("/settings")
def settings_get(x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    return {
        "ok": True,
        "persona_system": get_meta("persona_system"),
        "persona_name": get_meta("persona_name"),
    }

# ====== 封包(匯入/匯出/預覽) ======
@app.post("/bundle/import")
def bundle_import(body: BundleReq, x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    for m in body.memories or []:
        write_memory(m.content.strip(), m.tags or [])
    if body.persona_system is not None:
        set_meta("persona_system", body.persona_system)
    if body.persona_name is not None:
        set_meta("persona_name", body.persona_name)
    return {"ok": True}

@app.get("/bundle/export")
def bundle_export(x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    return {
        "ok": True,
        "memories": peek_memory(1000),
        "persona_system": get_meta("persona_system"),
        "persona_name": get_meta("persona_name"),
    }

@app.get("/bundle/preview")
def bundle_preview(x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    return {"ok": True, "rows": peek_memory(10)}

# ====== GPT 呼叫工具 ======
def call_openai_chat(messages: List[Dict[str, str]], model: str) -> str:
    if not OPENAI_API_KEY:
        return ""  # 無金鑰時由上層處理
    url = f"{OPENAI_BASE}/chat/completions".rstrip("/")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or OPENAI_MODEL, "messages": messages, "temperature": 0.3}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_deepseek_chat(messages: List[Dict[str, str]], model: str) -> str:
    if not DEEPSEEK_API_KEY:
        return ""
    url = f"{DEEPSEEK_BASE}/v1/chat/completions".rstrip("/")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or DEEPSEEK_MODEL, "messages": messages, "temperature": 0.3}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# ====== /compose：核心（人格注入 + 記憶檢索 + 生成 + 自動回寫） ======
@app.post("/compose")
def compose(body: ComposeReq, x_auth_token: Optional[str] = Header(None)):
    require_auth(x_auth_token)
    _input = (body.input or "").strip()
    if not _input:
        return {"ok": True, "output": "願主，尚未提供輸入文字。", "model_used": "local-fallback"}

    # 1) 取人格
    persona_sys = get_meta("persona_system") or ""
    persona_name = get_meta("persona_name") or "OathLink"

    # 2) 記憶檢索
    top_k = int(body.top_k or 5)
    hits = search_memory_like(_input[:100], top_k)  # 簡單LIKE，可換向量庫
    memory_block = ""
    if hits:
        lines = [f"- {h['content']}" for h in hits]
        memory_block = "\n".join(lines)

    # 3) 組 messages
    sys = f"{persona_sys}\n你的人格名號：{persona_name}。請保持統一語風。"
    user = f"【輸入】\n{_input}"
    if memory_block:
        user += f"\n\n【可用記憶】\n{memory_block}\n"

    messages = [
        {"role": "system", "content": sys},
        {"role": "user",   "content": user + "\n請輸出最終回覆。"}
    ]

    # 4) 呼叫模型（先用OpenAI，沒金鑰就試 DeepSeek，都沒有則本地拼接）
    provider = (body.provider or "").lower().strip()
    model    = (body.model or "").strip()
    output_text = ""
    model_used = "local-fallback"

    try:
        if provider in ("openai", "") and OPENAI_API_KEY:
            output_text = call_openai_chat(messages, model or OPENAI_MODEL)
            model_used = model or OPENAI_MODEL
        elif provider == "deepseek" and DEEPSEEK_API_KEY:
            output_text = call_deepseek_chat(messages, model or DEEPSEEK_MODEL)
            model_used = model or DEEPSEEK_MODEL
        elif DEEPSEEK_API_KEY:
            output_text = call_deepseek_chat(messages, model or DEEPSEEK_MODEL)
            model_used = model or DEEPSEEK_MODEL
        else:
            # 無外部金鑰 → 本地退化模式
            output_text = (
                "願主，以下為本地退化模式（未連雲）之回覆：\n"
                "1) 已整合輸入與歷史記憶。\n"
                "2) 若需更精煉文本，請設定 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。"
            )
    except Exception as e:
        output_text = f"願主，外部生成失敗：{str(e)}。改用本地退化模式。"
        model_used = "local-fallback"

    # 5) 自動回寫記憶：保存輸入與輸出（標上 tags）
    tags = body.tags or []
    try:
        write_memory(_input, ["compose_input"] + tags)
        write_memory(output_text, ["compose_output"] + tags)
    except Exception:
        pass

    # 6) 回傳
    return {
        "ok": True,
        "prompt": {
            "system": persona_sys,
            "user":   user
        },
        "context_hits": hits,
        "output": output_text,
        "model_used": model_used,
        "search_mode": "like",
        "ts": time.time(),
    }

# ====== 啟動時初始化 ======
init_db()
