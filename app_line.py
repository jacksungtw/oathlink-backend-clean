# app_line.py —— LINE Bot + 綠界金流（含 CheckMacValue 驗證）+ SQLite 落盤 + CORS
# 適用：Python 3.10+、Flask、requests、waitress（Windows 建議）
# 啟動（建議）：python -m waitress --listen=127.0.0.1:5000 app_line:app

import os, json, time, random, hmac, hashlib, base64, urllib.parse, sqlite3, traceback
from datetime import datetime
import logging
import requests
from flask import Flask, request, jsonify, make_response, abort
import re

def _norm_base(url: str) -> str:
    if not url:
        return ""
    # 去不可見空白、全形空白
    url = re.sub(r"\s+", "", url)
    # 移除尾端斜線
    return url.rstrip("/")

PUBLIC_BASE_URL = _norm_base(os.getenv("PUBLIC_BASE_URL", ""))

# =========[ 環境變數 ]=========
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
PUBLIC_BASE_URL           = os.getenv("PUBLIC_BASE_URL", "")  # 例：https://xxxxx.ngrok-free.app
ECPAY_MERCHANT_ID         = os.getenv("ECPAY_MERCHANT_ID", "2000132")        # 測試
ECPAY_HASH_KEY            = os.getenv("ECPAY_HASH_KEY", "pwFHCqoQZGmho4w6")  # 測試
ECPAY_HASH_IV             = os.getenv("ECPAY_HASH_IV", "EkRm7iFT261dpevs")   # 測試
ECPAY_STAGE               = os.getenv("ECPAY_STAGE", "true").lower() == "true"

# =========[ Flask 與日誌 ]=========
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- CORS（Flask 原生處理）---
ALLOWED_ORIGINS = "*"   # 可改成白名單：["http://localhost:8501", "https://你的前端網域"]

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    allow_origin = "*" if ALLOWED_ORIGINS == "*" else (origin if origin in ALLOWED_ORIGINS else "null")
    resp.headers["Access-Control-Allow-Origin"] = allow_origin
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Auth-Token"
    return resp

# 統一處理預檢，避免 405/400
@app.route("/", methods=["OPTIONS"])
@app.route("/<path:any_path>", methods=["OPTIONS"])
def cors_preflight(any_path=None):
    return ("", 200)

# favicon 避免 405 噪音
@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return ("", 204)

# =========[ 訂單暫存 + SQLite 落盤 ]=========
ORDERS = {}  # 記憶體暫存（重啟會失），SQLite 會永久化
DB_PATH = "orders.db"

def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders(
        trade_no   TEXT PRIMARY KEY,
        amount     INTEGER,
        user_id    TEXT,
        status     TEXT,
        created_at INTEGER
    );
    """)
    con.commit(); con.close()

def db_upsert(trade_no, amount, user_id, status):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO orders(trade_no, amount, user_id, status, created_at)
        VALUES(?,?,?,?, strftime('%s','now'))
        ON CONFLICT(trade_no) DO UPDATE SET
          amount=excluded.amount, user_id=excluded.user_id, status=excluded.status;
    """, (trade_no, amount, user_id, status))
    con.commit(); con.close()

db_init()

# =========[ LINE API —— 無 SDK 直呼 ]=========
def line_reply_text(reply_token: str, text: str):
    if not (LINE_CHANNEL_ACCESS_TOKEN and reply_token):
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text[:4900]}]}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        if r.status_code >= 300:
            app.logger.warning(f"[LINE reply] http={r.status_code} body={r.text}")
    except Exception:
        app.logger.exception("[LINE reply] exception")

def line_push_text(to_user_id: str, text: str):
    if not (LINE_CHANNEL_ACCESS_TOKEN and to_user_id):
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    payload = {"to": to_user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        if r.status_code >= 300:
            app.logger.warning(f"[LINE push] http={r.status_code} body={r.text}")
    except Exception:
        app.logger.exception("[LINE push] exception")

def verify_line_signature(raw_body: bytes, signature: str) -> bool:
    if not (LINE_CHANNEL_SECRET and signature):
        # 測試期允許略過
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expect, signature)

# =========[ 綠界工具：CheckMacValue & 端點 ]=========
def _ecpay_encode(s: str) -> str:
    # 綠界規格：URL-encode 後改成小寫百分比，再做 MD5、轉大寫
    return urllib.parse.quote_plus(s, safe='-_.!*()')

def ecpay_checkmac(params: dict) -> str:
    data = {k: v for k, v in params.items() if k.lower() != "checkmacvalue"}
    sorted_items = sorted(data.items(), key=lambda x: x[0].lower())
    raw = f"HashKey={ECPAY_HASH_KEY}&" + "&".join([f"{k}={v}" for k, v in sorted_items]) + f"&HashIV={ECPAY_HASH_IV}"
    encoded = _ecpay_encode(raw).lower()
    mac = hashlib.md5(encoded.encode("utf-8")).hexdigest().upper()
    return mac

def ecpay_endpoint() -> str:
    return (
        "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5"
        if ECPAY_STAGE else
        "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5"
    )

def html_escape(s):
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))

# =========[ 路由 ]=========
@app.get("/")
def root():
    return "OK"

@app.get("/health")
def health():
    resp = make_response(jsonify(ok=True, ts=int(time.time())))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# =====[ LINE webhook ]=====
@app.post("/callback")
def callback():
    try:
        signature = request.headers.get("X-Line-Signature", "")
        raw = request.get_data()  # bytes
        if not verify_line_signature(raw, signature):
            return "Invalid signature", 400

        body = request.get_json(silent=True) or {}
        events = body.get("events", [])
        for ev in events:
            etype = ev.get("type")
            if etype == "message":
                msg = ev.get("message", {})
                if msg.get("type") == "text":
                    text = (msg.get("text") or "").strip()
                    reply_token = ev.get("replyToken", "")
                    user_id = ev.get("source", {}).get("userId", "")

                    # /pay 指令
                    if text.lower().startswith("/pay"):
                        try:
                            amount = int(text.split()[1])
                        except Exception:
                            amount = 300

                        if not PUBLIC_BASE_URL:
                            line_reply_text(reply_token, "願主、師父，尚未設定 PUBLIC_BASE_URL，無法產出金流連結。")
                        else:
                            trade_no = f"T{int(time.time())}{random.randint(100,999)}"
                            ORDERS[trade_no] = {"amount": amount, "userId": user_id}
                            db_upsert(trade_no, amount, user_id, "INIT")
                            pay_url = f"{PUBLIC_BASE_URL}/pay?amount={amount}&tno={trade_no}"
                            msg = (
                                f"願主、師父，為您建立訂單 {trade_no} 金額 {amount} 元。\n"
                                f"👉 點此付款：{pay_url}"
                            )
                            line_reply_text(reply_token, msg)
                    else:
                        # 回聲＋敬語
                        line_reply_text(reply_token, f"願主、師父，您剛才說：{text}")

        return "OK"
    except Exception:
        app.logger.exception("/callback error")
        return "Internal Server Error", 500

# =====[ 付款連結：導至綠界 ]=====
@app.get("/pay")
def pay_page():
    """
    用法：
      GET /pay?amount=300&tno=T123456
    回傳自動送出的 HTML 表單，跳至綠界 Cashier。
    """
    try:
        if not PUBLIC_BASE_URL:
            app.logger.error("[/pay] PUBLIC_BASE_URL 未設定")
            return "PUBLIC_BASE_URL 未設定", 400

        raw_amount = (request.args.get("amount") or "").strip()
        try:
            amount = int(raw_amount) if raw_amount else 300
        except Exception:
            amount = 300

        trade_no = (request.args.get("tno") or "").strip()
        if not trade_no:
            trade_no = f"T{int(time.time())}{random.randint(100,999)}"

        order = ORDERS.get(trade_no, {})
        user_id = order.get("userId", "")
        db_upsert(trade_no, amount, user_id, "INIT")

        trade_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        item_name  = "智慧廟宇-功德金"
        trade_desc = "TempleDonation"
        return_url = f"{PUBLIC_BASE_URL}/ecpay_return"

        params = {
            "MerchantID":        ECPAY_MERCHANT_ID,
            "MerchantTradeNo":   trade_no,
            "MerchantTradeDate": trade_date,
            "PaymentType":       "aio",
            "TotalAmount":       str(amount),
            "TradeDesc":         trade_desc,
            "ItemName":          item_name,
            "ReturnURL":         return_url,
            "ChoosePayment":     "Credit",
            "ClientBackURL":     PUBLIC_BASE_URL,
        }
        params["CheckMacValue"] = ecpay_checkmac(params)

        fields = "\n".join(
            f'<input type="hidden" name="{html_escape(k)}" value="{html_escape(v)}"/>'
            for k, v in params.items()
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>前往綠界付款</title></head>"
            "<body onload=\"document.forms[0].submit();\" style='font-family: system-ui, sans-serif;'>"
            f"<form method=\"post\" action=\"{ecpay_endpoint()}\">"
            f"{fields}"
            "<noscript><button type=\"submit\">前往綠界付款</button></noscript>"
            "</form>"
            "<p>正在前往綠界金流… 若未自動跳轉，請點選上方按鈕。</p>"
            "</body></html>"
        )
        resp = make_response(html, 200)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    except Exception as e:
        app.logger.exception(f"/pay error: {e}")
        return "Internal Server Error", 500

# =====[ 綠界回傳：驗 MAC、落盤、推播 ]=====
@app.post("/ecpay_return")
def ecpay_return():
    try:
        form = request.form.to_dict()
        recv_mac = form.pop("CheckMacValue", "")
        calc_mac = ecpay_checkmac(form)

        if recv_mac != calc_mac:
            app.logger.warning(f"[ECPAY] CheckMacValue 不符 recv={recv_mac} calc={calc_mac}")
            return "0|ERR_MAC", 200

        tno = form.get("MerchantTradeNo", "")
        rtn_code = form.get("RtnCode", "")      # 1 = 成功（信用卡即時）
        rtn_msg  = form.get("RtnMsg", "")
        amt      = int(form.get("TradeAmt", "0") or 0)

        order = ORDERS.get(tno, {})
        user_id = order.get("userId", "")

        status = "PAID" if rtn_code == "1" else f"RtnCode_{rtn_code}"
        db_upsert(tno, amt or order.get("amount", 0), user_id, status)

        if rtn_code == "1" and user_id:
            line_push_text(user_id, f"願主、師父，訂單 {tno} 已完成付款（{amt} 元）。功德圓滿。")

        app.logger.info(f"[ECPAY] tno={tno} rtn={rtn_code} msg={rtn_msg} amount={amt}")
        return "1|OK", 200
    except Exception:
        app.logger.exception("/ecpay_return error")
        return "0|ERR_EX", 200

# =====[ （可選）付款完成回到這裡顯示結果 ]=====
@app.get("/pay_result")
def pay_result():
    return "<h3>願主、師父，交易流程已完成，請返回 LINE 查看通知。</h3>", 200

# =====[ 全域錯誤保險 ]=====
@app.errorhandler(Exception)
def handle_any(e):
    app.logger.exception(f"Unhandled error: {e}")
    return "Internal Server Error", 500
