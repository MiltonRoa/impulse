# -*- coding: utf-8 -*-
"""
BOT LIVE BINANCE FUTURES - ESTRATEGIA TREND (1m gated) con LightGBM
Correcciones solicitadas:
- p_long y p_short se calculan SIEMPRE (si features están listas), aunque luego no opere por gating.
- Se imprimen SIEMPRE todas las features (29) con sus valores en consola, Telegram y 2da línea del CSV.
- La lista de features se toma del modelo .pkl; si no existe, usa fallback de 29 features (los que pasaste).
"""

import asyncio
import json
import time
import math
import random
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import deque, defaultdict
import os
import sys
import aiohttp
import websockets
from binance import AsyncClient, BinanceAPIException, Client
import logging

import numpy as np
import pandas as pd
from joblib import load as joblib_load

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ======================================================================
# Credenciales (RECOMENDADO: variables de entorno)
# ======================================================================
API_KEY = 'X2oer6R2r3ZnRvcypc2K7TlDSKbb5GcjdeSi8l6nEkhbPW0lAYB5q7tTLSVQ2wFb'
API_SECRET = 'q9pNzEyldHeUml9wVIkorDEbb5wo0ZgqgI3szIxHtVuegtMI3SjDqPYpvfTAQdv2'
TELEGRAM_BOT_TOKEN = '7683862762:AAFsjnRZ0elEccfbn-0kI_pAiQUKk5vOJtQ'
TELEGRAM_CHAT_ID = '1485497941'
# ======================================================================
# Parámetros generales
# ======================================================================
LEVERAGE = 30
RISK_PER_TRADE = Decimal("200")
LOCAL_TZ = ZoneInfo("America/Asuncion")

FIXED_SYMBOLS = [
     "ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
    "OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","APTUSDT","FILUSDT","ATOMUSDT","INJUSDT",
    "AXSUSDT","RUNEUSDT","STXUSDT","GALAUSDT","SANDUSDT","EGLDUSDT","MINAUSDT","IMXUSDT",
    "BLURUSDT","1INCHUSDT","AAVEUSDT","ALGOUSDT","MANAUSDT","ORDIUSDT","WIFUSDT","JUPUSDT",
    "ENSUSDT","ARUSDT","SFPUSDT","CHZUSDT","TWTUSDT","HBARUSDT","BTCUSDT","BNBUSDT","LINKUSDT","TRXUSDT","UNIUSDT","NEARUSDT",
"ARBUSDT","MATICUSDT","ETCUSDT","XLMUSDT","ICPUSDT","FTMUSDT",
"SEIUSDT","TIAUSDT","LDOUSDT","RNDRUSDT","BOMEUSDT",

]

WS_PING_INTERVAL = 10
WS_PING_TIMEOUT = 60
REFILTER_INTERVAL = 100000

INIT_CANDLES = 500
ATR_PERIOD = 14
ATR5M_HISTORY_MAXLEN = 1000
ORDER_CHECK_INTERVAL = 1
TRADE_COOLDOWN_SECONDS = 2700  # 45 min
DEBUG_SYMBOLS_PER_MINUTE = 5

# ======================================================================
# Estrategia TREND (modelos + thresholds)
# ======================================================================
MODEL_TREND_LONG_PATH = Path(__file__).with_name("hit30_k2_v1_long.pkl")
MODEL_TREND_SHORT_PATH = Path(__file__).with_name("hit30_k2_v1_short.pkl")

# Umbrales (ajustalos)
P_TREND_LONG_THRESHOLD = 0.79
P_TREND_SHORT_THRESHOLD = 0.74

TP_K_ATR5M = 2.2
SL_M_ATR5M = 1.5

# EMA / geom
EMA_FAST = 9
EMA_MID  = 21
EMA_SLOW = 50
EMA_SLOWER = 100
EMA_SLOWEST = 200

SLOPE_LAGS = (3, 9, 21, 30)
CURV_LAGS = (5, 15)

VOL_Z_WIN = 120
VOL_Z_MIN = 60
ATR_Z_WIN = 200
ATR_Z_MIN = 100
TAKER_ROLL_20 = 20
TAKER_ROLL_30 = 30
CLIP = 50.0

# ======================================================================
# Utils
# ======================================================================
def _mean_std(values):
    cleaned = [float(v) for v in values if math.isfinite(float(v))]
    n = len(cleaned)
    if n == 0:
        return 0.0, 0.0
    mean = sum(cleaned) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in cleaned) / (n - 1)
    std = math.sqrt(variance)
    return mean, std

def count_decimal_places(value: str) -> int:
    d = Decimal(value)
    return max(0, -d.normalize().as_tuple().exponent)

def wilder_atr_update(prev_atr, tr: float, period: int) -> float:
    try:
        prev = float(prev_atr) if prev_atr is not None else None
    except Exception:
        prev = None
    if prev is None or (isinstance(prev, float) and (math.isnan(prev) or prev <= 0)):
        return float(tr)
    return float(prev) + (float(tr) - float(prev)) / float(period)

# ======================================================================
# Telegram
# ======================================================================
async def _send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    max_retries = 3
    delay = 1
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                response = await session.post(url, data=payload)
                if 200 <= response.status < 300:
                    return
                body = await response.text()
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[ERROR Telegram] Status {response.status}: {body}")
            except Exception as exc:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      "[ERROR Telegram] No se pudo enviar mensaje:", exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
    print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
          "[ERROR Telegram] Se agotaron los reintentos")

# ======================================================================
# CSV señales
# ======================================================================
SIGNALS_DIR = Path(__file__).resolve().parent / "signals"

def _ensure_signals_dir() -> None:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

def _signals_csv_path(timestamp: datetime) -> Path:
    date_str = timestamp.strftime("%Y-%m-%d")
    return SIGNALS_DIR / f"{date_str}.csv"

def _append_signal_to_csv(symbol: str, entry: str, tp: str, sl: str, timestamp: datetime, indicator_line: str) -> None:
    _ensure_signals_dir()
    path = _signals_csv_path(timestamp)
    header_line = f"{symbol},{entry},{tp},{sl},{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(header_line + "\n")
        fh.write(indicator_line + "\n")

async def _daily_signal_file_task() -> None:
    while True:
        now = datetime.now(LOCAL_TZ)
        _ensure_signals_dir()
        _signals_csv_path(now).touch(exist_ok=True)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sleep_seconds = max(1, (next_midnight - now).total_seconds())
        await asyncio.sleep(sleep_seconds)

# ======================================================================
# ML load
# ======================================================================
_MODEL_CACHE = {}
_MODEL_ATTEMPTED = set()

def _load_trained_model(path: Path | str):
    global _MODEL_CACHE, _MODEL_ATTEMPTED
    model_path = Path(path)
    if model_path in _MODEL_ATTEMPTED:
        return _MODEL_CACHE.get(model_path)
    _MODEL_ATTEMPTED.add(model_path)

    if not model_path.exists():
        print(f"[ML-MODEL][ERROR] No se encontró el archivo: {model_path}")
        _MODEL_CACHE[model_path] = None
        return None

    try:
        print(f"[ML-MODEL] Cargando modelo desde: {model_path}")
        _MODEL_CACHE[model_path] = joblib_load(model_path)
        print("[ML-MODEL] Modelo cargado correctamente.")
    except Exception as exc:
        logger.exception("Error al cargar el modelo ML", exc_info=exc)
        _MODEL_CACHE[model_path] = None

    return _MODEL_CACHE.get(model_path)

# ======================================================================
# Formateo features (IMPRIME TODAS)
# ======================================================================
def _fmt6(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return "nan"
        return f"{v:.6f}"
    except Exception:
        return "nan"

def _format_ml_features_multiline(symbol: str, features: dict, ordered_feature_names: list[str]) -> str:
    """
    Devuelve un texto multilínea con:
    - header
    - todas las features (ordered_feature_names) en múltiples líneas
    - metas al final
    """
    status = features.get("_status", "ok")
    reason = features.get("_reason", "")

    header = f"[ML-FEATURES] {symbol}"
    if status != "ok":
        header += f" | {status}"
        if reason:
            header += f" | {reason}"

    rows = []
    # features primero
    pairs = [f"{k}={_fmt6(features.get(k))}" for k in ordered_feature_names]

    # metas después
    meta_keys = [
        "p_long_raw","p_long","p_thr_long","p_short_raw","p_short","p_thr_short",
        "tp_k_atr5m","sl_m_atr5m",
        "trend_1m_long","trend_1m_short",
        "_status","_reason"
    ]
    for mk in meta_keys:
        if mk in ["_status","_reason"]:
            pairs.append(f"{mk}={features.get(mk,'')}")
        else:
            pairs.append(f"{mk}={_fmt6(features.get(mk))}")

    # chunk 6 por línea
    chunk = []
    for i, p in enumerate(pairs, start=1):
        chunk.append(p)
        if i % 6 == 0:
            rows.append(", ".join(chunk))
            chunk = []
    if chunk:
        rows.append(", ".join(chunk))

    return header + "\n" + "\n".join(rows)

def _format_indicator_line(metrics_line: str, ml_features: dict, ordered_feature_names: list[str]) -> str:
    """
    2da línea del CSV: metrics + todas las features + metas
    """
    parts = [metrics_line]
    for name in ordered_feature_names:
        parts.append(f"{name}={_fmt6(ml_features.get(name))}")

    # metas
    parts += [
        f"p_long_raw={_fmt6(ml_features.get('p_long_raw'))}",
        f"p_long={_fmt6(ml_features.get('p_long'))}",
        f"p_thr_long={_fmt6(ml_features.get('p_thr_long'))}",
        f"p_short_raw={_fmt6(ml_features.get('p_short_raw'))}",
        f"p_short={_fmt6(ml_features.get('p_short'))}",
        f"p_thr_short={_fmt6(ml_features.get('p_thr_short'))}",
        f"tp_k_atr5m={_fmt6(ml_features.get('tp_k_atr5m'))}",
        f"sl_m_atr5m={_fmt6(ml_features.get('sl_m_atr5m'))}",
        f"trend_1m_long={_fmt6(ml_features.get('trend_1m_long'))}",
        f"trend_1m_short={_fmt6(ml_features.get('trend_1m_short'))}",
        f"_status={ml_features.get('_status','ok')}",
        f"_reason={ml_features.get('_reason','')}",
    ]
    return " ".join(parts).strip()

# ======================================================================
# Binance: ALGO orders SL/TP
# ======================================================================
async def futures_create_algo_conditional(
    client,
    *,
    symbol,
    side,
    order_type,
    trigger_price,
    quantity=None,
    close_position=False,
    reduce_only=False,
    client_algo_id=None,
):
    params = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "triggerPrice": str(trigger_price),
        "workingType": "MARK_PRICE",
        "timestamp": int(time.time() * 1000),
    }
    if client_algo_id is not None:
        params["clientAlgoId"] = client_algo_id

    if close_position:
        params["closePosition"] = "true"
    elif reduce_only:
        if quantity is None:
            raise ValueError("quantity es obligatorio cuando reduce_only=True")
        params["reduceOnly"] = "true"
        params["quantity"] = str(quantity)
    elif quantity is not None:
        params["quantity"] = str(quantity)

    resp = await client._request_futures_api("post", "algoOrder", signed=True, data=params)
    algo_id = resp.get("algoId") if isinstance(resp, dict) else None
    return resp, algo_id

async def create_algo_sl_tp_orders(client, symbol, side, sl_price, tp_price, tp_quantity):
    try:
        sl_order, stop_algo_id = await futures_create_algo_conditional(
            client,
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            trigger_price=str(sl_price),
            close_position=True,
        )
    except Exception as exc:
        logger.exception("Error al crear SL (ALGO) para %s", symbol, exc_info=exc)
        await _send_telegram_message(
            "⚠️ ERROR al crear SL/TP (ALGO) "
            f"{symbol}. La posición quedó ABIERTA sin SL/TP.\n{exc}"
        )
        return None, None, None, None

    if tp_quantity is None:
        msg = "No se pudo determinar la cantidad real para TP; se deja la posición abierta"
        logger.error(msg)
        await _send_telegram_message(f"⚠️ ERROR al crear SL/TP (ALGO) {symbol}. {msg}")
        return sl_order, stop_algo_id, None, None

    try:
        tp_order, tp_algo_id = await futures_create_algo_conditional(
            client,
            symbol=symbol,
            side=side,
            order_type="TAKE_PROFIT_MARKET",
            trigger_price=str(tp_price),
            quantity=tp_quantity,
            reduce_only=True,
        )
    except Exception as exc:
        logger.exception("Error al crear TP (ALGO) para %s", symbol, exc_info=exc)
        await _send_telegram_message(
            "⚠️ ERROR al crear SL/TP (ALGO) "
            f"{symbol}. La posición quedó ABIERTA con SL, sin TP.\n{exc}"
        )
        return sl_order, stop_algo_id, None, None

    return sl_order, stop_algo_id, tp_order, tp_algo_id

# ======================================================================
# Orden de entrada + SL/TP (mantiene tu lógica)
# ======================================================================
async def place_futures_order_async(
    client,
    bot,
    sym,
    signal,  # "long" o "short"
    entry,
    sl,
    tp1,
    atr1m=None,
):
    now = datetime.now(LOCAL_TZ)

    atr_5m = bot.atr_values.get(sym) if hasattr(bot, "atr_values") else None
    atr_5m_str = f"{atr_5m:.6f}" if atr_5m is not None else "None"
    atr1m_str = f"{atr1m:.6f}" if atr1m is not None else "None"

    entry_str = f"{entry:.6f}"
    sl_str = f"{sl:.6f}"
    tp1_str = f"{tp1:.6f}"
    close1m_str = entry_str

    last_debug = getattr(bot, "_last_debug", None) or {}
    symbol_debug = last_debug.get(sym, {}) if isinstance(last_debug, dict) else {}
    ml_features = symbol_debug.get("ml_features", {}) if isinstance(symbol_debug, dict) else {}
    if not isinstance(ml_features, dict):
        ml_features = {}

    metrics_line = " | ".join([f"Close1m: {close1m_str}", f"ATR5m: {atr_5m_str}", f"ATR1m: {atr1m_str}"])

    # CSV (2 líneas)
    indicator_line = _format_indicator_line(metrics_line, ml_features, bot.feature_names_ordered)
    try:
        _append_signal_to_csv(sym, entry_str, tp1_str, sl_str, now, indicator_line)
    except Exception as exc:
        logger.exception("Error al guardar señal en CSV para %s", sym, exc_info=exc)

    # Telegram (con TODAS las features)
    price_line = f"Entry: {entry_str}  SL: {sl_str}  TP: {tp1_str}\n"
    features_block = _format_ml_features_multiline(sym, ml_features, bot.feature_names_ordered)
    telegram_text = (f"🔔 TREND {signal.upper()}\n{signal.upper()} {sym}\n" + price_line + metrics_line) + f"\n{features_block}"
    await _send_telegram_message(telegram_text)

    # margen aislado
    try:
        await client.futures_change_margin_type(symbol=sym, marginType="ISOLATED")
    except BinanceAPIException as e:
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "[INFO] No se pudo cambiar a margen aislado para", sym, "error:", e)

    # leverage
    leverage_to_use = LEVERAGE
    try:
        await client.futures_change_leverage(symbol=sym, leverage=LEVERAGE)
    except BinanceAPIException as e:
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– ERROR AL AJUSTAR APALANCAMIENTO (", sym, "):", e)
        try:
            brackets = await client.futures_leverage_bracket(symbol=sym)
            max_leverage = brackets[0]["brackets"][0]["initialLeverage"]
            if LEVERAGE > max_leverage:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"– Ajustando al apalancamiento máximo {max_leverage} para {sym}")
                await client.futures_change_leverage(symbol=sym, leverage=max_leverage)
                leverage_to_use = max_leverage
            else:
                await _send_telegram_message(f"⚠️ No se pudo abrir posición {signal.upper()} {sym}\nMotivo: {e}")
                return False
        except Exception as e2:
            print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                  "– ERROR al obtener o ajustar apalancamiento máximo (", sym, "):", e2)
            await _send_telegram_message(f"⚠️ No se pudo abrir posición {signal.upper()} {sym}\nMotivo: {e2}")
            return False

    # notional y qty
    notional = float(RISK_PER_TRADE * leverage_to_use)
    raw_qty = notional / float(entry)
    print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
          "– Cálculo notional:", notional, "raw_qty:", raw_qty)

    filter_info = getattr(bot, "symbol_filters", {}).get(sym)
    if filter_info is None:
        err = f"[ERROR] Filtros no encontrados para {sym}"
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), err)
        await _send_telegram_message(f"⚠️ No se pudo abrir posición {signal.upper()} {sym}\nMotivo: {err}")
        return False

    step_size = filter_info["step_size"]
    min_qty = filter_info["min_qty"]
    precision_qty = filter_info["precision_qty"]
    tick_size = filter_info["tick_size"]
    precision_price = filter_info["precision_price"]
    min_notional = filter_info["min_notional"]

    qty = math.floor(raw_qty / step_size) * step_size
    quantity_num = qty
    quantity = f"{quantity_num:.{precision_qty}f}"

    if quantity_num < min_qty:
        quantity_num = min_qty
        quantity = f"{quantity_num:.{precision_qty}f}"

    if min_notional > 0:
        nominal = quantity_num * float(entry)
        if nominal < min_notional:
            needed_qty = min_notional / float(entry)
            qty2 = math.floor(needed_qty / step_size) * step_size
            if qty2 < min_qty:
                qty2 = min_qty
            quantity_num = qty2
            quantity = f"{quantity_num:.{precision_qty}f}"

    async def _real_tp_quantity():
        delays = [0, 0.2, 0.4, 0.6]
        position_amt = 0.0
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                pos = await client.futures_position_information(symbol=sym)
            except Exception as exc:
                logger.exception("Error al obtener la posición real para %s", sym, exc_info=exc)
                continue

            if isinstance(pos, list) and pos:
                entry_pos = next((p for p in pos if abs(float(p.get("positionAmt", 0))) > 0), None)
            else:
                entry_pos = pos if pos else None
            if entry_pos is None:
                continue

            try:
                position_amt = abs(float(entry_pos.get("positionAmt", 0)))
            except (TypeError, ValueError):
                position_amt = 0.0

            if position_amt > 0:
                break

        tp_qty_num = math.floor(position_amt / step_size) * step_size
        if tp_qty_num < min_qty:
            return None
        tp_qty_str = f"{tp_qty_num:.{precision_qty}f}"
        if float(tp_qty_str) <= 0:
            return None
        return tp_qty_str

    # separación mínima por tick
    entry_rounded = float(f"{entry:.{precision_price}f}")
    min_offset = tick_size
    if signal == "long":
        if entry_rounded - sl < min_offset:
            sl = entry_rounded - min_offset
        if tp1 - entry_rounded < min_offset:
            tp1 = entry_rounded + min_offset
    else:
        if sl - entry_rounded < min_offset:
            sl = entry_rounded + min_offset
        if entry_rounded - tp1 < min_offset:
            tp1 = entry_rounded - min_offset

    sl_price = f"{sl:.{precision_price}f}"
    tp1_price = f"{tp1:.{precision_price}f}"

    side_order = Client.SIDE_BUY if signal == "long" else Client.SIDE_SELL
    try:
        await client.futures_create_order(symbol=sym, side=side_order, type="MARKET", quantity=quantity)
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– MERCADO:", signal.upper(), sym, ", qty=", quantity)
        tp_quantity = await _real_tp_quantity()
    except BinanceAPIException as e:
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– ERROR ORDEN MARKET (", sym, "):", e)
        await _send_telegram_message(f"⚠️ No se pudo abrir posición {signal.upper()} {sym}\nError MARKET: {e}")
        return False
    except Exception as e:
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– ERROR GENÉRICO MARKET (", sym, "):", e)
        await _send_telegram_message(f"⚠️ No se pudo abrir posición {signal.upper()} {sym}\nExcepción: {e}")
        return False

    opposite = Client.SIDE_SELL if signal == "long" else Client.SIDE_BUY

    try:
        sl_order, stop_algo_id, tp_order1, tp_algo_id = await create_algo_sl_tp_orders(
            client, sym, opposite, sl_price, tp1_price, tp_quantity
        )
        bot.active_trades[sym] = {
            "close_side": opposite,
            "stop_algo_id": stop_algo_id,
            "tp_algo_id": tp_algo_id,
            "entry_price": float(f"{entry:.{precision_price}f}"),
            "precision_price": precision_price,
        }
    except Exception as e:
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– ERROR al crear SL/TP (ALGO) (", sym, "):", e)
        logger.error("Fallo al programar SL/TP (ALGO) para %s", sym, exc_info=e)
        return False

    asyncio.create_task(bot._monitor_trade(sym))
    return True

# ======================================================================
# Bot
# ======================================================================
class TrendBot:
    def __init__(self, api_key, api_secret, enable_dashboard=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = []
        self.enable_dashboard = enable_dashboard
        self.dashboard_task = None

        self.active_trades = {}
        self.pending_orders = set()
        self.last_signal_time = {}

        # ATR / historia
        self.atr_values = {}          # ATR5m
        self.atr_values_1m = {}       # ATR1m (telemetry)
        self.prev_close_5m = {}
        self.prev_close_1m = {}
        self.atr5m_history = defaultdict(lambda: deque(maxlen=ATR5M_HISTORY_MAXLEN))
        self.atr5m_tr_buffer = defaultdict(lambda: deque(maxlen=ATR_PERIOD))

        # OHLC 1m hist
        self.closes_history = {}
        self.high_history = {}
        self.low_history = {}

        # EMA26 5m
        self.ema26_5m = {}

        # EMA 1m histories (para slope/curv)
        self.ema1m_hist = defaultdict(lambda: {
            "ema9": deque(maxlen=500),
            "ema21": deque(maxlen=500),
            "ema50": deque(maxlen=500),
            "ema100": deque(maxlen=500),
            "ema200": deque(maxlen=500),
        })

        # EMA 5m regime 9/21/50/100/200
        self.close5m_hist = defaultdict(lambda: deque(maxlen=INIT_CANDLES))
        self.ema5m_hist = defaultdict(lambda: {
            "ema9": deque(maxlen=500),
            "ema21": deque(maxlen=500),
            "ema50": deque(maxlen=500),
            "ema100": deque(maxlen=500),
            "ema200": deque(maxlen=500),
        })

        # struct history 1m
        self.struct_history_1m = defaultdict(lambda: deque(maxlen=INIT_CANDLES))

        # fan width hist (para fan_expansion)
        self.fanw_hist = defaultdict(lambda: deque(maxlen=200))

        # atr5m causal por minuto
        self.atr5m_causal_hist_1m = defaultdict(lambda: deque(maxlen=500))

        # filtros símbolo
        self.symbol_filters = {}

        # ws
        self.ws = None

        # debug
        self._last_debug = {}
        self._debug_symbols = set()
        self._debug_minute = None
        self._summary_minute = None
        self._summary_counts = None
        self.last_5m_close_time = {}
        self._last_stale_warning_minute = {}
        self._stack5m_state = {}

        # cargar modelos trend
        self.packed_long = _load_trained_model(MODEL_TREND_LONG_PATH)
        self.packed_short = _load_trained_model(MODEL_TREND_SHORT_PATH)

        self.model_long = self.packed_long.get("model") if isinstance(self.packed_long, dict) else None
        self.model_short = self.packed_short.get("model") if isinstance(self.packed_short, dict) else None
        self.best_iteration_long = self.packed_long.get("best_iteration") if isinstance(self.packed_long, dict) else None
        self.best_iteration_short = self.packed_short.get("best_iteration") if isinstance(self.packed_short, dict) else None
        self.calibrator_long = self.packed_long.get("calibrator") if isinstance(self.packed_long, dict) else None
        self.calibrator_short = self.packed_short.get("calibrator") if isinstance(self.packed_short, dict) else None

        # features desde pkl (obligatorias)
        self.features_long = self.packed_long.get("features") if isinstance(self.packed_long, dict) else None
        self.features_short = self.packed_short.get("features") if isinstance(self.packed_short, dict) else None

        if not isinstance(self.features_long, list) or not isinstance(self.features_short, list):
            raise RuntimeError("[INIT][ERROR] El modelo no incluye 'features' en ambos .pkl.")
        if self.features_long != self.features_short:
            raise RuntimeError("[INIT][ERROR] Las features long/short no coinciden.")
        if len(self.features_long) != 60:
            raise RuntimeError(f"[INIT][ERROR] Se esperaban 60 features, recibidas {len(self.features_long)}.")

        self.feature_names_ordered = list(self.features_long)

        print(f"[INIT] Features activas ({len(self.feature_names_ordered)}): {self.feature_names_ordered}")

    # ---------------------------
    # Helpers
    # ---------------------------
    def _is_finite(self, x) -> bool:
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    def _ema_next(self, prev_ema, price: float, span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        if prev_ema is None or (not self._is_finite(prev_ema)):
            return float(price)
        return float(price) * alpha + float(prev_ema) * (1.0 - alpha)

    def _init_ema_from_prices(self, prices: list[float], span: int) -> list[float]:
        out = []
        ema_val = None
        for p in prices:
            ema_val = self._ema_next(ema_val, float(p), span)
            out.append(float(ema_val))
        return out

    def _zscore_last(self, values: list[float], min_n: int) -> float:
        vals = [float(v) for v in values if self._is_finite(v)]
        if len(vals) < min_n:
            return 0.0
        mean, std = _mean_std(vals)
        return (vals[-1] - mean) / std if std > 0 else 0.0

    def _calc_trend_1m_flags(self, sym: str) -> tuple[float, float]:
        """trend_1m_long, trend_1m_short"""
        try:
            e1 = self.ema1m_hist[sym]
            if e1["ema9"] and e1["ema21"] and e1["ema50"]:
                ema9 = float(e1["ema9"][-1])
                ema21 = float(e1["ema21"][-1])
                ema50 = float(e1["ema50"][-1])
                long_flag = 1.0 if (ema9 > ema21 > ema50) else 0.0
                short_flag = 1.0 if (ema9 < ema21 < ema50) else 0.0
                return long_flag, short_flag
        except Exception:
            pass
        return 0.0, 0.0

    # ---------------------------
    # Features HIT (exact names del entrenamiento)
    # ---------------------------
    def _build_hit_features(self, sym, bar):
        """
        Devuelve: (feats:dict|None, reason:str|None)
        """
        def _not_ready(reason: str):
            return None, reason

        atr5m = self.atr_values.get(sym)
        if atr5m is None or (not self._is_finite(atr5m)) or atr5m <= 0:
            return _not_ready("ATR5m no listo")

        o = float(bar.get("open", 0.0))
        h = float(bar.get("high", 0.0))
        l = float(bar.get("low", 0.0))
        c = float(bar.get("close", 0.0))
        vol = float(bar.get("volume", 0.0))
        buy = float(bar.get("taker_buy_base_volume", 0.0))
        if not all(self._is_finite(x) for x in (o, h, l, c, vol, buy)):
            return _not_ready("OHLC/vol no finito")

        closes = list(self.closes_history.get(sym, []))
        if len(closes) < EMA_SLOWEST + 5:
            return _not_ready(f"closes_history corto ({len(closes)})")

        e1 = self.ema1m_hist[sym]
        e9_hist = e1["ema9"]
        e21_hist = e1["ema21"]
        e50_hist = e1["ema50"]
        e100_hist = e1["ema100"]
        e200_hist = e1["ema200"]

        needed_hist = max(max(SLOPE_LAGS), 2 * max(CURV_LAGS)) + 1
        if len(e200_hist) < needed_hist:
            return _not_ready(f"EMA1m hist corto ({len(e200_hist)})")

        ema9 = float(e9_hist[-1])
        ema21 = float(e21_hist[-1])
        ema50 = float(e50_hist[-1])
        ema100 = float(e100_hist[-1])
        ema200 = float(e200_hist[-1])

        def _slope(hist, lag):
            if len(hist) <= lag:
                return math.nan
            return (float(hist[-1]) - float(hist[-1 - lag])) / atr5m

        def _curv(hist, lag):
            if len(hist) <= (2 * lag):
                return math.nan
            t0 = float(hist[-1])
            t1 = float(hist[-1 - lag])
            t2 = float(hist[-1 - 2 * lag])
            return (t0 - 2.0 * t1 + t2) / atr5m

        # Features crudas del minuto actual (sin normalización)
        feats = {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
            "taker_buy_base_volume": buy,
            "atr5m": atr5m,
            "dist_close_ema9": (c - ema9) / atr5m,
            "dist_close_ema21": (c - ema21) / atr5m,
            "dist_close_ema50": (c - ema50) / atr5m,
            "dist_close_ema100": (c - ema100) / atr5m,
            "dist_close_ema200": (c - ema200) / atr5m,
            "spread_9_21": (ema9 - ema21) / atr5m,
            "spread_21_50": (ema21 - ema50) / atr5m,
            "spread_50_100": (ema50 - ema100) / atr5m,
            "spread_100_200": (ema100 - ema200) / atr5m,
            "spread_9_200": (ema9 - ema200) / atr5m,
        }

        for lag in SLOPE_LAGS:
            feats[f"slope_ema21_lag{lag}"] = _slope(e21_hist, lag)
            feats[f"slope_ema50_lag{lag}"] = _slope(e50_hist, lag)
            feats[f"slope_ema200_lag{lag}"] = _slope(e200_hist, lag)

        for lag in CURV_LAGS:
            feats[f"curv_ema21_lag{lag}"] = _curv(e21_hist, lag)
            feats[f"curv_ema50_lag{lag}"] = _curv(e50_hist, lag)

        rng = h - l
        if rng <= 0:
            return _not_ready("range<=0")
        body = abs(c - o)
        feats["range_to_atr"] = rng / atr5m
        feats["body_to_atr"] = body / atr5m

        fan_width = (
            abs(ema9 - ema21)
            + abs(ema21 - ema50)
            + abs(ema50 - ema100)
            + abs(ema100 - ema200)
        )
        fan_width_atr = fan_width / atr5m
        if not self._is_finite(fan_width_atr):
            return _not_ready("fan_width_atr no finito")
        feats["fan_width_atr"] = fan_width_atr

        self.fanw_hist[sym].append(float(fan_width_atr))
        for lag in (3, 9, 21):
            if len(self.fanw_hist[sym]) <= lag:
                return _not_ready(f"fan_width hist corto ({len(self.fanw_hist[sym])})")
            feats[f"fan_expansion_lag{lag}"] = fan_width_atr - float(self.fanw_hist[sym][-1 - lag])

        taker_imbalance = ((2.0 * buy - vol) / vol) if vol > 0 else 0.0
        feats["taker_imbalance"] = taker_imbalance

        hist_struct = list(self.struct_history_1m.get(sym, []))
        if len(hist_struct) < TAKER_ROLL_30:
            return _not_ready(f"struct_history corto ({len(hist_struct)})")

        def _mean_imbalance(last_n):
            values = []
            for hh in hist_struct[-last_n:]:
                vv = float(hh.get("volume", 0.0))
                bb = float(hh.get("taker_buy_base_volume", 0.0))
                if self._is_finite(vv) and self._is_finite(bb) and vv > 0:
                    values.append((2.0 * bb - vv) / vv)
                else:
                    values.append(0.0)
            return sum(values) / float(last_n)

        feats["taker_imbalance_20"] = _mean_imbalance(TAKER_ROLL_20)
        feats["taker_imbalance_30"] = _mean_imbalance(TAKER_ROLL_30)

        vol_series = []
        for hh in hist_struct[-VOL_Z_WIN:]:
            vv = float(hh.get("volume", 0.0))
            if self._is_finite(vv):
                vol_series.append(vv)
        if len(vol_series) < VOL_Z_MIN:
            return _not_ready(f"vol_series corto ({len(vol_series)})")
        feats["vol_z"] = self._zscore_last(vol_series, min_n=VOL_Z_MIN)

        atr_hist = list(self.atr5m_causal_hist_1m[sym])[-ATR_Z_WIN:]
        if len(atr_hist) < ATR_Z_MIN:
            return _not_ready(f"atr_hist corto ({len(atr_hist)})")
        feats["atr_z"] = self._zscore_last(atr_hist, min_n=ATR_Z_MIN)

        e5 = self.ema5m_hist[sym]
        if len(e5["ema200"]) < 4:
            return _not_ready(f"EMA5m hist corto ({len(e5['ema200'])})")

        ema9_5m = float(e5["ema9"][-1])
        ema21_5m = float(e5["ema21"][-1])
        ema26_5m = float(self.ema26_5m.get(sym, math.nan))
        ema50_5m = float(e5["ema50"][-1])
        ema100_5m = float(e5["ema100"][-1])
        ema200_5m = float(e5["ema200"][-1])

        # EMAs 5m crudas ya calculadas por el bot (sin recálculo)
        feats["ema9_5m"] = ema9_5m
        feats["ema21_5m"] = ema21_5m
        feats["ema26_5m"] = ema26_5m
        feats["ema50_5m"] = ema50_5m
        feats["ema100_5m"] = ema100_5m
        feats["ema200_5m"] = ema200_5m
        feats["ema_stack_5m_long"] = 1.0 if (ema9_5m > ema21_5m > ema50_5m > ema100_5m > ema200_5m) else 0.0
        feats["ema_stack_5m_short"] = 1.0 if (ema9_5m < ema21_5m < ema50_5m < ema100_5m < ema200_5m) else 0.0
        feats["dist_close_ema50_5m"] = (c - ema50_5m) / atr5m
        feats["dist_close_ema200_5m"] = (c - ema200_5m) / atr5m
        feats["spread_21_50_5m"] = (ema21_5m - ema50_5m) / atr5m
        feats["spread_50_200_5m"] = (ema50_5m - ema200_5m) / atr5m
        feats["slope_ema50_5m_1"] = (ema50_5m - float(e5["ema50"][-2])) / atr5m
        feats["slope_ema50_5m_3"] = (ema50_5m - float(e5["ema50"][-4])) / atr5m
        feats["slope_ema200_5m_1"] = (ema200_5m - float(e5["ema200"][-2])) / atr5m
        feats["slope_ema200_5m_3"] = (ema200_5m - float(e5["ema200"][-4])) / atr5m

        def _clip_value(value: float, low: float, high: float) -> float:
            if not self._is_finite(value):
                return value
            return min(max(value, low), high)

        symmetric_prefixes = (
            "dist_close_ema",
            "spread_",
            "slope_",
            "curv_",
            "fan_expansion_lag",
        )
        for key, value in list(feats.items()):
            if key.startswith(symmetric_prefixes):
                feats[key] = _clip_value(float(value), -CLIP, CLIP)

        for key in ("range_to_atr", "body_to_atr", "fan_width_atr"):
            if key in feats:
                feats[key] = _clip_value(float(feats[key]), 0.0, CLIP)

        for key in ("vol_z", "atr_z"):
            if key in feats:
                feats[key] = _clip_value(float(feats[key]), -10.0, 10.0)

        # Validación estricta: TODAS las features del modelo deben existir y ser finitas
        for k in self.feature_names_ordered:
            if k not in feats:
                return _not_ready(f"feature faltante {k}")
            if not self._is_finite(feats.get(k)):
                return _not_ready(f"feature no finita {k}")

        return feats, None

    # ---------------------------
    # Señal (ML) - p_long y p_short SIEMPRE
    # ---------------------------
    def get_ml_signal_for_latest_bar(self, sym, bar, is_warmup=False):
        # cooldown / trade activo (solo bloquea la operación, NO el cálculo)
        now_ts = time.time()
        if not is_warmup:
            last_ts = self.last_signal_time.get(sym, 0)
            if now_ts - last_ts < TRADE_COOLDOWN_SECONDS:
                # igual calculamos y guardamos snapshot para debug
                pass
            if sym in self.active_trades:
                pass

        trend_1m_long, trend_1m_short = self._calc_trend_1m_flags(sym)

        feats, reason = self._build_hit_features(sym, bar)
        if feats is None:
            # snapshot con ceros para features + metas
            snapshot = {k: 0.0 for k in self.feature_names_ordered}
            snapshot.update({
                "trend_1m_long": float(trend_1m_long),
                "trend_1m_short": float(trend_1m_short),
                "_status": "not_ready",
                "_reason": reason or "no especificado",
                "p_long_raw": math.nan,
                "p_long": math.nan,
                "p_thr_long": float(P_TREND_LONG_THRESHOLD),
                "p_short_raw": math.nan,
                "p_short": math.nan,
                "p_thr_short": float(P_TREND_SHORT_THRESHOLD),
                "tp_k_atr5m": float(TP_K_ATR5M),
                "sl_m_atr5m": float(SL_M_ATR5M),
            })
            self._last_debug.setdefault(sym, {})["ml_features"] = snapshot
            return None

        # DataFrame EXACTO
        x_row = {f: float(feats[f]) for f in self.feature_names_ordered}
        if not all(self._is_finite(v) for v in x_row.values()):
            snapshot = dict(feats)
            snapshot.update({
                "trend_1m_long": float(trend_1m_long),
                "trend_1m_short": float(trend_1m_short),
                "_status": "not_ready",
                "_reason": "features no finitas",
                "p_long_raw": math.nan,
                "p_long": math.nan,
                "p_thr_long": float(P_TREND_LONG_THRESHOLD),
                "p_short_raw": math.nan,
                "p_short": math.nan,
                "p_thr_short": float(P_TREND_SHORT_THRESHOLD),
                "tp_k_atr5m": float(TP_K_ATR5M),
                "sl_m_atr5m": float(SL_M_ATR5M),
            })
            self._last_debug.setdefault(sym, {})["ml_features"] = snapshot
            return None

        x_df = pd.DataFrame([x_row], columns=self.feature_names_ordered)

        # ---------
        # CORRECCIÓN: p_long y p_short SIEMPRE
        # ---------
        p_long_raw = math.nan
        p_short_raw = math.nan
        p_long = math.nan
        p_short = math.nan
        errors = []

        if self.model_long is not None:
            try:
                p_long_raw = float(
                    self.model_long.predict_proba(x_df, num_iteration=self.best_iteration_long)[0][1]
                )
            except Exception as exc:
                logger.exception("Error predict_proba trend long", exc_info=exc)
                errors.append("predict_proba long")

        if self.model_short is not None:
            try:
                p_short_raw = float(
                    self.model_short.predict_proba(x_df, num_iteration=self.best_iteration_short)[0][1]
                )
            except Exception as exc:
                logger.exception("Error predict_proba trend short", exc_info=exc)
                errors.append("predict_proba short")

        p_long = p_long_raw
        p_short = p_short_raw

        status = "ok" if not errors else "warn"
        reason2 = ", ".join(errors) if errors else ""

        snapshot = dict(feats)
        snapshot.update({
            "trend_1m_long": float(trend_1m_long),
            "trend_1m_short": float(trend_1m_short),
            "_status": status,
            "_reason": reason2,
            "p_long_raw": float(p_long_raw) if math.isfinite(p_long_raw) else math.nan,
            "p_long": float(p_long) if math.isfinite(p_long) else math.nan,
            "p_thr_long": float(P_TREND_LONG_THRESHOLD),
            "p_short_raw": float(p_short_raw) if math.isfinite(p_short_raw) else math.nan,
            "p_short": float(p_short) if math.isfinite(p_short) else math.nan,
            "p_thr_short": float(P_TREND_SHORT_THRESHOLD),
            "tp_k_atr5m": float(TP_K_ATR5M),
            "sl_m_atr5m": float(SL_M_ATR5M),
        })
        self._last_debug.setdefault(sym, {})["ml_features"] = snapshot

        if is_warmup:
            return None

        # gating de operación
        # - solo LONG si ema_stack_5m_long==1
        # - solo SHORT si ema_stack_5m_short==1
        ema_stack_5m_long = float(feats.get("ema_stack_5m_long", 0.0))
        ema_stack_5m_short = float(feats.get("ema_stack_5m_short", 0.0))
        long_ready = (ema_stack_5m_long == 1.0) and math.isfinite(p_long) and (p_long >= P_TREND_LONG_THRESHOLD)
        short_ready = (ema_stack_5m_short == 1.0) and math.isfinite(p_short) and (p_short >= P_TREND_SHORT_THRESHOLD)

        # bloqueo por cooldown / trade activo SOLO para operar
        last_ts = self.last_signal_time.get(sym, 0)
        if now_ts - last_ts < TRADE_COOLDOWN_SECONDS:
            return None
        if sym in self.active_trades:
            return None

        if not long_ready and not short_ready:
            return None

        entry_price = float(bar.get("close", 0.0))
        atr5m = float(self.atr_values.get(sym, 0.0) or 0.0)
        if atr5m <= 0:
            return None

        # si ambos listos, elegir mayor prob
        if long_ready and (not short_ready or p_long >= p_short):
            tp_price = entry_price + TP_K_ATR5M * atr5m
            sl_price = entry_price - SL_M_ATR5M * atr5m
            side = "LONG"
        else:
            tp_price = entry_price - TP_K_ATR5M * atr5m
            sl_price = entry_price + SL_M_ATR5M * atr5m
            side = "SHORT"

        entry_time = bar.get("open_time_local")
        if isinstance(entry_time, datetime):
            entry_time = entry_time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "symbol": sym,
            "side": side,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "entry_time": entry_time,
            "atr5m": atr5m,
            "p_long": p_long,
            "p_short": p_short,
        }

    # ---------------------------
    # Monitor trade
    # ---------------------------
    async def _monitor_trade(self, sym):
        while sym in self.active_trades:
            await asyncio.sleep(ORDER_CHECK_INTERVAL)
            try:
                pos_info = await self.client.futures_position_information(symbol=sym)
                if isinstance(pos_info, list):
                    pos_entry = next((p for p in pos_info if float(p.get("positionAmt", 0)) != 0), None)
                else:
                    pos_entry = pos_info if float(pos_info.get("positionAmt", 0)) != 0 else None
                position_amt = float(pos_entry["positionAmt"]) if pos_entry else 0.0
            except Exception as e:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[ERROR monitor] No se pudo obtener posicion de {sym}:", e)
                continue

            if position_amt == 0:
                try:
                    await self.client._request_futures_api(
                        "delete", "algoOpenOrders", signed=True,
                        data={"symbol": sym, "timestamp": int(time.time() * 1000)},
                    )
                    print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                          f"[INFO] Algo orders canceladas para {sym}")
                except Exception as e:
                    logger.exception("Error al cancelar algo orders para %s", sym, exc_info=e)

                self.active_trades.pop(sym, None)
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[INFO] Trade cerrado para {sym}")
                break

    # ---------------------------
    # Tracking order tasks
    # ---------------------------
    def _track_order_task(self, task, sym, now_ts=None):
        if task is None:
            return
        self.pending_orders.add(task)

        def _done(t):
            self.pending_orders.discard(t)
            exc = t.exception()
            if exc:
                logger.exception("Error en tarea de orden para %s", sym, exc_info=exc)
            else:
                if now_ts is not None:
                    try:
                        if t.result():
                            self.last_signal_time[sym] = now_ts
                    except Exception as e:
                        logger.exception("Error al obtener resultado de tarea para %s", sym, exc_info=e)

        task.add_done_callback(_done)

    # ---------------------------
    # DEBUG (1m) - IMPRIME TODAS LAS FEATURES
    # ---------------------------
    def _debug_state(self, sym):
        now_str = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
        atr_5m = float(self.atr_values.get(sym, 0.0) or 0.0)
        atr_1m = float(self.atr_values_1m.get(sym, 0.0) or 0.0)

        close_last = None
        try:
            ch = self.closes_history.get(sym)
            if ch and len(ch) > 0:
                close_last = float(ch[-1])
        except Exception:
            close_last = None

        close_str = f"{float(close_last):.6f}" if close_last is not None else "None"
        print(now_str, f"[DEBUG_1m] {sym} Close1m:{close_str} ATR5m:{atr_5m:.6f} ATR1m:{atr_1m:.6f}")

        last = self._last_debug.get(sym, {})
        mf = last.get("ml_features") if isinstance(last, dict) else None
        if isinstance(mf, dict):
            rep = _format_ml_features_multiline(sym, mf, self.feature_names_ordered)
            for line in rep.splitlines():
                print(now_str, line)
        else:
            print(now_str, f"[ML-FEATURES] {sym} | sin snapshot")

    def _select_debug_symbols(self):
        if not self.symbols:
            self._debug_symbols = set()
            return
        sample_size = min(DEBUG_SYMBOLS_PER_MINUTE, len(self.symbols))
        self._debug_symbols = set(random.sample(self.symbols, sample_size))

    def _reset_summary_counts(self):
        self._summary_counts = {
            "seen": 0,
            "status_ok": 0,
            "status_not_ready": 0,
            "status_warn": 0,
            "ema_stack_long": 0,
            "ema_stack_short": 0,
            "ema_stack_mixed": 0,
        }

    def _count_stale_symbols(self, now_ms: int) -> int:
        stale = 0
        for sym in self.symbols:
            last_close = self.last_5m_close_time.get(sym)
            if not last_close:
                continue
            if now_ms - int(last_close) > 10 * 60 * 1000:
                stale += 1
        return stale

    def _print_summary(self, now_ms: int):
        if not self._summary_counts:
            return
        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=LOCAL_TZ)
        stale_count = self._count_stale_symbols(now_ms)
        summary = self._summary_counts
        line = (
            f"{now_dt.strftime('%H:%M:%S')} [SUMMARY_1m] "
            f"seen={summary['seen']} "
            f"status_ok={summary['status_ok']} status_not_ready={summary['status_not_ready']} "
            f"status_warn={summary['status_warn']} "
            f"stack_long={summary['ema_stack_long']} stack_short={summary['ema_stack_short']} "
            f"stack_mixed={summary['ema_stack_mixed']} "
            f"stale5m={stale_count}"
        )
        print(line)

    def _roll_minute(self, close_time_ms: int):
        minute = close_time_ms // 60000
        if self._summary_minute is None:
            self._summary_minute = minute
            self._debug_minute = minute
            self._reset_summary_counts()
            self._select_debug_symbols()
            return
        if minute != self._summary_minute:
            self._print_summary(close_time_ms)
            self._summary_minute = minute
            self._debug_minute = minute
            self._reset_summary_counts()
            self._select_debug_symbols()

    # ---------------------------
    # Cache filtros
    # ---------------------------
    async def _cache_symbol_filters(self):
        try:
            info = await self.client.futures_exchange_info()
        except Exception as e:
            print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                  "[ERROR] No se pudo obtener exchange_info:", e)
            return

        for s in info.get("symbols", []):
            try:
                filters = s.get("filters", [])
                lot = next(f for f in filters if f["filterType"] == "LOT_SIZE")
                price = next(f for f in filters if f["filterType"] == "PRICE_FILTER")
                min_notional_filter = next((f for f in filters if f["filterType"] == "MIN_NOTIONAL"), None)
                if min_notional_filter:
                    min_notional = float(min_notional_filter.get("minNotional") or min_notional_filter.get("notional") or 0)
                else:
                    min_notional = 0.0

                step_str = lot["stepSize"]
                tick_str = price["tickSize"]
                self.symbol_filters[s["symbol"]] = {
                    "step_size": float(step_str),
                    "min_qty": float(lot["minQty"]),
                    "precision_qty": count_decimal_places(step_str),
                    "tick_size": float(tick_str),
                    "precision_price": count_decimal_places(tick_str),
                    "min_notional": min_notional,
                }
            except Exception as e:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[ERROR] Procesando filtros de {s.get('symbol')}:", e)

        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              f"[INIT] Filtros cargados: {len(self.symbol_filters)} símbolos")

    # ---------------------------
    # Init símbolos + WARMUP REAL
    # ---------------------------
    async def _init_symbols(self):
        seen = set()
        ordered_symbols = []
        for sym in FIXED_SYMBOLS:
            if sym not in seen:
                ordered_symbols.append(sym)
                seen.add(sym)

        available_symbols = [sym for sym in ordered_symbols if sym in self.symbol_filters]
        missing_symbols = [sym for sym in ordered_symbols if sym not in self.symbol_filters]

        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              f"[INIT] Lista fija ({len(available_symbols)}/{len(ordered_symbols)})")
        if missing_symbols:
            print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                  "[INIT] Filtros faltantes para:", ", ".join(missing_symbols))

        self.symbols = list(available_symbols)

        for s in self.symbols:
            self.last_signal_time[s] = 0
            self.closes_history[s] = deque(maxlen=INIT_CANDLES)
            self.high_history[s] = deque(maxlen=INIT_CANDLES)
            self.low_history[s] = deque(maxlen=INIT_CANDLES)
            self.struct_history_1m[s].clear()
            self.atr5m_history[s].clear()
            self.atr5m_tr_buffer[s].clear()
            self.fanw_hist[s].clear()
            self.ema1m_hist[s]["ema9"].clear()
            self.ema1m_hist[s]["ema21"].clear()
            self.ema1m_hist[s]["ema50"].clear()
            self.ema1m_hist[s]["ema100"].clear()
            self.ema1m_hist[s]["ema200"].clear()
            self.close5m_hist[s].clear()
            self.ema5m_hist[s]["ema9"].clear()
            self.ema5m_hist[s]["ema21"].clear()
            self.ema5m_hist[s]["ema50"].clear()
            self.ema5m_hist[s]["ema100"].clear()
            self.ema5m_hist[s]["ema200"].clear()
            self.atr5m_causal_hist_1m[s].clear()
            self.last_5m_close_time[s] = None
            self._last_stale_warning_minute[s] = None
            self._stack5m_state[s] = None

        valid_symbols = []
        for sym in list(self.symbols):
            try:
                # 5m warmup
                klines5 = await self.client.futures_klines(symbol=sym, interval="5m", limit=INIT_CANDLES)
                if not klines5 or len(klines5) < max(EMA_SLOWEST + 5, ATR_PERIOD + 5):
                    raise RuntimeError(f"Warmup 5m insuficiente: {len(klines5) if klines5 else 0}")

                closes5 = [float(c[4]) for c in klines5]
                highs5 = [float(c[2]) for c in klines5]
                lows5 = [float(c[3]) for c in klines5]

                # EMA26 5m
                ema26_series = self._init_ema_from_prices(closes5, 26)
                ema26 = ema26_series[-1] if ema26_series else None
                self.ema26_5m[sym] = ema26

                # EMA 5m 9/21/50/100/200
                self.close5m_hist[sym].extend(closes5[-INIT_CANDLES:])
                ema9_5  = self._init_ema_from_prices(closes5, EMA_FAST)
                ema21_5 = self._init_ema_from_prices(closes5, EMA_MID)
                ema50_5 = self._init_ema_from_prices(closes5, EMA_SLOW)
                ema100_5 = self._init_ema_from_prices(closes5, EMA_SLOWER)
                ema200_5 = self._init_ema_from_prices(closes5, EMA_SLOWEST)
                for v in ema9_5[-INIT_CANDLES:]:
                    self.ema5m_hist[sym]["ema9"].append(float(v))
                for v in ema21_5[-INIT_CANDLES:]:
                    self.ema5m_hist[sym]["ema21"].append(float(v))
                for v in ema50_5[-INIT_CANDLES:]:
                    self.ema5m_hist[sym]["ema50"].append(float(v))
                for v in ema100_5[-INIT_CANDLES:]:
                    self.ema5m_hist[sym]["ema100"].append(float(v))
                for v in ema200_5[-INIT_CANDLES:]:
                    self.ema5m_hist[sym]["ema200"].append(float(v))

                # ATR Wilder 5m + history
                trs5 = []
                prev_close = None
                for i in range(len(closes5)):
                    cl = closes5[i]
                    hh = highs5[i]
                    ll = lows5[i]
                    if prev_close is None:
                        tr = hh - ll
                    else:
                        tr = max(hh - ll, abs(hh - prev_close), abs(ll - prev_close))
                    trs5.append(tr)
                    prev_close = cl

                self.atr5m_history[sym].clear()
                if len(trs5) >= ATR_PERIOD:
                    cur = sum(trs5[:ATR_PERIOD]) / ATR_PERIOD
                    self.atr5m_history[sym].append(float(cur))
                    for tr in trs5[ATR_PERIOD:]:
                        cur = wilder_atr_update(cur, tr, ATR_PERIOD)
                        self.atr5m_history[sym].append(float(cur))
                    atr5m_last = float(cur)
                else:
                    atr5m_last = float(sum(trs5)/len(trs5)) if trs5 else 0.0
                    self.atr5m_history[sym].append(atr5m_last)

                atr5m_series_5m = []
                if trs5:
                    cur_series = None
                    if len(trs5) >= ATR_PERIOD:
                        for idx, tr in enumerate(trs5):
                            if idx < ATR_PERIOD - 1:
                                cur_series = sum(trs5[:idx + 1]) / float(idx + 1)
                            elif idx == ATR_PERIOD - 1:
                                cur_series = sum(trs5[:ATR_PERIOD]) / ATR_PERIOD
                            else:
                                cur_series = wilder_atr_update(cur_series, tr, ATR_PERIOD)
                            atr5m_series_5m.append(float(cur_series))
                    else:
                        for idx in range(len(trs5)):
                            cur_series = sum(trs5[:idx + 1]) / float(idx + 1)
                            atr5m_series_5m.append(float(cur_series))

                self.atr_values[sym] = atr5m_last
                self.prev_close_5m[sym] = float(closes5[-1])
                if klines5 and len(klines5[-1]) > 6:
                    self.last_5m_close_time[sym] = int(klines5[-1][6])

                self.atr5m_tr_buffer[sym].clear()
                for tr in trs5[-ATR_PERIOD:]:
                    self.atr5m_tr_buffer[sym].append(float(tr))

                # 1m warmup
                klines1 = await self.client.futures_klines(symbol=sym, interval="1m", limit=INIT_CANDLES)
                min_1m = max(EMA_SLOWEST + 2 * max(CURV_LAGS) + 5, VOL_Z_WIN + 5, ATR_Z_WIN + 5, TAKER_ROLL_30 + 5)
                if not klines1 or len(klines1) < min_1m:
                    raise RuntimeError(f"Warmup 1m insuficiente: {len(klines1) if klines1 else 0}")

                self.closes_history[sym].clear()
                self.high_history[sym].clear()
                self.low_history[sym].clear()
                self.struct_history_1m[sym].clear()

                trs_1m = []
                prev_close1m = None
                closes1 = []

                for candle in klines1:
                    open_v = float(candle[1])
                    high_v = float(candle[2])
                    low_v  = float(candle[3])
                    close_v = float(candle[4])
                    vol_v = float(candle[5])
                    taker_buy = float(candle[9]) if len(candle) > 9 else 0.0
                    close_time = int(candle[6]) if len(candle) > 6 else 0

                    closes1.append(close_v)
                    self.closes_history[sym].append(close_v)
                    self.high_history[sym].append(high_v)
                    self.low_history[sym].append(low_v)

                    self.struct_history_1m[sym].append({
                        "open": open_v, "high": high_v, "low": low_v, "close": close_v,
                        "volume": vol_v, "taker_buy_base_volume": taker_buy,
                        "close_time": close_time,
                    })

                    if prev_close1m is None:
                        tr = high_v - low_v
                    else:
                        tr = max(high_v - low_v, abs(high_v - prev_close1m), abs(low_v - prev_close1m))
                    trs_1m.append(tr)
                    prev_close1m = close_v

                # ATR1m
                if len(trs_1m) >= ATR_PERIOD:
                    cur1 = sum(trs_1m[:ATR_PERIOD]) / ATR_PERIOD
                    for tr in trs_1m[ATR_PERIOD:]:
                        cur1 = wilder_atr_update(cur1, tr, ATR_PERIOD)
                    atr1m_last = float(cur1)
                else:
                    atr1m_last = float(sum(trs_1m)/len(trs_1m)) if trs_1m else 0.0

                self.atr_values_1m[sym] = atr1m_last
                self.prev_close_1m[sym] = float(prev_close1m) if prev_close1m is not None else None

                # EMA 1m 9/21/50/100/200 init
                ema9_series  = self._init_ema_from_prices(closes1, EMA_FAST)
                ema21_series = self._init_ema_from_prices(closes1, EMA_MID)
                ema50_series = self._init_ema_from_prices(closes1, EMA_SLOW)
                ema100_series = self._init_ema_from_prices(closes1, EMA_SLOWER)
                ema200_series = self._init_ema_from_prices(closes1, EMA_SLOWEST)

                for v in ema9_series[-INIT_CANDLES:]:
                    self.ema1m_hist[sym]["ema9"].append(float(v))
                for v in ema21_series[-INIT_CANDLES:]:
                    self.ema1m_hist[sym]["ema21"].append(float(v))
                for v in ema50_series[-INIT_CANDLES:]:
                    self.ema1m_hist[sym]["ema50"].append(float(v))
                for v in ema100_series[-INIT_CANDLES:]:
                    self.ema1m_hist[sym]["ema100"].append(float(v))
                for v in ema200_series[-INIT_CANDLES:]:
                    self.ema1m_hist[sym]["ema200"].append(float(v))

                # Prefill fanw_hist (para fan_expansion)
                self.fanw_hist[sym].clear()
                atr5m_now = float(self.atr_values.get(sym, 0.0) or 0.0)
                if atr5m_now > 0 and self._is_finite(atr5m_now):
                    start = max(0, len(closes1) - 200)
                    for i in range(start, len(closes1)):
                        e9  = float(ema9_series[i])
                        e21 = float(ema21_series[i])
                        e50 = float(ema50_series[i])
                        e100 = float(ema100_series[i])
                        e200 = float(ema200_series[i])
                        fw = (
                            abs(e9 - e21)
                            + abs(e21 - e50)
                            + abs(e50 - e100)
                            + abs(e100 - e200)
                        )
                        fwa = fw / atr5m_now
                        if self._is_finite(fwa):
                            self.fanw_hist[sym].append(float(fwa))

                # Prefill atr5m causal per 1m usando histórico ATR5m alineado al tiempo
                self.atr5m_causal_hist_1m[sym].clear()
                atr5m_by_open_time = {}
                for candle5, atr_val in zip(klines5, atr5m_series_5m):
                    if candle5 and len(candle5) > 0:
                        atr5m_by_open_time[int(candle5[0])] = float(atr_val)

                earliest_atr = atr5m_series_5m[0] if atr5m_series_5m else atr5m_last
                last_atr = None
                for candle1 in klines1:
                    open_time_1m = int(candle1[0])
                    open_time_5m = open_time_1m - (open_time_1m % (5 * 60 * 1000))
                    atr_val = atr5m_by_open_time.get(open_time_5m)
                    if atr_val is None:
                        atr_val = last_atr if last_atr is not None else earliest_atr
                    last_atr = atr_val
                    self.atr5m_causal_hist_1m[sym].append(float(atr_val))

                # warmup: calcular snapshot una vez (para ver todo OK)
                last_bar = self.struct_history_1m[sym][-1]
                bar_payload = dict(last_bar)
                bar_payload["open_time_local"] = datetime.now(LOCAL_TZ)
                bar_payload["is_closed"] = True
                _ = self.get_ml_signal_for_latest_bar(sym, bar_payload, is_warmup=True)

                valid_symbols.append(sym)
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[WARMUP OK] {sym} 1m={len(klines1)} 5m={len(klines5)} ATR5m={atr5m_last:.6f}")

            except Exception as e:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      f"[ERROR] Al precargar datos de {sym}:", e)

        self.symbols = valid_symbols
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "– Pares seleccionados:", self.symbols)

    # ---------------------------
    # Handle 1m (LIVE)
    # ---------------------------
    async def _handle_kline_1m(self, sym, bar):
        if bar is None or not bool(bar.get("is_closed", False)):
            return

        close_price = float(bar.get("close", 0))
        open_val = float(bar.get("open", 0))
        high_val = float(bar.get("high", 0))
        low_val = float(bar.get("low", 0))
        volume = float(bar.get("volume", 0))
        taker_buy_volume = float(bar.get("taker_buy_base_volume", 0))
        close_time = int(bar.get("close_time") or 0)
        open_time_local = bar.get("open_time_local")
        if close_time:
            self._roll_minute(close_time)

        self.closes_history.setdefault(sym, deque(maxlen=INIT_CANDLES)).append(close_price)
        self.high_history.setdefault(sym, deque(maxlen=INIT_CANDLES)).append(high_val)
        self.low_history.setdefault(sym, deque(maxlen=INIT_CANDLES)).append(low_val)

        self.struct_history_1m[sym].append({
            "open": open_val, "high": high_val, "low": low_val, "close": close_price,
            "volume": volume, "taker_buy_base_volume": taker_buy_volume,
            "close_time": close_time,
        })

        # EMA 1m update
        prev9 = self.ema1m_hist[sym]["ema9"][-1] if self.ema1m_hist[sym]["ema9"] else None
        prev21 = self.ema1m_hist[sym]["ema21"][-1] if self.ema1m_hist[sym]["ema21"] else None
        prev50 = self.ema1m_hist[sym]["ema50"][-1] if self.ema1m_hist[sym]["ema50"] else None
        prev100 = self.ema1m_hist[sym]["ema100"][-1] if self.ema1m_hist[sym]["ema100"] else None
        prev200 = self.ema1m_hist[sym]["ema200"][-1] if self.ema1m_hist[sym]["ema200"] else None
        self.ema1m_hist[sym]["ema9"].append(self._ema_next(prev9, close_price, EMA_FAST))
        self.ema1m_hist[sym]["ema21"].append(self._ema_next(prev21, close_price, EMA_MID))
        self.ema1m_hist[sym]["ema50"].append(self._ema_next(prev50, close_price, EMA_SLOW))
        self.ema1m_hist[sym]["ema100"].append(self._ema_next(prev100, close_price, EMA_SLOWER))
        self.ema1m_hist[sym]["ema200"].append(self._ema_next(prev200, close_price, EMA_SLOWEST))

        # ATR1m update
        prev_close1m = self.prev_close_1m.get(sym)
        if prev_close1m is None:
            tr = high_val - low_val
        else:
            tr = max(high_val - low_val, abs(high_val - prev_close1m), abs(low_val - prev_close1m))

        prev_atr1m = self.atr_values_1m.get(sym)
        self.atr_values_1m[sym] = wilder_atr_update(prev_atr1m, tr, ATR_PERIOD) if prev_atr1m else float(tr)
        self.prev_close_1m[sym] = close_price

        atr5m_causal = float(self.atr_values.get(sym, 0.0) or 0.0)
        if self._is_finite(atr5m_causal) and atr5m_causal > 0:
            self.atr5m_causal_hist_1m[sym].append(float(atr5m_causal))

        bar_payload = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_price,
            "volume": volume,
            "taker_buy_base_volume": taker_buy_volume,
            "close_time": close_time,
            "open_time_local": open_time_local,
            "is_closed": True,
        }

        # siempre calcula y guarda snapshot con p_long y p_short
        signal = self.get_ml_signal_for_latest_bar(sym, bar_payload, is_warmup=False)

        # resumen por símbolo
        if self._summary_counts is not None:
            self._summary_counts["seen"] += 1
            last_debug = self._last_debug.get(sym, {}) if isinstance(self._last_debug, dict) else {}
            ml_features = last_debug.get("ml_features") if isinstance(last_debug, dict) else None
            status = ml_features.get("_status") if isinstance(ml_features, dict) else None
            if status == "ok":
                self._summary_counts["status_ok"] += 1
            elif status == "warn":
                self._summary_counts["status_warn"] += 1
            else:
                self._summary_counts["status_not_ready"] += 1

            ema_stack_long = 0
            ema_stack_short = 0
            if isinstance(ml_features, dict):
                ema_stack_long = 1 if float(ml_features.get("ema_stack_5m_long", 0.0)) == 1.0 else 0
                ema_stack_short = 1 if float(ml_features.get("ema_stack_5m_short", 0.0)) == 1.0 else 0
            if ema_stack_long:
                self._summary_counts["ema_stack_long"] += 1
            elif ema_stack_short:
                self._summary_counts["ema_stack_short"] += 1
            else:
                self._summary_counts["ema_stack_mixed"] += 1

        # warnings de stale 5m
        last_5m_close = self.last_5m_close_time.get(sym)
        if last_5m_close:
            age_ms = close_time - int(last_5m_close)
            if age_ms > 10 * 60 * 1000:
                minute_key = close_time // 60000
                last_warn_minute = self._last_stale_warning_minute.get(sym)
                if last_warn_minute != minute_key:
                    self._last_stale_warning_minute[sym] = minute_key
                    age_min = age_ms / 60000.0
                    now_str = datetime.fromtimestamp(close_time / 1000, tz=LOCAL_TZ).strftime("%H:%M:%S")
                    print(now_str, f"[WARN] 5m stale {sym} age_min={age_min:.1f}")

        # DEBUG (imprime todo solo para algunos)
        if sym in self._debug_symbols:
            self._debug_state(sym)

        if signal and signal.get("side", "").upper() in ("LONG", "SHORT") and signal.get("atr5m", 0) > 0:
            now_ts = time.time()
            atr_1m = self.atr_values_1m.get(sym, 0.0)
            signal_side = signal["side"].lower()
            task = asyncio.create_task(
                place_futures_order_async(
                    self.client,
                    self,
                    sym,
                    signal_side,
                    signal["entry_price"],
                    signal["sl_price"],
                    signal["tp_price"],
                    atr1m=atr_1m,
                )
            )
            self._track_order_task(task, sym, now_ts)

    # ---------------------------
    # Handle 5m (LIVE)
    # ---------------------------
    async def _handle_kline_5m(self, sym, bar):
        if bar is None or not bool(bar.get("is_closed", False)):
            return

        close_5m = float(bar.get("close", 0))
        high_5m = float(bar.get("high", 0))
        low_5m = float(bar.get("low", 0))
        close_time = int(bar.get("close_time") or 0)
        if close_time:
            self.last_5m_close_time[sym] = close_time

        # EMA26 5m
        prev26 = self.ema26_5m.get(sym)
        ema26_new = self._ema_next(prev26, close_5m, 26)
        self.ema26_5m[sym] = ema26_new

        # EMA 5m 9/21/50/100/200 update
        self.close5m_hist[sym].append(close_5m)

        p9 = self.ema5m_hist[sym]["ema9"][-1] if self.ema5m_hist[sym]["ema9"] else None
        p21 = self.ema5m_hist[sym]["ema21"][-1] if self.ema5m_hist[sym]["ema21"] else None
        p50 = self.ema5m_hist[sym]["ema50"][-1] if self.ema5m_hist[sym]["ema50"] else None
        p100 = self.ema5m_hist[sym]["ema100"][-1] if self.ema5m_hist[sym]["ema100"] else None
        p200 = self.ema5m_hist[sym]["ema200"][-1] if self.ema5m_hist[sym]["ema200"] else None

        self.ema5m_hist[sym]["ema9"].append(self._ema_next(p9, close_5m, EMA_FAST))
        self.ema5m_hist[sym]["ema21"].append(self._ema_next(p21, close_5m, EMA_MID))
        self.ema5m_hist[sym]["ema50"].append(self._ema_next(p50, close_5m, EMA_SLOW))
        self.ema5m_hist[sym]["ema100"].append(self._ema_next(p100, close_5m, EMA_SLOWER))
        self.ema5m_hist[sym]["ema200"].append(self._ema_next(p200, close_5m, EMA_SLOWEST))

        # TR
        prev_close = self.prev_close_5m.get(sym)
        if prev_close is None:
            tr = high_5m - low_5m
        else:
            tr = max(high_5m - low_5m, abs(high_5m - prev_close), abs(low_5m - prev_close))

        buf = self.atr5m_tr_buffer[sym]
        buf.append(tr)

        prev_atr = self.atr_values.get(sym)
        if prev_atr is None or prev_atr <= 0:
            if len(buf) >= ATR_PERIOD:
                atr = sum(list(buf)[-ATR_PERIOD:]) / ATR_PERIOD
            else:
                atr = sum(buf) / len(buf)
        else:
            atr = wilder_atr_update(prev_atr, tr, ATR_PERIOD)

        self.atr_values[sym] = float(atr)
        self.prev_close_5m[sym] = close_5m

        self.atr5m_history[sym].append(float(atr))

        ema9_5m = float(self.ema5m_hist[sym]["ema9"][-1]) if self.ema5m_hist[sym]["ema9"] else math.nan
        ema21_5m = float(self.ema5m_hist[sym]["ema21"][-1]) if self.ema5m_hist[sym]["ema21"] else math.nan
        ema50_5m = float(self.ema5m_hist[sym]["ema50"][-1]) if self.ema5m_hist[sym]["ema50"] else math.nan
        ema100_5m = float(self.ema5m_hist[sym]["ema100"][-1]) if self.ema5m_hist[sym]["ema100"] else math.nan
        ema200_5m = float(self.ema5m_hist[sym]["ema200"][-1]) if self.ema5m_hist[sym]["ema200"] else math.nan

        if ema9_5m > ema21_5m > ema50_5m > ema100_5m > ema200_5m:
            stack_state = "LONG"
        elif ema9_5m < ema21_5m < ema50_5m < ema100_5m < ema200_5m:
            stack_state = "SHORT"
        else:
            stack_state = "MIXED"

        if self._stack5m_state.get(sym) != stack_state:
            self._stack5m_state[sym] = stack_state
            now_str = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
            print(
                now_str,
                "[STACK5M]",
                sym,
                stack_state,
                f"ema9={_fmt6(ema9_5m)}",
                f"ema21={_fmt6(ema21_5m)}",
                f"ema50={_fmt6(ema50_5m)}",
                f"ema100={_fmt6(ema100_5m)}",
                f"ema200={_fmt6(ema200_5m)}",
            )

    # ---------------------------
    # Periodic refilter
    # ---------------------------
    async def _periodic_refilter(self):
        while True:
            await asyncio.sleep(REFILTER_INTERVAL)
            ts_str = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
            print(ts_str, "– Refiltrando pares... actuales:", self.symbols)
            await self._init_symbols()
            if self.ws:
                await self.ws.close()

    # ---------------------------
    # Dashboard opcional
    # ---------------------------
    async def _dashboard_loop(self):
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print("Active Trades")
            print("-" * 60)
            for sym, t in self.active_trades.items():
                print(f"{sym:<10} TPAlgo: {t.get('tp_algo_id')} StopAlgo: {t.get('stop_algo_id')}")
            print("\nPress Ctrl+C to exit")
            await asyncio.sleep(1)

    # ---------------------------
    # Parse klines
    # ---------------------------
    def _parse_ws_kline(self, kline):
        if not kline:
            return None
        open_time = int(kline.get("t", 0))
        close_time = int(kline.get("T", 0))
        open_time_local = datetime.fromtimestamp(open_time / 1000, tz=LOCAL_TZ) if open_time else None
        return {
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", 0)),
            "taker_buy_base_volume": float(kline.get("V", 0)),
            "open_time": open_time,
            "close_time": close_time,
            "open_time_local": open_time_local,
            "is_closed": bool(kline.get("x", False)),
        }

    # ---------------------------
    # Reader WS
    # ---------------------------
    async def _reader(self, ws):
        async for msg_str in ws:
            msg = json.loads(msg_str)
            stream = msg.get("stream", "")
            data = msg.get("data", {})
            sym = stream.split("@")[0].upper()

            if "@kline_1m" in stream:
                bar = self._parse_ws_kline(data.get("k", {}))
                if bar and bar.get("is_closed"):
                    await self._handle_kline_1m(sym, bar)

            elif "@kline_5m" in stream:
                bar5 = self._parse_ws_kline(data.get("k", {}))
                if bar5 and bar5.get("is_closed"):
                    await self._handle_kline_5m(sym, bar5)

    # ---------------------------
    # Start
    # ---------------------------
    async def start(self):
        if not self.api_key or not self.api_secret:
            print("[ERROR] BINANCE_API_KEY / BINANCE_API_SECRET no configurados.")
            print("Setealos como variables de entorno (recomendado).")
            return

        if self.model_long is None or self.model_short is None:
            print("[ERROR] Modelos no cargados correctamente. El bot NO operará.")
            return

        self.client = await AsyncClient.create(self.api_key, self.api_secret)

        await self._cache_symbol_filters()

        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              f"[WARMUP] Precargando {INIT_CANDLES} velas 1m y 5m por símbolo...")
        await self._init_symbols()

        asyncio.create_task(self._periodic_refilter())
        asyncio.create_task(_daily_signal_file_task())

        if self.enable_dashboard:
            self.dashboard_task = asyncio.create_task(self._dashboard_loop())

        while True:
            streams = []
            for s in self.symbols:
                sl = s.lower()
                streams.append(f"{sl}@kline_1m")
                streams.append(f"{sl}@kline_5m")
            uri = "wss://fstream.binance.com/stream?streams=" + "/".join(streams)
            print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                  "[WS] Conectando:", uri)

            try:
                async with websockets.connect(uri, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT) as ws:
                    self.ws = ws
                    await self._reader(ws)

            except (asyncio.CancelledError, websockets.exceptions.ConnectionClosedError) as e:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      "[INFO] WS cerrado; reintentando en 5s. Error:", e)
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n", datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      "[INFO] Detenido por el usuario (Ctrl+C).")
                break
            except Exception as e:
                print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                      "[ERROR] WS u otro error:", e)
                await asyncio.sleep(5)

        try:
            await self.client.close_connection()
        except Exception as e:
            print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
                  "[INFO] Error al cerrar conexión:", e)
        print(datetime.now(LOCAL_TZ).strftime("%H:%M:%S"),
              "[INFO] Bot finalizado.")

        if self.dashboard_task:
            self.dashboard_task.cancel()
            try:
                await self.dashboard_task
            except asyncio.CancelledError:
                pass

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    enable_dash = os.getenv("DASHBOARD") == "1" or "--dashboard" in sys.argv
    try:
        bot = TrendBot(API_KEY, API_SECRET, enable_dashboard=enable_dash)
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        pass
