import customtkinter as ctk
from tkinter import messagebox, TclError
import requests
import json
import time
import datetime
import hmac
import hashlib
import base64
import traceback
import threading
import queue
from decimal import Decimal, ROUND_DOWN, getcontext, InvalidOperation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class TranslationManager:
    def __init__(self, locale_dir="locales", default_lang="en"):
        self.locale_dir = locale_dir
        self.languages = {}
        self.current_lang_data = {}
        self._load_languages()
        self.set_language(default_lang)

    def _load_languages(self):
        """加载所有可用的语言文件"""
        for filename in os.listdir(self.locale_dir):
            if filename.endswith(".json"):
                lang_code = filename.split('.')[0]
                try:
                    with open(os.path.join(self.locale_dir, filename), 'r', encoding='utf-8-sig') as f:
                        self.languages[lang_code] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Failed to load language file {filename}: {e}")

    def set_language(self, lang_code):
        """设置当前语言"""
        if lang_code in self.languages:
            self.current_lang_data = self.languages[lang_code]
            print(f"Language set to: {lang_code}")
        else:
            print(f"Warning: Language '{lang_code}' not found. Falling back to default.")
            # Fallback to the first available language if the chosen one is not found
            if self.languages:
                first_lang = list(self.languages.keys())[0]
                self.current_lang_data = self.languages[first_lang]


    def get(self, key, **kwargs):
        """获取翻译文本，支持格式化"""
        text = self.current_lang_data.get(key, key) # 如果找不到键，返回键本身以便调试
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError as e:
                print(f"Formatting error for key '{key}': missing placeholder {e}")
                return text
        return text

    def get_available_languages(self):
        """返回所有可用语言代码的列表"""
        return list(self.languages.keys())

getcontext().prec = 28

# ==============================================================================
# --- 1. CONFIGURATION & GLOBALS ---
# ==============================================================================
API_CREDENTIALS = {
    "live": {
        "trading": {
            "api_key": os.getenv("OKX_LIVE_TRADING_API_KEY"),
            "api_secret": os.getenv("OKX_LIVE_TRADING_API_SECRET"),
            "password": os.getenv("OKX_LIVE_TRADING_PASSWORD")
        },
        "monitoring": {
            "api_key": os.getenv("OKX_LIVE_MONITORING_API_KEY"),
            "api_secret": os.getenv("OKX_LIVE_MONITORING_API_SECRET"),
            "password": os.getenv("OKX_LIVE_MONITORING_PASSWORD")
        }
    },
    "demo": {
        "trading": {
            "api_key": os.getenv("OKX_DEMO_TRADING_API_KEY"),
            "api_secret": os.getenv("OKX_DEMO_TRADING_API_SECRET"),
            "password": os.getenv("OKX_DEMO_TRADING_PASSWORD")
        },
        "monitoring": {
            "api_key": os.getenv("OKX_DEMO_MONITORING_API_KEY"),
            "api_secret": os.getenv("OKX_DEMO_MONITORING_API_SECRET"),
            "password": os.getenv("OKX_DEMO_MONITORING_PASSWORD")
        }
    }
}
PROXIES = {
    'http': 'http://127.0.0.1:7897',
    'https':'http://127.0.0.1:7897'
}
#SOMETIMES YOU NEED TO CHANGE IT(EXAMPLE)
BASE_URL = 'https://www.okx.com'; REQUEST_TIMEOUT = 5; LOG_FILE = "trading_log.csv"; CONFIG_FILE = "config.json"
IS_DEMO_MODE = True

config_lock = threading.Lock()
state_lock = threading.Lock()
NEW_ORDER_GRACE_PERIOD_SECONDS = 15
RATE_LIMIT_ERROR_CODES = ['50011']; TRANSIENT_ERROR_CODES = ['50000']
PARTIAL_SUCCESS_PATHS = ['/api/v5/trade/cancel-batch-orders', '/api/v5/trade/batch-orders']
IGNORABLE_FAILURE_PATHS = {'/api/v5/trade/cancel-batch-orders': ['1'], '/api/v5/trade/cancel-order': ['1']}
RATE_LIMIT_WAIT_SECONDS = 7
REGRID_ATR_FACTOR = Decimal('1.5')
CHASE_SPREAD_FACTOR = Decimal('0.1')
STOP_LOSS_RETRY_TIMEOUT_SECONDS = 15; STOP_LOSS_RETRY_ATTEMPTS = 3; STOP_LOSS_ORDER_CHECK_DELAY = 2
CRITICAL_ERROR_WAIT_SECONDS = 10
STOP_LOSS_COOLDOWN_MINUTES = 15
SCANNER_MIN_VOLUME_USDT = 1000000; SMALL_CAPITAL_THRESHOLD = 100; MAX_SPREAD_PCT_FILTER = 0.5
MICRO_CAPITAL_THRESHOLD = 15; MICRO_CAPITAL_MIN_TPM = 10
PIN_BAR_WICK_TO_BODY_RATIO = 3; GRID_PAIR_VOLATILITY_FACTOR = 150
REGRID_ATR_FACTOR = Decimal('1.5')
MIN_ORDER_VALUE_USDT = Decimal('1.0'); MAKER_FEE = Decimal('0.0008'); TAKER_FEE = Decimal('0.001')
SPREAD_BUFFER_PCT = Decimal('0.05'); VOLATILITY_SCORE_WEIGHT = Decimal('0.6'); LIQUIDITY_SCORE_WEIGHT = Decimal('0.1'); ACTIVITY_SCORE_WEIGHT = Decimal('0.1')
ATR_SCORE_WEIGHT = Decimal('0.2')
PIN_BAR_WICK_TO_BODY_RATIO = 4
STOP_LOSS_ATR_MULTIPLIER = Decimal('2.5'); GUARDIAN_POLL_FREQUENCY_SECONDS = 1.5
DEFAULT_POLL_FREQUENCY = 4; DEFAULT_SYMBOL = 'LTC-USDT'; DEFAULT_ORDER_SIZE = Decimal('100.0'); DEFAULT_GRID_PAIRS = 5; DEFAULT_SPREAD = Decimal('0.005'); DEFAULT_STEP = Decimal('0.002')
TREND_FILTER_EMA_PERIOD = 100 
dynamic_config = {
    'POLL_FREQUENCY_SECONDS': DEFAULT_POLL_FREQUENCY, 'SYMBOL_API_FORMAT': DEFAULT_SYMBOL,
    'SYMBOL_DISPLAY': DEFAULT_SYMBOL.replace('-', '/'), 'BASE_CURRENCY': DEFAULT_SYMBOL.split('-')[0], 'QUOTE_CURRENCY': DEFAULT_SYMBOL.split('-')[1],
    'ORDER_SIZE_USDT': DEFAULT_ORDER_SIZE, 'GRID_PAIRS': DEFAULT_GRID_PAIRS, 'SPREAD_PERCENTAGE': DEFAULT_SPREAD, 'GRID_STEP_PERCENTAGE': DEFAULT_STEP,
    'atr_based_stop_loss_usdt': Decimal('0'), 'entry_price': Decimal('0'), 'stop_loss_price': Decimal('0')
}
# ==============================================================================
# --- 2. UTILITY & ROBUST API REQUESTS (Restored Readability & FIXED) ---
# ==============================================================================
class NetworkError(Exception): pass
def print_to_queue(q, msg_type, data):
    if q:
        try: q.put_nowait({'type': msg_type, 'data': data})
        except queue.Full: pass
def get_timestamp(): return datetime.datetime.now(datetime.UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
def create_signature(timestamp, method, request_path, body_str, secret_key):
    prehash_string = f"{timestamp}{method}{request_path}{body_str or ''}"
    secret_key_bytes = secret_key.encode('utf-8'); prehash_bytes = prehash_string.encode('utf-8')
    hmac_digest = hmac.new(secret_key_bytes, prehash_bytes, hashlib.sha256).digest()
    return base64.b64encode(hmac_digest).decode('utf-8')
def okx_request(method, request_path, body_str="", max_retries=3, key_type="trading"):
    mode = "demo" if IS_DEMO_MODE else "live"
    creds = API_CREDENTIALS[mode][key_type]
    if "YOUR_" in creds.get('api_key', 'YOUR_KEY'):
        raise ValueError(f"API credentials for {mode.upper()}/{key_type.upper()} key are not configured.")
    url = f"{BASE_URL}{request_path}"
    for attempt in range(max_retries):
        try:
            timestamp = get_timestamp()
            signature = create_signature(timestamp, method, request_path, body_str, creds["api_secret"])
            headers = {'Content-Type': 'application/json', 'OK-ACCESS-KEY': creds["api_key"], 'OK-ACCESS-SIGN': signature, 'OK-ACCESS-PASSPHRASE': creds["password"], 'OK-ACCESS-TIMESTAMP': timestamp,}
            if IS_DEMO_MODE: headers['x-simulated-trading'] = '1'
            response = requests.request(method, url, headers=headers, data=body_str, proxies=PROXIES, timeout=REQUEST_TIMEOUT)
            response.raise_for_status(); response_json = response.json(); api_code = response_json.get('code', '0')
            if api_code != '0':
                if request_path in PARTIAL_SUCCESS_PATHS and api_code == '2': return response_json
                if request_path in IGNORABLE_FAILURE_PATHS and api_code in IGNORABLE_FAILURE_PATHS[request_path]: return response_json
                if api_code in RATE_LIMIT_ERROR_CODES: print_to_queue(None, 'log', "Rate limit hit. Waiting..."); time.sleep(RATE_LIMIT_WAIT_SECONDS); continue
                if api_code in TRANSIENT_ERROR_CODES and attempt < max_retries - 1: time.sleep(1 * (2 ** attempt)); continue
                raise Exception(f"API Error: {response_json.get('msg')} (Code: {api_code})")
            data = response_json.get('data')
            if data and isinstance(data, list) and len(data) > 0 and 'sCode' in data[0] and data[0].get('sCode') != '0' and not (request_path in PARTIAL_SUCCESS_PATHS):
                 raise Exception(f"Order Error: {data[0].get('sMsg')} (sCode: {data[0].get('sCode')})")
            return response_json
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ProxyError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1: time.sleep(1 * (2 ** attempt)); continue
            raise NetworkError(f"Network error after {max_retries} retries: {e}") from e
    raise Exception("Request failed after all retries.")
def is_trend_down(symbol_api, log_queue, t):
    """
    使用EMA判断当前是否处于下跌趋势。
    返回 True 表示处于下跌趋势 (应暂停开仓)，False 表示可以交易。
    """
    try:
        limit = TREND_FILTER_EMA_PERIOD * 2
        kline_resp = okx_request('GET', f"/api/v5/market/history-candles?instId={symbol_api}&bar=15m&limit={limit}")
        kline_data = kline_resp.get('data')

        if not kline_data or len(kline_data) < TREND_FILTER_EMA_PERIOD:
            print_to_queue(log_queue, 'log', t("log_trend_data_insufficient"))
            return False

        closes = [float(k[4]) for k in kline_data]
        closes.reverse()
        
        df = pd.DataFrame(closes, columns=['close'])
        df['ema'] = df['close'].ewm(span=TREND_FILTER_EMA_PERIOD, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        current_ema = df['ema'].iloc[-1]

        if current_price < current_ema:
            print_to_queue(log_queue, 'log', t("log_trend_down", price=f"{current_price:.4f}", period=TREND_FILTER_EMA_PERIOD, ema=f"{current_ema:.4f}"))
            return True
        else:
            return False
            
    except Exception as e:
        print_to_queue(log_queue, 'log', t("log_trend_error", error=e))
        return False
# ==============================================================================
# --- 3. MARKET SCANNER & ANALYSIS (V22.0 "VOLATILITY HUNTER" I18N EDITION) ---
# ==============================================================================
class MarketScanner:
    def __init__(self, log_queue, translator_func):
        """
        初始化市场扫描器。
        :param log_queue: 用于发送日志消息的队列。
        :param translator_func: 国际化翻译函数 (t)。
        """
        self.log_queue = log_queue
        self.t = translator_func

    def log(self, message_key, **kwargs):
        """
        发送一条经过翻译的日志消息到队列。
        :param message_key: 语言文件中的键。
        :param kwargs: 用于格式化字符串的变量。
        """
        # 使用翻译函数生成最终的日志消息
        translated_message = self.t(message_key, **kwargs)
        # 加上固定的 [Scanner] 前缀
        print_to_queue(self.log_queue, 'log', f"[Scanner] {translated_message}")

    def _calculate_atr(self, kline_data):
        if not kline_data or len(kline_data) < 14: return 0.0
        df = pd.DataFrame(kline_data, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm']).apply(pd.to_numeric)
        df['h-l'] = df['h'] - df['l']
        df['h-pc'] = abs(df['h'] - df['c'].shift(1))
        df['l-pc'] = abs(df['l'] - df['c'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df['tr'].rolling(window=14).mean().iloc[-1]

    def _analyze_candidate(self, inst_id, tickers_map):
        try:
            # 1. Basic liquidity filter
            ticker = tickers_map.get(inst_id)
            if not ticker: return None
            volume_usdt = Decimal(ticker.get('volCcy24h', '0'))
            liquidity_score = Decimal(str(np.log10(float(volume_usdt + 1))))

            # 2. Fetch K-lines for deep volatility analysis
            kline_resp = okx_request('GET', f"/api/v5/market/history-candles?instId={inst_id}&bar=15m&limit=96")
            kline_data = kline_resp.get('data')
            if not kline_data: return None

            # 3. Volatility Analysis
            pin_count, total_pin_depth_pct = 0, Decimal('0.0')
            closes = []
            for k in kline_data:
                high, low, open_p, close_p = float(k[2]), float(k[3]), float(k[1]), float(k[4])
                closes.append(close_p)
                body_len = abs(open_p - close_p)
                if body_len > 1e-9:
                    total_wick = (high - max(open_p, close_p)) + (min(open_p, close_p) - low)
                    if (total_wick / body_len) > PIN_BAR_WICK_TO_BODY_RATIO:
                        pin_count += 1
                        total_pin_depth_pct += (Decimal(total_wick) / Decimal(close_p)) * 100

            avg_price = np.mean(closes) if closes else 1.0
            atr_value = self._calculate_atr(kline_data)
            atr_pct = (Decimal(atr_value) / Decimal(avg_price)) * 100 if avg_price > 0 else Decimal('0')
            volatility_score = Decimal(pin_count) + (total_pin_depth_pct * 2)
            atr_score = atr_pct * 10

            # 4. Activity Score
            trades_resp = okx_request('GET', f'/api/v5/market/trades?instId={inst_id}&limit=100')
            trades_data = trades_resp.get('data')
            tpm = 0
            if trades_data and len(trades_data) >= 2:
                 time_diff = (int(trades_data[0]['ts']) - int(trades_data[-1]['ts'])) / 1000
                 if time_diff > 0: tpm = (len(trades_data) / time_diff) * 60
            activity_score = Decimal(str(np.log10(tpm + 1)))

            # 5. Final Weighted Score
            final_score = (volatility_score * VOLATILITY_SCORE_WEIGHT) + (atr_score * ATR_SCORE_WEIGHT) + \
                          (liquidity_score * LIQUIDITY_SCORE_WEIGHT) + (activity_score * ACTIVITY_SCORE_WEIGHT)

            return {"instId": inst_id, "score": final_score, "tpm": tpm, "atr_value": Decimal(atr_value),
                    "atr_sl_usdt": Decimal(atr_value) * STOP_LOSS_ATR_MULTIPLIER,
                    "volatility_pct": atr_pct, "pin_count": pin_count}

        except Exception as e:
            # [i18n] 使用翻译键记录错误日志
            self.log("scan_error_analyzing_candidate", symbol=inst_id, error=e)
            return None

    def _get_recommended_params_for_candidate(self, analysis_result, initial_usdt):
        # [i18n] 使用翻译键记录日志
        self.log("scan_autoconfig_start", symbol=analysis_result['instId'])
        atr_pct = analysis_result['volatility_pct'] / 100
        new_spread = max(Decimal('0.0015'), atr_pct * Decimal('0.5'))
        new_step = new_spread * Decimal('0.4')
        tpm = analysis_result.get('tpm', 50)
        poll_freq = 4 if tpm < 50 else 3 if tpm < 200 else 2
        
        recommended_params = {
            "SYMBOL_API_FORMAT": analysis_result['instId'],
            "SPREAD_PERCENTAGE": new_spread, "GRID_STEP_PERCENTAGE": new_step,
            "POLL_FREQUENCY_SECONDS": poll_freq, "atr_based_stop_loss_usdt": analysis_result['atr_sl_usdt']
        }
        
        if initial_usdt > SMALL_CAPITAL_THRESHOLD:
            recommended_params['GRID_PAIRS'] = max(2, min(7, 7 - int(atr_pct * GRID_PAIR_VOLATILITY_FACTOR)))
        else:
            recommended_params['GRID_PAIRS'] = 1
        
        if initial_usdt <= MICRO_CAPITAL_THRESHOLD:
            recommended_params['ORDER_SIZE_USDT'] = initial_usdt
        
        # [i18n] 使用翻译键记录推荐参数详情
        self.log("scan_autoconfig_details_volatility", atr_pct=f"{atr_pct:.4%}", tpm=f"{analysis_result['tpm']:.1f}")
        self.log("scan_autoconfig_details_params", spread=f"{new_spread:.4%}", step=f"{new_step:.4%}")
        self.log("scan_autoconfig_details_grids", pairs=recommended_params.get('GRID_PAIRS', self.t("not_applicable")))
        
        return recommended_params
    
    def run_scan(self, initial_usdt: Decimal):
        # [i18n] 使用翻译键记录日志
        self.log("scan_start_banner", capital=initial_usdt)
        try:
            self.log("scan_fetching_market_data")
            tickers_map = {t['instId']: t for t in okx_request('GET', "/api/v5/market/tickers?instType=SPOT")['data']}
            insts = okx_request('GET', "/api/v5/public/instruments?instType=SPOT")['data']
            
            min_vol = 50000 if initial_usdt <= MICRO_CAPITAL_THRESHOLD else SCANNER_MIN_VOLUME_USDT
            candidates = [
                i['instId'] for i in insts if i['instId'] in tickers_map and i['quoteCcy'] == 'USDT' and i['state'] == 'live'
                and Decimal(tickers_map[i['instId']].get('volCcy24h', '0')) > min_vol
                and tickers_map[i['instId']].get('last') is not None and Decimal(tickers_map[i['instId']]['last']) > 0
                and initial_usdt >= Decimal(i['minSz']) * Decimal(tickers_map[i['instId']]['last'])
            ]

            self.log("scan_candidates_found", count=len(candidates))
            results = []
            for i, inst_id in enumerate(candidates):
                print_to_queue(self.log_queue, 'progress', (i + 1, len(candidates)))
                if res := self._analyze_candidate(inst_id, tickers_map):
                     results.append(res)
                     if res['score'] > 15:
                         self.log("scan_promising_candidate", symbol=f"{res['instId']:<12}", score=f"{res['score']:.1f}", 
                                  volatility=f"{res['volatility_pct']:.2f}%", pins=res['pin_count'])
                time.sleep(0.1)

            print_to_queue(self.log_queue, 'progress', None)
            if not results:
                self.log("scan_no_candidates_found"); return None
            
            best = max(results, key=lambda x: x['score'])
            
            # [i18n] 使用翻译键记录最终结果
            self.log("scan_final_result_symbol", symbol=best['instId'])
            self.log("scan_final_result_score", score=f"{best['score']:.1f}")
            self.log("scan_final_result_volatility", volatility=f"{best['volatility_pct']:.2f}%")
            self.log("scan_final_result_pins", pins=best['pin_count'])
            
            return self._get_recommended_params_for_candidate(best, initial_usdt)

        except Exception:
            # [i18n] 使用翻译键记录严重错误
            self.log("scan_critical_error", traceback=traceback.format_exc())
            return None
# ==============================================================================
# --- 4. BOT LOGIC & DATA PERSISTENCE (Restored Readability & FIXED) ---
# ==============================================================================
def get_state_file_name():
    with config_lock: symbol = dynamic_config.get('SYMBOL_API_FORMAT', DEFAULT_SYMBOL)
    return f"trading_state_{symbol}.json"
def save_state(grid_orders, profit_orders, total_position_size, total_position_cost):
    state_file = get_state_file_name()
    try:
        with state_lock:
            grid_orders_copy, profit_orders_copy = grid_orders.copy(), profit_orders.copy()
        with config_lock:
            entry_price_str = f"{dynamic_config.get('entry_price', Decimal('0'))}"
            stop_loss_price_str = f"{dynamic_config.get('stop_loss_price', Decimal('0'))}"
            amount_prec = dynamic_config.get('amount_precision', 8)
            size_str = f"{total_position_size:.{amount_prec}f}"
        state_data = {'grid_orders': grid_orders_copy, 'profit_orders': list(profit_orders_copy), 'entry_price': entry_price_str, 'stop_loss_price': stop_loss_price_str, 'total_position_size': size_str, 'total_position_cost': str(total_position_cost)}
        temp_file = f"{state_file}.tmp"
        with open(temp_file, 'w') as f: json.dump(state_data, f, indent=4)
        os.replace(temp_file, state_file)
    except Exception as e: print(f"Error saving state: {e}")
def load_state():
    state_file = get_state_file_name()
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f: state_data = json.load(f)
            return (state_data.get('grid_orders', {}), set(state_data.get('profit_orders', [])), Decimal(state_data.get('entry_price', '0')), Decimal(state_data.get('stop_loss_price', '0')), Decimal(state_data.get('total_position_size', '0')), Decimal(state_data.get('total_position_cost', '0')))
        except (json.JSONDecodeError, IOError, InvalidOperation) as e: print(f"Error loading {state_file}, starting fresh: {e}")
    return {}, set(), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
def fetch_balances_and_price(log_queue, t):
    try:
        balance_data = okx_request('GET', "/api/v5/account/balance")['data'][0]
        total_value_usdt = Decimal(balance_data.get('totalEq', '0'))
        base_bal, quote_bal, avail_base_bal, avail_quote_bal = (Decimal('0'),) * 4
        with config_lock: base_ccy, quote_ccy = dynamic_config['BASE_CURRENCY'], dynamic_config['QUOTE_CURRENCY']
        for item in balance_data.get('details', []):
            if item['ccy'] == base_ccy: 
                base_bal, avail_base_bal = Decimal(item.get('eq', '0')), Decimal(item.get('availEq', '0'))
            elif item['ccy'] == quote_ccy: 
                quote_bal, avail_quote_bal = Decimal(item.get('eq', '0')), Decimal(item.get('availEq', '0'))
        
        with config_lock: symbol_api = dynamic_config["SYMBOL_API_FORMAT"]
        current_price = Decimal(okx_request('GET', f'/api/v5/market/ticker?instId={symbol_api}')['data'][0]['last'])
        
        return {"base_bal": base_bal, "quote_bal": quote_bal, "avail_base_bal": avail_base_bal, 
                "avail_quote_bal": avail_quote_bal, "current_price": current_price, "total_value_usdt": total_value_usdt}
    except Exception as e: 
        # [i18n] 使用翻译函数记录日志
        print_to_queue(log_queue, 'log', t("log_fetch_balance_error", error=e))
        return None
def guardian_loop(log_queue, stop_event, status_queue, t): # <-- 使用简短的 't' 作为翻译函数名
    # 使用 't' 来生成翻译后的日志
    print_to_queue(log_queue, 'log', t("log_guardian_started"))
    while not stop_event.is_set():
        try:
            with config_lock:
                entry_price, stop_loss_price, symbol_api = dynamic_config.get('entry_price',0), dynamic_config.get('stop_loss_price',0), dynamic_config["SYMBOL_API_FORMAT"]
            if entry_price > 0 and stop_loss_price > 0:
                if (ticker_data := okx_request('GET', f'/api/v5/market/ticker?instId={symbol_api}', key_type="monitoring")) and (current_price := Decimal(ticker_data['data'][0]['last'])) <= stop_loss_price:
                    status_queue.put("STOP_LOSS_TRIGGERED")
                    print_to_queue(log_queue, 'log', t("log_guardian_sl_triggered", price=current_price, sl_price=stop_loss_price))
                    time.sleep(15)
            time.sleep(GUARDIAN_POLL_FREQUENCY_SECONDS)
        except Exception as e:
            print_to_queue(log_queue, 'log', t("log_guardian_error", error=e))
            time.sleep(5)
    print_to_queue(log_queue, 'log', t("log_guardian_stopped"))
def execute_smart_stop_loss(log_queue, min_sz, amount_precision, price_precision, lot_sz, t):
    """
    [修复4 - 紧急止损]
    1. 取消该交易对的所有挂单。
    2. 立即通过市价单卖出所有可用余额。
    """
    print_to_queue(log_queue, 'log', t("log_sl_received"))
    with config_lock: cfg = dynamic_config.copy()
    symbol_api = cfg['SYMBOL_API_FORMAT']

    try:
        if all_pending := okx_request('GET', f"/api/v5/trade/orders-pending?instType=SPOT&instId={symbol_api}"):
            if cancel_reqs := [{"instId": symbol_api, "ordId": o['ordId']} for o in all_pending.get('data', [])]:
                okx_request('POST', '/api/v5/trade/cancel-batch-orders', body_str=json.dumps(cancel_reqs))
                print_to_queue(log_queue, 'log', t("log_sl_cancelled_orders", count=len(cancel_reqs)))
    except Exception as e:
        print_to_queue(log_queue, 'log', t("log_sl_cancel_error", error=e))

    try:
        balance_state = fetch_balances_and_price(log_queue, t) # 注意：这里也需要传递 t
        if balance_state and balance_state['avail_base_bal'] >= min_sz:
            sell_amount = (balance_state['avail_base_bal'] / lot_sz).quantize(Decimal('0'), ROUND_DOWN) * lot_sz
            if sell_amount >= min_sz:
                print_to_queue(log_queue, 'log', t("log_sl_market_sell", amount=sell_amount, currency=cfg['BASE_CURRENCY']))
                payload = {"instId": symbol_api, "tdMode": "cash", "side": "sell", "ordType": "market", "sz": f"{sell_amount:.{amount_precision}f}"}
                okx_request('POST', '/api/v5/trade/order', body_str=json.dumps(payload))
                print_to_queue(log_queue, 'log', t("log_sl_market_sell_sent"))
            else:
                print_to_queue(log_queue, 'log', t("log_sl_below_min_size"))
        else:
            print_to_queue(log_queue, 'log', t("log_sl_no_position"))
    except Exception as e:
        print_to_queue(log_queue, 'log', t("log_sl_market_sell_error", error=e))
def update_pnl_and_log(state, account_initial_value, log_queue, amount_precision, t):
    pnl = state["total_value_usdt"] - account_initial_value
    pnl_data = {"base_bal": f"{state['base_bal']}", "quote_bal": f"{state['quote_bal']:.4f}", "total_value_usdt": f"{state['total_value_usdt']:.4f}", "pnl": f"{pnl:+.4f}", "pnl_raw": pnl}
    print_to_queue(log_queue, 'pnl', pnl_data)
    
    # [i18n] 使用翻译函数记录日志
    log_msg = t("log_pnl_update", 
                timestamp=time.strftime('%H:%M:%S'), 
                pnl=f"{pnl:+.4f}", 
                net_worth=f"{state['total_value_usdt']:.4f}", 
                price=state['current_price'])
    print_to_queue(log_queue, 'log', log_msg)

    with open(LOG_FILE, 'a', newline='') as f:
        f.write(f"{datetime.datetime.now(datetime.UTC).isoformat()},{state['total_value_usdt']:.4f},{pnl:.4f}\n")
def synchronize_orders(grid_orders, profit_orders, log_queue, price_precision, t):
    with config_lock: cfg = dynamic_config.copy()
    newly_filled_buys, newly_filled_sells = [], []; all_tracked_ids = list(grid_orders.keys()) + list(profit_orders)
    pending_orders_response = okx_request('GET', f"/api/v5/trade/orders-pending?instType=SPOT&instId={cfg['SYMBOL_API_FORMAT']}")
    if not all_tracked_ids: return grid_orders, profit_orders, pending_orders_response, [], []
    if not (fills_resp := okx_request('GET', f"/api/v5/trade/fills?instType=SPOT&instId={cfg['SYMBOL_API_FORMAT']}&limit=100")): return grid_orders, profit_orders, pending_orders_response, [], []
    fills_by_order_id = {oid:[] for oid in all_tracked_ids}
    for fill in fills_resp.get('data', []):
        if fill['ordId'] in fills_by_order_id:
            fills_by_order_id[fill['ordId']].append(fill)
    for ordId in list(grid_orders.keys()):
        if fills := fills_by_order_id.get(ordId):
            avg_price = sum(Decimal(f['fillPx'])*Decimal(f['fillSz']) for f in fills)/sum(Decimal(f['fillSz']) for f in fills);
            print_to_queue(log_queue, 'log', t("log_order_sync_buy_filled", order_id=ordId, price=f"{avg_price:.{price_precision}f}"))
            newly_filled_buys.extend(fills)
            grid_orders.pop(ordId, None)
    for ordId in list(profit_orders):
        if fills := fills_by_order_id.get(ordId): 
            print_to_queue(log_queue, 'log', t("log_order_sync_sell_filled", order_id=ordId))
            newly_filled_sells.extend(fills)
            profit_orders.discard(ordId)
    still_pending_ids, now = {o['ordId'] for o in pending_orders_response.get('data', [])}, time.time()
    for ordId in list(grid_orders.keys()):
        if not (order_data := grid_orders.get(ordId)) or not isinstance(order_data, dict):
            grid_orders.pop(ordId, None); continue
        if ordId not in still_pending_ids and (now - order_data.get('timestamp', 0) > NEW_ORDER_GRACE_PERIOD_SECONDS):
            grid_orders.pop(ordId, None)
    for ordId in list(profit_orders):
        if ordId not in still_pending_ids:
            profit_orders.discard(ordId)
    return grid_orders, profit_orders, pending_orders_response, newly_filled_buys, newly_filled_sells
def cancel_stale_grid_orders(pending_orders_response, profit_orders, grid_orders, log_queue, symbol_api, t):
    active_grid_ids, pending_api_ids = set(grid_orders.keys()), {o['ordId'] for o in (pending_orders_response.get('data') or [])}
    if ids_to_cancel := list((pending_api_ids - active_grid_ids) - profit_orders):
        print_to_queue(log_queue, 'log', t("log_order_sync_cancel_stale", count=len(ids_to_cancel)))
        cancel_req = [{"instId": symbol_api, "ordId": oid} for oid in ids_to_cancel]
        okx_request('POST', '/api/v5/trade/cancel-batch-orders', body_str=json.dumps(cancel_req))
        for oid in ids_to_cancel: grid_orders.pop(oid, None)
    return grid_orders
def place_new_strategy_orders(state, user_entered_capital, effective_mode, log_queue, price_precision, amount_precision, lot_sz, min_sz, t, chase_mode=False):
    placed_orders, body = {}, []
    with config_lock: cfg = dynamic_config.copy()
    
    effective_spread = cfg['SPREAD_PERCENTAGE']
    if chase_mode:
        chase_factor = Decimal(cfg.get('CHASE_SPREAD_FACTOR', '0.1'))
        effective_spread *= chase_factor
        # [i18n] 使用翻译函数记录日志
        print_to_queue(log_queue, 'log', t("log_chase_mode_activated", spread=f"{effective_spread:.4%}"))

    if effective_mode == "狙击" and state['avail_base_bal'] < lot_sz and state['avail_quote_bal'] >= user_entered_capital:
        px = (state['current_price'] * (Decimal('1') - effective_spread)).quantize(Decimal(f'1e-{price_precision}'), ROUND_DOWN)
        if px > 0:
            sz = user_entered_capital / px
            snapped_sz = (sz / lot_sz).quantize(Decimal('0'), ROUND_DOWN) * lot_sz
            if snapped_sz >= min_sz:
                body.append({"instId": cfg['SYMBOL_API_FORMAT'], "tdMode": "cash", "side": "buy", "ordType": "limit", "sz": f"{snapped_sz:.{amount_precision}f}", "px": f"{px:.{price_precision}f}"})
    
    elif effective_mode == "网格":
        base_px = state["current_price"] * (Decimal('1') - effective_spread)
        pairs = cfg.get('GRID_PAIRS', 1)
        cap_per_order = (min(user_entered_capital, state['avail_quote_bal']) / Decimal(pairs)) * (Decimal('1') - MAKER_FEE)
        if cap_per_order > MIN_ORDER_VALUE_USDT:
            for i in range(pairs):
                px = (base_px * (Decimal('1') - cfg['GRID_STEP_PERCENTAGE'] * i)).quantize(Decimal(f'1e-{price_precision}'), ROUND_DOWN)
                if px > 0:
                    sz = cap_per_order / px
                    snapped_sz = (sz / lot_sz).quantize(Decimal('0'), ROUND_DOWN) * lot_sz
                    if snapped_sz >= min_sz:
                        body.append({"instId": cfg['SYMBOL_API_FORMAT'], "tdMode": "cash", "side": "buy", "ordType": "limit", "sz": f"{snapped_sz:.{amount_precision}f}", "px": f"{px:.{price_precision}f}"})
    
    if body and (resp := okx_request('POST', '/api/v5/trade/batch-orders', json.dumps(body))) and resp.get('data'):
        s_count = 0
        for req, res in zip(body, resp['data']):
            if res.get('sCode') == '0' and res.get('ordId'):
                placed_orders[res['ordId']], s_count = {'side': req['side'], 'timestamp': time.time()}, s_count + 1
        if s_count > 0:
            # [i18n] 使用翻译函数记录日志
            print_to_queue(log_queue, 'log', t("log_place_order_success", count=s_count))
    return placed_orders
def initialize_log_file():
    """检查日志文件，如果不存在则创建并写入表头。"""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, 'w', newline='') as f:
            f.write("timestamp,total_value_usdt,pnl\n")
# ==============================================================================
# --- THE MAIN TRADING LOOP (V22.0 "ROBUST & PROFITABLE" I18N EDITION) ---
# ==============================================================================
def main_trading_loop(log_queue, stop_event, user_entered_capital, trading_mode, status_queue, t):
    """
    主交易循环，已完全国际化。
    :param t: 翻译函数 (translator.get)
    """
    # --- 初始化 ---
    initialize_log_file()
    print_to_queue(log_queue, 'status', t("status_initializing"))
    
    # [i18n] 从本地状态文件加载
    grid_orders, profit_orders, _, _, total_position_size, total_position_cost = load_state()
    cooldown_end_time = None

    # --- 获取交易对精度信息 ---
    try:
        with config_lock:
            cfg = dynamic_config.copy()
            symbol_api = cfg["SYMBOL_API_FORMAT"]
        inst_details = okx_request('GET', f'/api/v5/public/instruments?instType=SPOT&instId={symbol_api}')['data'][0]
        price_precision = len(inst_details['tickSz'].split('.')[-1]) if '.' in inst_details['tickSz'] else 0
        amount_precision = len(inst_details['lotSz'].split('.')[-1]) if '.' in inst_details['lotSz'] else 0
        if inst_details['lotSz'] == '1': amount_precision = 0
        lot_sz, min_sz = Decimal(inst_details['lotSz']), Decimal(inst_details['minSz'])
        with config_lock:
            dynamic_config.update(price_precision=price_precision, amount_precision=amount_precision)
        
        # [i18n] 使用翻译函数记录日志
        print_to_queue(log_queue, 'log', t("log_instrument_details", px_prec=price_precision, sz_prec=amount_precision, lot_sz=lot_sz, min_sz=min_sz))
    except Exception as e:
        print_to_queue(log_queue, 'log', t("log_startup_failed_details", error=e)); return

    # --- 初始状态同步与恢复 ---
    initial_state = fetch_balances_and_price(log_queue, t)
    if not initial_state:
        print_to_queue(log_queue, 'log', t("log_startup_failed_state")); return
    
    if total_position_size > min_sz and initial_state['base_bal'] < min_sz:
        print_to_queue(log_queue, 'log', t("log_state_mismatch_reset"))
        total_position_size, total_position_cost = Decimal('0'), Decimal('0')

    account_initial_value = initial_state["total_value_usdt"]
    
    effective_mode = ("狙击" if user_entered_capital <= MICRO_CAPITAL_THRESHOLD else "网格") if "自动" in trading_mode else trading_mode.replace("模式", "").replace("Mode", "").strip()
    
    print_to_queue(log_queue, 'log', t("log_bot_start_success", net_worth=f"{account_initial_value:.4f}", mode=effective_mode, capital=user_entered_capital))
    print_to_queue(log_queue, 'status', t("status_running"))

    # --- 主循环开始 ---
    while not stop_event.is_set():
        try:
            # --- 止损与冷却逻辑 ---
            try:
                if status_queue.get_nowait() == "STOP_LOSS_TRIGGERED" and not cooldown_end_time:
                    # [i18n] 将翻译函数 't' 传递给辅助函数
                    execute_smart_stop_loss(log_queue, min_sz, amount_precision, price_precision, lot_sz, t)
                    cooldown_end_time = time.time() + STOP_LOSS_COOLDOWN_MINUTES * 60
                    print_to_queue(log_queue, 'status', t("status_cooldown_info", minutes=STOP_LOSS_COOLDOWN_MINUTES))
                    grid_orders, profit_orders, total_position_size, total_position_cost = {}, set(), Decimal('0'), Decimal('0')
                    with config_lock: dynamic_config['entry_price'], dynamic_config['stop_loss_price'] = Decimal('0'), Decimal('0')
                    save_state(grid_orders, profit_orders, total_position_size, total_position_cost)
                    continue
            except queue.Empty: pass
            
            if cooldown_end_time and time.time() < cooldown_end_time:
                remaining = int(cooldown_end_time - time.time())
                print_to_queue(log_queue, 'log', t("log_cooldown_countdown", minutes=remaining // 60, seconds=remaining % 60)); stop_event.wait(5); continue
            elif cooldown_end_time:
                print_to_queue(log_queue, 'log', t("log_cooldown_over")); cooldown_end_time = None

            # --- 同步订单与更新状态 ---
            # [i18n] 将翻译函数 't' 传递给辅助函数
            grid_orders, profit_orders, pending, newly_filled_buys, newly_filled_sells = synchronize_orders(grid_orders, profit_orders, log_queue, price_precision, t)
            state = fetch_balances_and_price(log_queue,t)
            if state is None: stop_event.wait(cfg.get('POLL_FREQUENCY_SECONDS', 4)); continue
            update_pnl_and_log(state, account_initial_value, log_queue, amount_precision, t) # [i18n]
            
            if dynamic_config.get('capital_mode', '固定金额') == '固定金额':
                effective_capital = Decimal(dynamic_config.get('capital_value', '100'))
            else:
                effective_capital = state['total_value_usdt'] * (Decimal(dynamic_config.get('capital_value', '20')) / Decimal('100'))
            
            pos_data = {'size': total_position_size, 'avg_price': (total_position_cost / total_position_size) if total_position_size > 0 else Decimal('0'), 'sl_price': dynamic_config.get('stop_loss_price', Decimal('0'))}
            print_to_queue(log_queue, 'position', pos_data)
            grid_orders = cancel_stale_grid_orders(pending, profit_orders, grid_orders, log_queue, symbol_api, t) # [i18n]
            with config_lock: cfg = dynamic_config.copy()

            atr_for_regrid = cfg.get('atr_based_stop_loss_usdt', Decimal('0')) / STOP_LOSS_ATR_MULTIPLIER
            if grid_orders and pending.get('data') and atr_for_regrid > 0:
                orders_to_cancel = [
                    {"instId": symbol_api, "ordId": o['ordId']} 
                    for o in pending['data'] 
                    if o['ordId'] in grid_orders and o['side'] == 'buy' and state['current_price'] > Decimal(o['px']) + atr_for_regrid * REGRID_ATR_FACTOR
                ]
                if orders_to_cancel:
                    print_to_queue(log_queue, 'log', t("log_smart_chase_cancel", count=len(orders_to_cancel)))
                    okx_request('POST', '/api/v5/trade/cancel-batch-orders', body_str=json.dumps(orders_to_cancel))
                    for o in orders_to_cancel: grid_orders.pop(o['ordId'], None)

            # --- 状态化持仓管理 ---
            if newly_filled_buys:
                for fill in newly_filled_buys:
                    fill_size, fill_price = Decimal(fill['fillSz']), Decimal(fill['fillPx'])
                    total_position_cost += (fill_size * fill_price)
                    total_position_size += fill_size
                with config_lock: dynamic_config['entry_price'] = total_position_cost / total_position_size
                print_to_queue(log_queue, 'log', t("log_position_update_buy", size=total_position_size, avg_price=f"{dynamic_config['entry_price']:.{price_precision}f}"))

            if newly_filled_sells:
                print_to_queue(log_queue, 'log', t("log_profit_order_filled"))
                total_position_size, total_position_cost = Decimal('0'), Decimal('0')
                with config_lock: dynamic_config['entry_price'], dynamic_config['stop_loss_price'] = Decimal('0'), Decimal('0')

            # --- 动态追踪止损 ---
            if total_position_size > min_sz:
                avg_entry_price = total_position_cost / total_position_size
                with config_lock:
                    current_sl = dynamic_config.get('stop_loss_price', Decimal('0'))
                    if current_sl == 0:
                        initial_sl = (avg_entry_price - cfg['atr_based_stop_loss_usdt']).quantize(Decimal(f'1e-{price_precision}'), ROUND_DOWN)
                        dynamic_config['stop_loss_price'] = initial_sl
                        print_to_queue(log_queue, 'log', t("log_sl_set", price=initial_sl))
                    
                    trailing_stop_candidate = (state['current_price'] - cfg['atr_based_stop_loss_usdt']).quantize(Decimal(f'1e-{price_precision}'), ROUND_DOWN)
                    if trailing_stop_candidate > current_sl:
                        dynamic_config['stop_loss_price'] = trailing_stop_candidate
                        print_to_queue(log_queue, 'log', t("log_sl_trailed", price=trailing_stop_candidate))

            # --- 手续费感知的盈利挂单 ---
            if total_position_size >= min_sz and not profit_orders:
                for po in (p for p in pending.get('data', []) if p['side'] == 'sell'):
                    print_to_queue(log_queue, 'log', t("log_cancelling_zombie_sell", order_id=po['ordId']))
                    okx_request('POST', '/api/v5/trade/cancel-order', body_str=json.dumps({"instId": symbol_api, "ordId": po['ordId']}))

                if state['avail_base_bal'] >= min_sz:
                    try:
                        avg_entry_price = total_position_cost / total_position_size
                        break_even_price = avg_entry_price * (Decimal('1') + MAKER_FEE) / (Decimal('1') - TAKER_FEE)
                        profit_px = (break_even_price * (Decimal('1') + cfg['SPREAD_PERCENTAGE'])).quantize(Decimal(f'1e-{price_precision}'), ROUND_DOWN)
                        sell_qty = (state['avail_base_bal'] / lot_sz).quantize(Decimal('0'), ROUND_DOWN) * lot_sz
                        
                        print_to_queue(log_queue, 'log', t("log_profit_order_details", avg_price=f"{avg_entry_price:.{price_precision}f}", breakeven=f"{break_even_price:.{price_precision}f}", target=profit_px))
                        if sell_qty >= min_sz:
                            payload = {"instId": symbol_api, "tdMode": "cash", "side": "sell", "ordType": "limit", "sz": f"{sell_qty:.{amount_precision}f}", "px": f"{profit_px:.{price_precision}f}"}
                            resp = okx_request('POST', '/api/v5/trade/order', json.dumps(payload))
                            if resp and resp['data'][0].get('sCode') == '0':
                                profit_orders.add(resp['data'][0]['ordId'])
                                print_to_queue(log_queue, 'log', t("log_profit_order_placed", qty=f"{sell_qty:.{amount_precision}f}", price=profit_px))
                    except Exception as e:
                        print_to_queue(log_queue, 'log', t("log_place_order_error", error=e))

            # --- 放置新的网格买单 ---
            if not grid_orders and total_position_size < min_sz:
                if not is_trend_down(symbol_api, log_queue, t): # [i18n]
                    if new_orders := place_new_strategy_orders(state, effective_capital, effective_mode, log_queue, price_precision, amount_precision, lot_sz, min_sz, t): # [i18n]
                        grid_orders.update(new_orders)

            save_state(grid_orders, profit_orders, total_position_size, total_position_cost)
            stop_event.wait(cfg.get('POLL_FREQUENCY_SECONDS', 4))
        except Exception:
            print_to_queue(log_queue, 'log', t("log_main_loop_critical_error", traceback=traceback.format_exc())); stop_event.wait(CRITICAL_ERROR_WAIT_SECONDS)
    
    print_to_queue(log_queue, 'status', t("status_stopped"))
# ==============================================================================
# --- GUI APPLICATION (COMPLETE AND CORRECTED) ---
# ==============================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Initialize Translation Manager (default to Chinese)
        self.translator = TranslationManager(default_lang="zh")

        # 2. Set up the main window
        self.title(self.translator.get("app_title"))
        self.geometry("900x800")
        self.bot_thread = self.guardian_thread = self.stop_event = self.log_queue = self.status_queue = None
        self.user_entered_capital = Decimal('0')
        self.recommended_params = {}

        # --- Main Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.left_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")
        self.left_frame.grid_rowconfigure(1, weight=0)
        self.left_frame.grid_rowconfigure(2, weight=1)
        
        # --- Top Control Frame ---
        top_ctrl_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        top_ctrl_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # --- Language Selector ---
        self.lang_selector = ctk.CTkOptionMenu(top_ctrl_frame, values=self.translator.get_available_languages())
        self.lang_selector.set("zh")
        self.lang_selector.pack(side="left", padx=(0, 10), pady=10)
        self.lang_selector.configure(command=self._on_language_change)

        # --- Demo Mode Switch ---
        self.demo_mode_switch = ctk.CTkSwitch(top_ctrl_frame, text="")
        self.demo_mode_switch.pack(side="left", padx=10, pady=10)
        self.demo_mode_switch.configure(command=self.toggle_demo_mode)
        self.lbl_live_warning = ctk.CTkLabel(top_ctrl_frame, text="", text_color="#D32F2F", font=ctk.CTkFont(weight="bold"))

        # --- Tab View ---
        self.tabview = ctk.CTkTabview(self.left_frame)
        self.tabview.grid(row=1, column=0, padx=10, pady=0, sticky="new")
        
        # Define the permanent, non-translated keys for the tabs
        self.tab_keys = ["tab_mode_selection", "tab_strategy_params"]
        
        # Create tabs using the currently translated text as their name/identifier
        self.tab_mode = self.tabview.add(self.translator.get(self.tab_keys[0]))
        self.tab_params = self.tabview.add(self.translator.get(self.tab_keys[1]))

        # --- Tab 1: Mode Selection ---
        self.lbl_capital_mode = ctk.CTkLabel(self.tab_mode, text="")
        self.lbl_capital_mode.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.capital_mode_selector = ctk.CTkSegmentedButton(self.tab_mode)
        self.capital_mode_selector.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")
        self.capital_mode_selector.configure(command=self.update_capital_entry_label)

        self.lbl_capital_entry = ctk.CTkLabel(self.tab_mode, text="")
        self.lbl_capital_entry.grid(row=2, column=0, padx=10, pady=(5, 0), sticky="w")
        self.entry_capital = ctk.CTkEntry(self.tab_mode)
        self.entry_capital.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        self.lbl_trading_mode = ctk.CTkLabel(self.tab_mode, text="")
        self.lbl_trading_mode.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        self.mode_selector = ctk.CTkSegmentedButton(self.tab_mode)
        self.mode_selector.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.mode_selector.configure(command=self.update_ui_for_mode)

        self.lbl_manual_symbol = ctk.CTkLabel(self.tab_mode, text="")
        self.lbl_manual_symbol.grid(row=6, column=0, padx=10, pady=(10, 0), sticky="w")
        self.entry_manual_symbol = ctk.CTkEntry(self.tab_mode)
        self.entry_manual_symbol.grid(row=7, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.btn_set_manual = ctk.CTkButton(self.tab_mode, text="")
        self.btn_set_manual.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.btn_set_manual.configure(command=self.set_manual_symbol)
        
        self.scan_frame = ctk.CTkFrame(self.tab_mode, fg_color="transparent")
        self.scan_frame.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.scan_frame.grid_columnconfigure(0, weight=1)
        self.btn_scan = ctk.CTkButton(self.scan_frame, text="", fg_color="teal", hover_color="#005555")
        self.btn_scan.grid(row=0, column=0, sticky="ew")
        self.btn_scan.configure(command=self.start_scan)
        self.scan_progress = ctk.CTkProgressBar(self.scan_frame, orientation="horizontal", mode="determinate"); self.scan_progress.set(0)

        # --- Tab 2: Strategy Parameters ---
        self.param_entries, self.param_checkboxes, self.param_labels = {}, {}, {}
        self.params_to_show = {'POLL_FREQUENCY_SECONDS':"param_poll_frequency",'SPREAD_PERCENTAGE':"param_spread",'GRID_STEP_PERCENTAGE':"param_grid_step",'GRID_PAIRS':"param_grid_pairs",'ORDER_SIZE_USDT':"param_order_size"}
        for i, (key, text_key) in enumerate(self.params_to_show.items()):
            label = ctk.CTkLabel(self.tab_params, text=text_key); label.grid(row=i, column=0, padx=10, pady=5, sticky="w"); self.param_labels[key] = label
            entry = ctk.CTkEntry(self.tab_params); entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew"); self.param_entries[key] = entry
            cb = ctk.CTkCheckBox(self.tab_params, text=""); cb.grid(row=i, column=2, padx=5); self.param_checkboxes[key] = cb
        
        i += 1
        self.lbl_atr_sl_title = ctk.CTkLabel(self.tab_params, text=""); self.lbl_atr_sl_title.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        self.lbl_atr_sl = ctk.CTkLabel(self.tab_params, text="N/A", text_color="orange"); self.lbl_atr_sl.grid(row=i, column=1, padx=5, pady=5, sticky="w")
        
        i += 1
        self.select_all_var = ctk.StringVar(value="on")
        self.cb_select_all = ctk.CTkCheckBox(self.tab_params, text="", variable=self.select_all_var, onvalue="on", offvalue="off")
        self.cb_select_all.grid(row=i, column=0, padx=10, pady=10, sticky="w")
        self.cb_select_all.configure(command=self.toggle_all_param_checkboxes)
        self.toggle_all_param_checkboxes()

        self.btn_apply_rec = ctk.CTkButton(self.tab_params, text="", state="disabled")
        self.btn_apply_rec.grid(row=i, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        self.btn_apply_rec.configure(command=self.apply_recommended_params)
        
        i += 1
        self.btn_save_settings = ctk.CTkButton(self.tab_params, text="")
        self.btn_save_settings.grid(row=i, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.btn_save_settings.configure(command=lambda: self.save_settings(show_messagebox=True))

        # --- Status and Control Frame ---
        status_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        status_frame.grid(row=2, column=0, padx=10, pady=10, sticky="sew")
        self.btn_start = ctk.CTkButton(status_frame, text=""); self.btn_start.pack(fill="x", padx=10, pady=5)
        self.btn_start.configure(command=self.start_bot)
        self.btn_stop = ctk.CTkButton(status_frame, text="", state="disabled", fg_color="#D32F2F", hover_color="#B71C1C"); self.btn_stop.pack(fill="x", padx=10, pady=5)
        self.btn_stop.configure(command=self.stop_bot)
        self.btn_plot = ctk.CTkButton(status_frame, text=""); self.btn_plot.pack(fill="x", padx=10, pady=5)
        self.btn_plot.configure(command=self.show_equity_curve)

        with config_lock: current_symbol_display = dynamic_config['SYMBOL_DISPLAY']
        self.lbl_current_symbol_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_current_symbol_title.pack(fill="x", padx=10, pady=(10, 0))
        self.lbl_current_symbol = ctk.CTkLabel(status_frame, text=current_symbol_display, text_color="cyan", anchor="w"); self.lbl_current_symbol.pack(fill="x", padx=10, pady=(0, 5))
        
        self.lbl_status_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_status_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_status = ctk.CTkLabel(status_frame, text="", text_color="gray", anchor="w"); self.lbl_status.pack(fill="x", padx=10, pady=(0, 5))
        
        self.lbl_total_value_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_total_value_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_total_value = ctk.CTkLabel(status_frame, text="N/A", font=ctk.CTkFont(size=16), anchor="w"); self.lbl_total_value.pack(fill="x", padx=10, pady=0)
        self.lbl_pnl_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_pnl_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_pnl = ctk.CTkLabel(status_frame, text="N/A", font=ctk.CTkFont(size=16), anchor="w"); self.lbl_pnl.pack(fill="x", padx=10, pady=(0, 10))
        
        sep = ctk.CTkFrame(status_frame, height=2, fg_color="gray20"); sep.pack(fill="x", padx=10, pady=5)
        
        self.lbl_position_size_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_position_size_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_position_size = ctk.CTkLabel(status_frame, text="N/A", text_color="cyan", anchor="w"); self.lbl_position_size.pack(fill="x", padx=10, pady=(0, 5))
        self.lbl_avg_price_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_avg_price_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_avg_price = ctk.CTkLabel(status_frame, text="N/A", text_color="cyan", anchor="w"); self.lbl_avg_price.pack(fill="x", padx=10, pady=(0, 5))
        self.lbl_stop_loss_title = ctk.CTkLabel(status_frame, text="", font=ctk.CTkFont(weight="bold"), anchor="w"); self.lbl_stop_loss_title.pack(fill="x", padx=10, pady=(5, 0))
        self.lbl_stop_loss = ctk.CTkLabel(status_frame, text="N/A", text_color="orange", anchor="w"); self.lbl_stop_loss.pack(fill="x", padx=10, pady=(0, 10))

        # --- Log Textbox ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=0); self.right_frame.grid(row=0, column=1, sticky="nsew"); self.right_frame.grid_columnconfigure(0, weight=1); self.right_frame.grid_rowconfigure(0, weight=1); self.log_textbox = ctk.CTkTextbox(self.right_frame, state="disabled", font=ctk.CTkFont(family="Courier New", size=12)); self.log_textbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # --- Final Initializations ---
        self.setup_logging_tags()
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.entry_capital.bind("<KeyRelease>", lambda e: self.update_ui_for_mode())
        if not IS_DEMO_MODE: self.lbl_live_warning.pack(side="right", padx=10, pady=10)
        else: self.demo_mode_switch.select()

        # Update all UI texts to the default language
        self._update_ui_texts()

    def _on_language_change(self, new_lang: str):
        if new_lang not in self.translator.get_available_languages(): return
        self.translator.set_language(new_lang)
        self._update_ui_texts()
        self.add_log_message(self.translator.get("log_language_switched", lang=new_lang))

    def _update_ui_texts(self):
        self.title(self.translator.get("app_title"))
        self.demo_mode_switch.configure(text=self.translator.get("demo_mode_switch"))
        self.lbl_live_warning.configure(text=self.translator.get("live_mode_warning"))

        # --- Tab View Update Logic (Corrected) ---
        current_selected_tab_text = self.tabview.get()
        all_possible_old_titles = {key: [lang_data.get(key, key) for lang_data in self.translator.languages.values()] for key in self.tab_keys}
        
        selected_key = None
        for key, possible_names in all_possible_old_titles.items():
            if current_selected_tab_text in possible_names:
                selected_key = key
                break
        if selected_key is None: selected_key = self.tab_keys[0]

        for key in self.tab_keys:
            new_name = self.translator.get(key)
            for old_name in all_possible_old_titles[key]:
                if old_name in self.tabview._tab_dict and old_name != new_name:
                    self.tabview._tab_dict[new_name] = self.tabview._tab_dict.pop(old_name)
                    break
        
        new_titles = [self.translator.get(key) for key in self.tab_keys]
        self.tabview._segmented_button.configure(values=new_titles)
        self.tabview.set(self.translator.get(selected_key))
        
        # --- Other UI Elements ---
        self.lbl_capital_mode.configure(text=self.translator.get("capital_mode_label"))
        capital_mode_keys = ["capital_mode_fixed", "capital_mode_percentage"]
        capital_mode_values = [self.translator.get(key) for key in capital_mode_keys]
        self.capital_mode_selector.configure(values=capital_mode_values)
        self.update_capital_entry_label(self.capital_mode_selector.get())

        self.lbl_trading_mode.configure(text=self.translator.get("trading_mode_label"))
        mode_keys = ["auto_mode", "sniper_mode", "grid_mode"]
        self.mode_selector.configure(values=[self.translator.get(key) for key in mode_keys])

        self.lbl_manual_symbol.configure(text=self.translator.get("manual_symbol_label"))
        self.entry_manual_symbol.configure(placeholder_text=self.translator.get("manual_symbol_placeholder"))
        self.btn_set_manual.configure(text=self.translator.get("manual_symbol_button"))
        self.btn_scan.configure(text=self.translator.get("scan_button"))

        for key, label in self.param_labels.items():
            label.configure(text=self.translator.get(self.params_to_show[key]))
        self.lbl_atr_sl_title.configure(text=self.translator.get("param_atr_sl_label"))
        self.cb_select_all.configure(text=self.translator.get("param_select_all"))
        self.btn_apply_rec.configure(text=self.translator.get("apply_params_button"))
        self.btn_save_settings.configure(text=self.translator.get("save_settings_button"))

        self.btn_start.configure(text=self.translator.get("start_bot_button"))
        self.btn_stop.configure(text=self.translator.get("stop_bot_button"))
        self.btn_plot.configure(text=self.translator.get("plot_button"))

        self.lbl_current_symbol_title.configure(text=self.translator.get("current_pair_label"))
        self.lbl_status_title.configure(text=self.translator.get("status_label"))
        self.lbl_total_value_title.configure(text=self.translator.get("total_value_label"))
        self.lbl_pnl_title.configure(text=self.translator.get("pnl_label"))
        self.lbl_position_size_title.configure(text=self.translator.get("position_size_label"))
        self.lbl_avg_price_title.configure(text=self.translator.get("avg_price_label"))
        self.lbl_stop_loss_title.configure(text=self.translator.get("stop_loss_label"))
        
        current_status_text = self.lbl_status.cget("text")
        if self.translator.get("status_running") in current_status_text:
             self.lbl_status.configure(text=self.translator.get("status_running"))
        elif self.translator.get("status_stopping") in current_status_text:
            self.lbl_status.configure(text=self.translator.get("status_stopping"))
        elif self.translator.get("status_cooldown", details="") in current_status_text:
             # This part is tricky, we might just leave the detailed status as is
             pass
        else: # Default to stopped
            self.lbl_status.configure(text=self.translator.get("status_stopped"))
    def setup_logging_tags(self):
        self.log_textbox.tag_config("ERROR", foreground="#F6465D"); self.log_textbox.tag_config("WARNING", foreground="#FFC300"); self.log_textbox.tag_config("SUCCESS", foreground="#00B17B"); self.log_textbox.tag_config("INFO", foreground="cyan"); self.log_textbox.tag_config("DEFAULT", foreground=self.log_textbox.cget("text_color")[1] if ctk.get_appearance_mode() == "Dark" else self.log_textbox.cget("text_color")[0])
    def add_log_message(self, message):
        if self.log_textbox.winfo_exists():
            self.log_textbox.configure(state="normal"); tag = "DEFAULT"
            if "❌" in message or "错误" in message or "失败" in message or "🚨" in message: tag = "ERROR"
            elif "⚠️" in message or "警告" in message: tag = "WARNING"
            elif "✅" in message or "成功" in message or "恭喜" in message: tag = "SUCCESS"
            elif "ℹ️" in message or "模式" in message or "推荐" in message or "📈" in message: tag = "INFO"
            self.log_textbox.insert("end", message + "\n", tag); self.log_textbox.see("end"); self.log_textbox.configure(state="disabled")
    def update_ui_for_mode(self, mode=None):
        if not (mode := self.mode_selector.get()): return
        try:
            capital = Decimal(self.entry_capital.get() or '0'); is_grid = mode == "网格模式" or (mode == "自动模式" and capital > MICRO_CAPITAL_THRESHOLD)
            for i, (key, text) in enumerate(self.params_to_show.items()):
                widgets = self.tabview.tab("策略参数").grid_slaves(row=i)
                should_show = (key in ['GRID_STEP_PERCENTAGE', 'GRID_PAIRS'] and is_grid) or (key == 'ORDER_SIZE_USDT' and not is_grid) or key not in ['GRID_STEP_PERCENTAGE', 'GRID_PAIRS', 'ORDER_SIZE_USDT']
                for w in widgets:
                    w.grid() if should_show else w.grid_remove()
        except (ValueError, TclError, KeyError): pass
    def load_settings(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f: config_str = json.load(f)
            processed_config = {}
            numeric_keys = {'POLL_FREQUENCY_SECONDS': int, 'GRID_PAIRS': int, 'capital': Decimal, 'SPREAD_PERCENTAGE': Decimal, 'GRID_STEP_PERCENTAGE': Decimal, 'ORDER_SIZE_USDT': Decimal, 'atr_based_stop_loss_usdt': Decimal, 'entry_price': Decimal, 'stop_loss_price': Decimal}
            for key, value in config_str.items():
                if key in numeric_keys: processed_config[key] = numeric_keys[key](value)
                else: processed_config[key] = value
            with config_lock: dynamic_config.update(processed_config)
            self.entry_capital.delete(0, 'end'); self.entry_capital.insert(0, str(dynamic_config.get('capital', '100'))); self.mode_selector.set(str(dynamic_config.get('trading_mode', '自动模式')))
            for key, entry in self.param_entries.items():
                if key in dynamic_config:
                    entry.delete(0, 'end'); display_val = dynamic_config[key] * 100 if "PERCENTAGE" in key else dynamic_config[key]
                    entry.insert(0, str(display_val))
            self._update_symbol_config_display(str(dynamic_config.get('SYMBOL_API_FORMAT', DEFAULT_SYMBOL)))
            atr_sl = dynamic_config.get('atr_based_stop_loss_usdt', Decimal('0')); self.lbl_atr_sl.configure(text=f"{atr_sl:.4f}" if atr_sl > 0 else "N/A"); self.add_log_message("✅ 配置文件加载成功。")
        except Exception as e: self.add_log_message(f"❌ 加载配置文件错误: {e}")
    def save_settings(self, show_messagebox=True):
        try:
            temp_config = {}
            # 遍历所有参数输入框
            for key, entry in self.param_entries.items():
                # 只处理那些可见的、在UI上存在的输入框
                if entry.winfo_exists():
                    # 获取输入框的文本值
                    value_str = entry.get()
                
                    # 【FIX】如果输入框为空，则跳过此参数，不进行保存
                    if not value_str:
                        continue

                    # 现在我们可以安全地转换它，因为我们已经排除了空字符串
                    val = Decimal(value_str)
                
                    if val <= 0:
                        # [i18n] 使用翻译函数来显示错误消息
                        error_msg = self.translator.get("param_must_be_positive", param_name=key)
                        raise ValueError(error_msg)
                
                    if "PERCENTAGE" in key:
                        temp_config[key] = val / 100
                    elif key in ['POLL_FREQUENCY_SECONDS', 'GRID_PAIRS']:
                        temp_config[key] = int(val)
                    else:
                        temp_config[key] = val

            with config_lock:
                dynamic_config.update(temp_config)
        
            cfg_to_save = {k: str(v) for k, v in dynamic_config.items() if not callable(v) and k not in ['SYMBOL_DISPLAY','BASE_CURRENCY','QUOTE_CURRENCY', 'price_precision', 'amount_precision']}
        
            # 确保从正确的输入框获取资金和模式
            cfg_to_save.update(
                trading_mode=self.mode_selector.get(),
                capital_mode=self.capital_mode_selector.get(),
                capital_value=self.entry_capital.get() or '0' # 如果为空，默认为'0'
            )

            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg_to_save, f, indent=4)
        
            if show_messagebox:
                self.add_log_message(self.translator.get("log_settings_saved"))

        except (ValueError, InvalidOperation) as e:
            if show_messagebox:
                messagebox.showerror(self.translator.get("error_title"), self.translator.get("save_settings_error", error=e))
            # 重新抛出异常，以便在控制台中看到详细信息，有助于调试
            raise ValueError(e)
    def _update_symbol_config_display(self, symbol):
        symbol = symbol.strip().upper()
        if not symbol or '-' not in symbol: return False
        base, quote = symbol.split('-')
        with config_lock:
            if dynamic_config.get('SYMBOL_API_FORMAT') != symbol:
                dynamic_config.update(SYMBOL_API_FORMAT=symbol, SYMBOL_DISPLAY=f"{base}/{quote}", BASE_CURRENCY=base, QUOTE_CURRENCY=quote, atr_based_stop_loss_usdt=Decimal('0'))
                self.lbl_atr_sl.configure(text="N/A"); self.add_log_message(f"⚠️ 交易对已更改，ATR重置。")
        self.lbl_current_symbol.configure(text=f"{base}/{quote}"); return True
    def update_pnl_display(self, pnl_data):
        self.lbl_total_value.configure(text=pnl_data['total_value_usdt']); self.lbl_pnl.configure(text=pnl_data['pnl'])
        self.lbl_pnl.configure(text_color="#00B17B" if pnl_data['pnl_raw'] > 0 else "#F6465D" if pnl_data['pnl_raw'] < 0 else "gray")
    def start_analysis_thread(self, target_func, *args):
        if self.btn_scan.cget('state') == 'disabled': return
        self.set_ui_state(is_scanning=True)
        self.log_queue = queue.Queue()
    
        # ▼▼▼【核心修改】▼▼▼
        # 将翻译函数作为最后一个参数添加到*args中
        all_args = args + (self.translator.get,)
        threading.Thread(target=target_func, args=all_args, daemon=True).start()
        # ▲▲▲【核心修改】▲▲▲
    
        self.after(100, self.check_queue)
    def start_scan(self):
        try: capital = Decimal(self.entry_capital.get())
        except: messagebox.showerror("错误", "资金无效。"); return
        if capital <= 0: messagebox.showerror("错误", "资金需 > 0。"); return
        self.add_log_message("\n--- 请求市场扫描 ---"); self.start_analysis_thread(self.run_scanner_thread, capital)
    def run_scanner_thread(self, capital, t): # <-- 接收 't'
        scanner = MarketScanner(self.log_queue, t) # <-- 将 't' 传给扫描器实例
        print_to_queue(self.log_queue, 'scan_result', scanner.run_scan(capital))
    def set_manual_symbol(self):
        symbol = self.entry_manual_symbol.get()
        try: capital = Decimal(self.entry_capital.get())
        except: messagebox.showerror("错误", "资金无效。"); return
        self.add_log_message(f"\n--- 分析手动币种: {symbol} ---"); self.start_analysis_thread(self._set_manual_symbol_thread, symbol, capital)
    # This method is inside class App(ctk.CTk):

    def _set_manual_symbol_thread(self, symbol, capital, t): # <-- 接收 't'
        scanner = MarketScanner(self.log_queue, t)
        try:
            # [CRITICAL FIX] The log command MUST be given to the 'scanner' object, not 'self'.
            scanner.log("获取所有币种的Ticker信息(批量)...")
            
            tickers = {t['instId']: t for t in okx_request('GET', "/api/v5/market/tickers?instType=SPOT")['data']}
            
            # The analyze function itself is called on the scanner.
            result = scanner._analyze_candidate(symbol.strip().upper(), tickers)
            
            if not result:
                raise ValueError(f"无法分析 {symbol}，可能是数据不足或币种无效。")
            
            # Once analysis is successful, get the recommended params.
            recommended_params = scanner._get_recommended_params_for_candidate(result, capital)
            print_to_queue(self.log_queue, 'scan_result', recommended_params)

        except Exception as e:
            # Here, it's the App's responsibility to report the failure, so we use `add_log_message`.
            self.add_log_message(f"❌ 分析手动币种时出错: {e}")
            self.add_log_message(f"{traceback.format_exc()}") # More detailed error for debugging
            print_to_queue(self.log_queue, 'scan_result', None)
    def start_bot(self):
        try: self.save_settings(show_messagebox=False)
        except ValueError as e: 
            messagebox.showerror(self.translator.get("start_failed_title"), self.translator.get("param_error", error=e))
            return
        with config_lock: atr_sl = dynamic_config.get('atr_based_stop_loss_usdt', Decimal('0'))
        if not isinstance(atr_sl, Decimal) or atr_sl <= 0: 
            messagebox.showerror(self.translator.get("start_failed_title"), self.translator.get("atr_missing_error"))
            return
        
        self.set_ui_state(is_running=True)
        # 在这里，我们不再硬编码日志消息，而是让后台线程自己记录启动日志
    
        capital, mode = Decimal(self.entry_capital.get()), self.mode_selector.get()
        self.log_queue, self.status_queue, self.stop_event = queue.Queue(), queue.Queue(), threading.Event()
    
        # ▼▼▼【核心修改】▼▼▼
        # 获取翻译函数本身，以便传递给线程
        translator_func = self.translator.get
    
        threading.Thread(target=self.start_threads, args=(capital, mode, translator_func), daemon=True).start()
        # ▲▲▲【核心修改】▲▲▲
    
        self.after(100, self.check_queue)
    def start_threads(self, capital, mode, translator_func): # <-- 添加新参数
        # ▼▼▼【核心修改】▼▼▼
        self.guardian_thread = threading.Thread(target=guardian_loop, args=(self.log_queue, self.stop_event, self.status_queue, translator_func), daemon=True)
        self.bot_thread = threading.Thread(target=main_trading_loop, args=(self.log_queue, self.stop_event, capital, mode, self.status_queue, translator_func), daemon=True)
        # ▲▲▲【核心修改】▲▲▲
    
        self.guardian_thread.start()
        self.bot_thread.start()
    def stop_bot(self):
        if self.stop_event: self.add_log_message(self.translator.get("log_stopping_bot")); self.lbl_status.configure(text=self.translator.get("status_stopping"), text_color="orange"); self.stop_event.set()
        self.btn_stop.configure(state="disabled")
    def check_queue(self):
        """
        主GUI循环，用于从后台线程队列中获取消息并更新UI。
        此方法已完全国际化。
        """
        try:
            # 一次性处理队列中的所有消息
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
            
                # --- 根据消息类型分派任务 ---
            
                if msg['type'] == 'log':
                    # 日志消息应在后台线程中预先翻译好
                    self.add_log_message(msg['data'])
            
                elif msg['type'] == 'status':
                    # 状态消息需要在这里进行翻译
                    raw_text = msg['data']
                    color = "gray"
                    translated_status = raw_text # 默认回退

                    # 识别状态并应用翻译
                    if "冷却" in raw_text or "Cooldown" in raw_text:
                        # 提取动态部分 (例如时间)
                        details = ""
                        if "..." in raw_text:
                            details = raw_text.split("...", 1)[1]

                        translated_status = self.translator.get("status_cooldown", details=details)
                        color = "orange"
                    elif "运行中" in raw_text or "Running" in raw_text:
                        translated_status = self.translator.get("status_running")
                        color = "cyan"
                    elif "正在停止" in raw_text or "Stopping" in raw_text:
                        translated_status = self.translator.get("status_stopping")
                        color = "orange"
                
                    self.lbl_status.configure(text=translated_status, text_color=color)

                elif msg['type'] == 'pnl':
                    # pnl数据是原始数字，由更新函数处理显示
                    self.update_pnl_display(msg['data'])
            
                elif msg['type'] == 'position':
                    # 持仓数据是原始数字，由更新函数处理显示
                    self.update_position_display(msg['data'])
            
                elif msg['type'] == 'scan_result':
                    # 扫描结果处理完毕后，应停止当前循环以避免冲突
                    self.process_scan_result(msg['data'])
                    return
            
                elif msg['type'] == 'progress':
                    # 更新扫描进度条
                    self.update_scan_progress(msg['data'])

        except queue.Empty:
            # 队列为空是正常情况，无需处理
            pass
    
        # --- 检查后台线程状态并更新UI ---
    
        is_bot_running = self.bot_thread and self.bot_thread.is_alive()
        is_scanning = self.btn_scan.cget('state') == 'disabled' and not is_bot_running

        # 如果机器人已停止运行，但UI状态仍显示为“运行中”或“正在停止”，则重置UI
        current_ui_status = self.lbl_status.cget("text")
        stopped_text = self.translator.get("status_stopped")
        error_text = self.translator.get("status_error") # 假设你在json中定义了 'status_error'

        if not is_bot_running and current_ui_status not in [stopped_text, error_text]:
            self.set_ui_state(is_running=False)
            self.lbl_status.configure(text=stopped_text, text_color="gray")

        # 如果机器人或扫描器仍在运行，则安排下一次检查
        if is_bot_running or is_scanning:
            self.after(500, self.check_queue)
    def process_scan_result(self, results):
        self.set_ui_state(is_scanning=False)
        self.recommended_params = results
        if results:
            self.add_log_message("✅ 分析成功！")
            self._update_symbol_config_display(results['SYMBOL_API_FORMAT'])
            self.apply_recommended_params(auto_apply_all=True)
            self.btn_apply_rec.configure(state="normal")
            with config_lock: dynamic_config['atr_based_stop_loss_usdt'] = results['atr_based_stop_loss_usdt']
            self.lbl_atr_sl.configure(text=f"{results['atr_based_stop_loss_usdt']:.4f}")
    def update_scan_progress(self, data):
        if data: self.scan_progress.grid(row=1, column=0, sticky="ew", pady=(5,0)); self.scan_progress.set(float(data[0])/data[1])
        else: self.scan_progress.grid_remove(); self.scan_progress.set(0)
    def show_equity_curve(self):
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0: return messagebox.showinfo("信息", "日志文件为空。")
        df = pd.read_csv(LOG_FILE); new_window = ctk.CTkToplevel(self); new_window.title("权益曲线"); new_window.geometry("800x600")
        plt.style.use('dark_background'); plt.rcParams['axes.unicode_minus'] = False; fig, ax = plt.subplots(figsize=(8,6), dpi=100)
        ax.plot(pd.to_datetime(df['timestamp']), df['total_value_usdt'], label="账户总净值 (USD)", color='cyan'); ax.set_title("资产权益曲线"); ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
        FigureCanvasTkAgg(fig, master=new_window).get_tk_widget().pack(fill=ctk.BOTH, expand=1)
    def toggle_demo_mode(self):
        global IS_DEMO_MODE
        if self.bot_thread and self.bot_thread.is_alive(): messagebox.showwarning("操作禁止", "机器人运行时无法切换模式！"); self.demo_mode_switch.toggle(); return
        if self.demo_mode_switch.get() == 0:
            if "YOUR_" in API_CREDENTIALS['live']['trading']['api_key']: messagebox.showerror("错误", "实盘API未配置！"); self.demo_mode_switch.select(); return
            if messagebox.askyesno("实盘警告", "您确定要切换到实盘模式吗？", icon='warning'): IS_DEMO_MODE = False; self.lbl_live_warning.pack(side="right", padx=10)
            else: self.demo_mode_switch.select()
        else: IS_DEMO_MODE = True; self.lbl_live_warning.pack_forget()
    def toggle_all_param_checkboxes(self):
        is_on = self.select_all_var.get() == "on"
        for cb in self.param_checkboxes.values(): (cb.select() if is_on else cb.deselect())
    def apply_recommended_params(self, auto_apply_all=False):
        if not self.recommended_params: return
        for key, entry in self.param_entries.items():
            if (self.param_checkboxes[key].get() == 1 or auto_apply_all) and key in self.recommended_params:
                val = self.recommended_params[key]
                display_val = val * 100 if "PERCENTAGE" in key else val
                entry.delete(0, 'end')
                entry.insert(0, f"{display_val:.4f}" if isinstance(display_val, Decimal) and 'PAIRS' not in key and 'SECONDS' not in key else str(int(display_val)))
    def set_ui_state(self, is_running=False, is_scanning=False):
        state = "disabled" if is_running or is_scanning else "normal"
        self.btn_start.configure(state="disabled" if is_running or is_scanning else "normal"); self.btn_stop.configure(state="normal" if is_running else "disabled")
        for w in [self.btn_save_settings, self.demo_mode_switch, self.entry_capital, self.entry_manual_symbol, self.mode_selector, self.btn_scan, self.btn_set_manual]: w.configure(state=state)
    def on_closing(self):
        if self.bot_thread and self.bot_thread.is_alive(): self.stop_bot(); self.after(3000, self.destroy_app)
        else: self.destroy_app()
    def destroy_app(self):
        try: self.save_settings(show_messagebox=False)
        except: pass
        self.destroy()
    def update_capital_entry_label(self, mode):
        """根据资金管理模式切换输入框的标签。"""
        if mode == self.translator.get("capital_mode_fixed"):
            self.lbl_capital_entry.configure(text=self.translator.get("capital_entry_fixed_label"))
            self.entry_capital.configure(placeholder_text=self.translator.get("capital_entry_fixed_placeholder"))
        else: # 净值占比
            self.lbl_capital_entry.configure(text=self.translator.get("capital_entry_percentage_label"))
            self.entry_capital.configure(placeholder_text=self.translator.get("capital_entry_percentage_placeholder"))

    def update_position_display(self, pos_data):
        """更新GUI上的持仓信息显示。"""
        # 从配置中获取精度信息
        with config_lock:
            amount_prec = dynamic_config.get('amount_precision', 8)
            price_prec = dynamic_config.get('price_precision', 8)
            base_currency = dynamic_config.get('BASE_CURRENCY', '')

        # 更新持仓数量
        pos_size = pos_data.get('size', Decimal('0'))
        if pos_size > 0:
            self.lbl_position_size.configure(text=f"{pos_size:.{amount_prec}f} {base_currency}")
        else:
            self.lbl_position_size.configure(text="N/A")

        # 更新平均成本
        avg_price = pos_data.get('avg_price', Decimal('0'))
        if avg_price > 0:
            self.lbl_avg_price.configure(text=f"{avg_price:.{price_prec}f}")
        else:
            self.lbl_avg_price.configure(text="N/A")

        # 更新止损价格
        sl_price = pos_data.get('sl_price', Decimal('0'))
        if sl_price > 0:
            self.lbl_stop_loss.configure(text=f"{sl_price:.{price_prec}f}")
        else:
            self.lbl_stop_loss.configure(text="N/A")
if __name__ == '__main__':
    try:
        ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
        app = App()
        app.mainloop()
    except Exception as e:
        traceback.print_exc()
