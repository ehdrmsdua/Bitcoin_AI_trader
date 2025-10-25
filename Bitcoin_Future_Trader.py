# Classification (XGBoost 버전)
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from joblib import load
import xgboost as xgb

# ===== 사용자 설정 =====
API_KEY = 'input_your_api_key'
API_SECRET = 'input_your_api_secret_key'

SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LOOK_BACK = 1

# XGBoost 분류기 모델과 스케일러 경로
XGB_MODEL_PATH =
SCALER_PATH =
# ===== 거래소 연결 =====
binance = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ===== 모델/스케일러 로드 =====
clf = xgb.XGBClassifier()
clf.load_model(XGB_MODEL_PATH)
scaler = load(SCALER_PATH)

# ===== 런타임 상태 =====
input_queue = []
current_position = None        # 'long' / 'short' / None
open_position_price = None

# ===== 유틸 =====
def round_amount(symbol, amount):
    """거래소 최소 수량·자리수에 맞춰 보정"""
    market = binance.market(symbol)
    prec = market['precision']['amount']
    min_qty = market.get('limits', {}).get('amount', {}).get('min', None)

    if prec is not None:
        amount = float(f"{amount:.{prec}f}")
    if min_qty is not None and amount < float(min_qty):
        amount = float(f"{float(min_qty):.{prec}f}") if prec is not None else float(min_qty)
    return amount

def set_stop_loss(symbol, position_size, last_price, is_long, stop_loss):
    stop_price = last_price * (1 - stop_loss) if is_long else last_price * (1 + stop_loss)
    side = 'SELL' if is_long else 'BUY'
    params = {
        'stopPrice': stop_price,
        'closePosition': True,   # 전량 청산
        # 'reduceOnly': True,    # 필요 시 활성화
    }
    binance.create_order(symbol, 'STOP_MARKET', side, position_size, None, params)
    print(f"Set stop-loss at {stop_price}")

def get_position_size(symbol):
    sym_raw = symbol.replace("/", "")
    positions = binance.fapiPrivateV2GetPositionRisk({'symbol': sym_raw})
    pos = 0.0
    for p in positions:
        if p['symbol'] == sym_raw:
            pos = abs(float(p['positionAmt']))
            break
    return pos

def cancel_stop_loss_orders(symbol):
    try:
        open_orders = binance.fetch_open_orders(symbol=symbol)
        for order in open_orders:
            if str(order.get('type', '')).lower() == 'stop_market':
                binance.cancel_order(order['id'], symbol=symbol)
        print("Stop loss orders cancelled successfully.")
    except Exception as e:
        print(f"Error cancelling stop loss orders: {e}")

def close_position(symbol, last_price):
    global current_position, open_position_price
    size = get_position_size(symbol)
    if size <= 0:
        print("No position to close.")
        current_position = None
        open_position_price = None
        return

    if current_position == 'long':
        pl = (last_price - open_position_price) * size
        binance.create_market_sell_order(symbol, size, params={'reduceOnly': True})
    elif current_position == 'short':
        pl = (open_position_price - last_price) * size
        binance.create_market_buy_order(symbol, size, params={'reduceOnly': True})
    else:
        pl = 0.0

    print(f"Profit/Loss: {pl} USDT")
    current_position = None
    open_position_price = None

def get_input(q):
    while True:
        try:
            q.append(input())
        except EOFError:
            break

# ===== 예측 및 트레이딩 =====
def predict_and_trade(amount_in_usdt, leverage, stop_loss):
    global current_position, open_position_price

    limit = 20 + LOOK_BACK
    since_time = datetime.utcnow() - timedelta(hours=limit)
    since = binance.parse8601(since_time.isoformat() + 'Z')

    candles = binance.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 특징: close, returns, high, low
    df['returns'] = df['close'].pct_change().fillna(0.0)
    window = df[['close', 'returns', 'high', 'low']].values[-LOOK_BACK:]

    # 스케일링 후 평탄화 → XGB 입력
    window_scaled = scaler.transform(window)              # (LOOK_BACK, 4)
    x_input = window_scaled.flatten().reshape(1, -1)      # (1, LOOK_BACK*4)

    # 상승 확률
    if hasattr(clf, "predict_proba"):
        prob_up = float(clf.predict_proba(x_input)[0][1])
    else:
        score = float(clf.decision_function(x_input)[0])
        prob_up = 1.0 / (1.0 + np.exp(-score))
    print(f"Predicted prob(up): {prob_up:.6f}")

    last_price = float(df['close'].iloc[-1])
    total_usdt_with_leverage = amount_in_usdt * leverage
    amount_in_coin = total_usdt_with_leverage / last_price
    amount_in_coin = round_amount(SYMBOL, amount_in_coin)

    threshold = 0.5
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 롱 진입
    if prob_up > threshold and current_position != 'long':
        print("time:", now)
        if current_position == 'short':
            close_position(SYMBOL, last_price)
            cancel_stop_loss_orders(SYMBOL)
            open_position_price = last_price
        else:
            open_position_price = last_price

        print(f"Going long with {amount_in_coin} {SYMBOL}")
        binance.create_market_buy_order(SYMBOL, amount_in_coin)
        current_position = 'long'
        set_stop_loss(SYMBOL, amount_in_coin, last_price, True, stop_loss)

    # 숏 진입
    elif prob_up <= threshold and current_position != 'short':
        print("time:", now)
        if current_position == 'long':
            close_position(SYMBOL, last_price)
            cancel_stop_loss_orders(SYMBOL)
            open_position_price = last_price
        else:
            open_position_price = last_price

        print(f"Going short with {amount_in_coin} {SYMBOL}")
        binance.create_market_sell_order(SYMBOL, amount_in_coin)
        current_position = 'short'
        set_stop_loss(SYMBOL, amount_in_coin, last_price, False, stop_loss)

# ===== 메인 루프 =====
def main():
    amount_in_usdt = int(input("Enter your amount - USDT: "))
    leverage = int(input("Enter your leverage: "))
    stop_loss = float(input("Enter your stop_loss: "))  # 예: 0.02

    t = threading.Thread(target=get_input, args=(input_queue,))
    t.daemon = True
    t.start()

    while True:
        if input_queue and input_queue[-1] == 'q':
            print("Program is exiting...")
            break
        current_time = datetime.utcnow()
        wait_time = 3600 - (current_time.minute * 60 + current_time.second)
        print("waiting...\n")
        time.sleep(wait_time + 60)
        predict_and_trade(amount_in_usdt, leverage, stop_loss)

if __name__ == "__main__":
    main()
