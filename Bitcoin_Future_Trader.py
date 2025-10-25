#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Classification (PyTorch 버전)
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from joblib import load
import torch
import torch.nn as nn

# ========= 사용자 설정 =========
API_KEY ='input_your_api_key'
API_SECRET = 'input_your_api_secret_key'

# 모델 설정
MODEL_TYPE = 'torchscript'  # 'torchscript' 또는 'state_dict'
MODEL_PATH = r"C:\Users\admin\Desktop\model_epoch_4k_150_24_82.pt"  # TorchScript(.pt) 기준
LOOK_BACK = 24

SCALER_PATH = r"C:\Users\admin\Desktop\coinScale\4k_scaler.save"  # StandardScaler

# ========= 바이낸스 연결 =========
binance = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= (선택) state_dict 모델 구조 정의 =========
# state_dict로 저장했다면 아래 클래스의 hidden_size, num_layers 등을 학습 때와 동일하게 맞추고
# MODEL_TYPE='state_dict' 로 바꿔서 사용하세요.
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # (B, H)
        logits = self.fc(last)         # (B, 1)
        prob = torch.sigmoid(logits)   # (B, 1) -> 0~1
        return prob

# ========= 모델/스케일러 로드 =========
def load_model_pytorch():
    if MODEL_TYPE == 'torchscript':
        model = torch.jit.load(MODEL_PATH, map_location=device)
        model.eval()
        return model
    else:
        # state_dict로 저장된 경우 (경로는 MODEL_PATH)
        model = SimpleLSTM(input_size=4, hidden_size=64, num_layers=1, dropout=0.0).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        # state가 {'state_dict': ...} 형태면 state['state_dict']로 로드
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

model = load_model_pytorch()
scaler = load(SCALER_PATH)

# ========= 런타임 상태 =========
input_queue = []
open_position_price = None
current_position = None  # 'long' / 'short' / None

# ========= 유틸 =========
def round_amount(symbol, amount):
    """마켓 최소 수량/자리수 맞춰 반올림. 최소 수량 미만이면 보정."""
    market = binance.market(symbol)
    prec = market['precision']['amount']
    min_qty = market.get('limits', {}).get('amount', {}).get('min', None)

    if prec is not None:
        amount = float(f"{amount:.{prec}f}")

    if min_qty is not None and amount < float(min_qty):
        amount = float(f"{float(min_qty):.{prec}f}") if prec is not None else float(min_qty)

    return amount

def set_stop_loss(symbol, position_size, last_price, is_long, stop_loss):
    if is_long:
        stop_price = last_price * (1 - stop_loss)
        side = 'SELL'
    else:
        stop_price = last_price * (1 + stop_loss)
        side = 'BUY'

    params = {
        'stopPrice': stop_price,
        'closePosition': True
        # 'reduceOnly': True,  # 필요시 활성화
    }
    binance.create_order(symbol, 'STOP_MARKET', side, position_size, None, params)
    print(f"Set stop-loss at {stop_price}")

def get_position_size(symbol):
    sym_raw = symbol.replace("/", "")
    positions = binance.fapiPrivateV2GetPositionRisk({'symbol': sym_raw})
    position_size = 0.0
    for p in positions:
        if p['symbol'] == sym_raw:
            position_size = abs(float(p['positionAmt']))
            break
    return position_size

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
    position_size = get_position_size(symbol)
    if position_size <= 0:
        print("No position to close.")
        current_position = None
        open_position_price = None
        return

    if current_position == 'long':
        profit_loss = (last_price - open_position_price) * position_size
        binance.create_market_sell_order(symbol, position_size, params={'reduceOnly': True})
    elif current_position == 'short':
        profit_loss = (open_position_price - last_price) * position_size
        binance.create_market_buy_order(symbol, position_size, params={'reduceOnly': True})
    else:
        profit_loss = 0.0

    print(f"Profit/Loss: {profit_loss} USDT")
    current_position = None
    open_position_price = None

def get_input(input_queue):
    while True:
        try:
            inp = input()
            input_queue.append(inp)
        except EOFError:
            break

# ========= 예측 및 트레이딩 =========
def predict_and_trade(amount_in_usdt, leverage, stop_loss):
    global current_position, open_position_price

    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 20 + LOOK_BACK

    since_time = datetime.utcnow() - timedelta(hours=limit)
    since = binance.parse8601(since_time.isoformat() + 'Z')

    candles = binance.fetch_ohlcv(symbol, timeframe, since, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df['returns'] = df['close'].pct_change().fillna(0.0)
    feats = df[['close', 'returns', 'high', 'low']].values[-LOOK_BACK:]

    # 스케일링
    feats_scaled = scaler.transform(feats).astype(np.float32)
    x = torch.from_numpy(feats_scaled).unsqueeze(0).to(device)  # (1, T, F)

    # 예측 (0~1 확률)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        prob_up = float(y.squeeze().detach().cpu().item())
    print(f"Predicted prob(up): {prob_up:.6f}")

    last_price = float(df['close'].iloc[-1])
    total_usdt_with_leverage = amount_in_usdt * leverage
    amount_in_coin = total_usdt_with_leverage / last_price
    amount_in_coin = round_amount(symbol, amount_in_coin)

    now = datetime.now()
    threshold = 0.5

    if prob_up > threshold and current_position != 'long':
        print("time:", now.strftime("%Y-%m-%d %H:%M:%S"))
        if current_position == 'short':
            close_position(symbol, last_price)
            cancel_stop_loss_orders(symbol)
            open_position_price = last_price
        else:
            open_position_price = last_price

        print(f"Going long with {amount_in_coin} {symbol}")
        binance.create_market_buy_order(symbol, amount_in_coin)
        current_position = 'long'
        set_stop_loss(symbol, amount_in_coin, last_price, True, stop_loss)

    elif prob_up <= threshold and current_position != 'short':
        print("time:", now.strftime("%Y-%m-%d %H:%M:%S"))
        if current_position == 'long':
            close_position(symbol, last_price)
            cancel_stop_loss_orders(symbol)
            open_position_price = last_price
        else:
            open_position_price = last_price

        print(f"Going short with {amount_in_coin} {symbol}")
        binance.create_market_sell_order(symbol, amount_in_coin)
        current_position = 'short'
        set_stop_loss(symbol, amount_in_coin, last_price, False, stop_loss)

# ========= 메인 =========
def main():
    # 입력
    amount_in_usdt = int(input("Enter your amount - USDT: "))
    leverage = int(input("Enter your leverage: "))
    stop_loss = float(input("Enter your stop_loss: "))  # 예: 0.02

    # 입력 스레드 (q 입력 시 종료)
    input_thread = threading.Thread(target=get_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

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

