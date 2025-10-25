# -*- coding: utf-8 -*-
"""
XGBoost 최적 파라미터로 테스트하는 코드
- 학습/검증/테스트 전부 합쳐 학습
- 내부 홀드아웃(9:1)으로 성능 평가
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import random

# CuPy 사용 가능 여부 확인 (GPU 데이터 처리용)
try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    CUPY_AVAILABLE = True
    print("CuPy 사용 가능 - 전체 GPU 파이프라인 활성화")
except (ImportError, RuntimeError, FileNotFoundError):
    CUPY_AVAILABLE = False
    print("CuPy 사용 불가 - CPU 모드로 실행")

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def add_derived_features(df):
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['SMA'] = calculate_sma(df['close'])
    df['close_diff'] = df['close'].diff()
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff()
    df['volume_diff'] = df['volume'].diff()
    df['close_change_ratio'] = df['close'].pct_change()
    df['high_change_ratio'] = df['high'].pct_change()
    df['low_change_ratio'] = df['low'].pct_change()
    df['volume_change_ratio'] = df['volume'].pct_change()
    df['close_ma_diff'] = df['close'] - df['close'].rolling(5).mean()
    close_mean = df['close'].rolling(10).mean()
    df['volatility_ratio'] = df['close'].rolling(10).std() / (close_mean + 1e-8)
    close_min = df['close'].rolling(20).min()
    close_max = df['close'].rolling(20).max()
    df['price_position'] = (df['close'] - close_min) / (close_max - close_min + 1e-8)
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    volume_sum = df['volume'].rolling(20).sum()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (volume_sum + 1e-8)
    df['price_vs_vwap'] = df['close'] / (df['vwap'] + 1e-8) - 1

    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
    else:
        df['hour'] = df.index % 24
        df['day_of_week'] = (df.index // 24) % 7
        df['day_of_month'] = (df.index // (24 * 30)) % 30
        df['month'] = (df.index // (24 * 30 * 12)) % 12

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def calculate_obv(close, volume):
    obv = np.zeros(len(close))
    obv[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_sma(prices, period=20):
    return prices.rolling(window=period).mean()

def create_sequences_with_lookback(features, targets, look_back):
    if look_back <= 1:
        return features, targets
    X, y = [], []
    for i in range(look_back, len(features)):
        seq = features[i - look_back:i].flatten()
        X.append(seq)
        y.append(targets[i])
    X_result = np.array(X)
    y_result = np.array(y)
    return X_result, y_result

def main():
    optimal_params = {
        'n_estimators': 345,
        'max_depth': 5,
        'learning_rate': 0.01013632140862702,
        'subsample': 0.7046845119954543,
        'colsample_bytree': 0.8179149394717654,
        'reg_alpha': 3.655913166451287,
        'reg_lambda': 5.949939970720599,
        'random_state': 42,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda'
    }

    look_back = 1

    print("=== XGBoost 최적 파라미터 테스트 ===")
    print(f"최적 파라미터: {optimal_params}")

    CSV_FILE_PATH = r"2023317.csv"
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"데이터 로드 완료: {df.shape}")
    except FileNotFoundError:
        print(f"데이터 파일을 찾을 수 없습니다. '{CSV_FILE_PATH}' 파일이 존재하는지 확인하세요.")
        return
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return

    # 파생 변수 및 타깃 생성
    df = add_derived_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['returns'] = df['close'].pct_change() * 100
    df['target'] = (df['returns'] > 0).astype(int).shift(-1)

    df = df.dropna(subset=['target'])

    base_feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'close_diff', 'high_diff', 'low_diff', 'volume_diff',
        'close_change_ratio', 'high_change_ratio', 'low_change_ratio', 'volume_change_ratio',
        'close_ma_diff', 'volatility_ratio', 'price_position',
        'momentum_5', 'momentum_10', 'vwap', 'price_vs_vwap',
        'RSI', 'MACD', 'MACD_Signal', 'OBV', 'ATR', 'SMA',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]

    # 넘파이로 변환
    features_np = df[base_feature_columns].values.astype(float)
    targets_np = df['target'].values.astype(int)

    # 6:2:2 분할(형식 유지) 후, 나중에 세 묶음 전부 합쳐 학습에 사용
    total_size = len(features_np)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)

    train_features_np = features_np[:train_size]
    train_targets_np = targets_np[:train_size]
    val_features_np = features_np[train_size:train_size + val_size]
    val_targets_np = targets_np[train_size:train_size + val_size]
    test_features_np = features_np[train_size + val_size:]
    test_targets_np = targets_np[train_size + val_size:]

    # 스케일링
    print("특성 스케일링 적용 중...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features_np)
    val_scaled = scaler.transform(val_features_np)
    test_scaled = scaler.transform(test_features_np)

    print(f"스케일링 전 통계 (훈련 데이터):")
    print(f"평균: {train_features_np.mean():.4f}")
    print(f"표준편차: {train_features_np.std():.4f}")
    print(f"스케일링 후 통계 (훈련 데이터):")
    print(f"평균: {train_scaled.mean():.4f}")
    print(f"표준편차: {train_scaled.std():.4f}")

    # 학습에 사용할 전체 묶음 = 학습 + 검증 + 테스트 (요청 사항 반영)
    combined_all_features = np.vstack([train_scaled, val_scaled, test_scaled])
    combined_all_targets = np.hstack([train_targets_np, val_targets_np, test_targets_np])

    print(f"데이터 분할(원본): Train={len(train_scaled)}, Val={len(val_scaled)}, Test={len(test_scaled)}")
    print(f"최종 학습용(합침): {len(combined_all_features)} (훈련+검증+테스트 모두 포함)")

    # 내부 홀드아웃(9:1)
    internal_train_size = int(0.9 * len(combined_all_features))
    internal_train_features = combined_all_features[:internal_train_size]
    internal_train_targets = combined_all_targets[:internal_train_size]
    internal_val_features = combined_all_features[internal_train_size:]
    internal_val_targets = combined_all_targets[internal_train_size:]

    print(f"내부 훈련: {len(internal_train_features)}, 내부 검증: {len(internal_val_features)}")

    # look_back 적용 (기본 1: 그대로 사용)
    if look_back > 1:
        train_X, train_y = create_sequences_with_lookback(internal_train_features, internal_train_targets, look_back)
        val_X, val_y = create_sequences_with_lookback(internal_val_features, internal_val_targets, look_back)
    else:
        train_X, train_y = internal_train_features, internal_train_targets
        val_X, val_y = internal_val_features, internal_val_targets

    # GPU 사용 시 CuPy로 변환
    if CUPY_AVAILABLE:
        train_X_gpu = cp.array(train_X)
        train_y_gpu = cp.array(train_y)
        val_X_gpu = cp.array(val_X)
        val_y_gpu = cp.array(val_y)
    else:
        train_X_gpu = train_X
        train_y_gpu = train_y
        val_X_gpu = val_X
        val_y_gpu = val_y

    # 모델 학습
    final_model = xgb.XGBClassifier(**optimal_params)
    print("\n=== 모델 훈련 시작 ===")
    print("전 데이터(학습+검증+테스트)로 학습, 내부 홀드아웃으로 평가")

    # 조기 종료를 쓰고 싶으면 early_stopping_rounds를 켜세요.
    # 여기서는 기본 그대로 둡니다.
    final_model.fit(
        train_X_gpu, train_y_gpu,
        eval_set=[(val_X_gpu, val_y_gpu)],
        verbose=False
    )

    print("훈련 완료!")

    # 내부 검증 성능 평가
    print("\n=== 내부 홀드아웃 성능 평가 ===")
    val_pred = final_model.predict(val_X_gpu)
    if CUPY_AVAILABLE:
        val_pred = cp.asnumpy(val_pred)
        val_y_cpu = cp.asnumpy(val_y_gpu)
    else:
        val_y_cpu = val_y

    acc = accuracy_score(val_y_cpu, val_pred)
    prec = precision_score(val_y_cpu, val_pred)
    rec = recall_score(val_y_cpu, val_pred)
    f1 = f1_score(val_y_cpu, val_pred)

    print(f"정확도: {acc:.4f}")
    print(f"정밀도: {prec:.4f}")
    print(f"재현율: {rec:.4f}")
    print(f"F1 점수: {f1:.4f}")

    cm = confusion_matrix(val_y_cpu, val_pred)
    print(f"\n=== 혼동 행렬 (내부 홀드아웃) ===")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    print(f"\n하락 예측 정확도: {cm[0,0] / (cm[0,0] + cm[0,1] + 1e-8):.4f}")
    print(f"상승 예측 정확도: {cm[1,1] / (cm[1,0] + cm[1,1] + 1e-8):.4f}")

    # 저장
    MODEL_SAVE_PATH = "models/model_xgb_optimal_test"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    final_model.save_model(f"{MODEL_SAVE_PATH}/model.json")
    print(f"\n모델이 {MODEL_SAVE_PATH}에 저장되었습니다.")

    import json, joblib
    full_params = optimal_params.copy()
    full_params['look_back'] = look_back
    with open(f"{MODEL_SAVE_PATH}/optimal_params.json", 'w') as f:
        json.dump(full_params, f, indent=4)
    print(f"최적 파라미터가 {MODEL_SAVE_PATH}/optimal_params.json에 저장되었습니다.")

    joblib.dump(StandardScaler().fit(features_np), f"{MODEL_SAVE_PATH}/scaler.pkl")
    print(f"스케일러가 {MODEL_SAVE_PATH}/scaler.pkl에 저장되었습니다.")

if __name__ == "__main__":
    main()
