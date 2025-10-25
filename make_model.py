# -*- coding: utf-8 -*-
"""
XGBoost 최적 파라미터로 테스트하는 코드
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
    # CUDA 라이브러리 테스트
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
    """파생 변수 생성"""
    # 기술적 지표 계산
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['SMA'] = calculate_sma(df['close'])
    
    # 차분값
    df['close_diff'] = df['close'].diff()
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff()
    df['volume_diff'] = df['volume'].diff()
    
    # 변화율
    df['close_change_ratio'] = df['close'].pct_change()
    df['high_change_ratio'] = df['high'].pct_change()
    df['low_change_ratio'] = df['low'].pct_change()
    df['volume_change_ratio'] = df['volume'].pct_change()
    
    # 이동평균 차이
    df['close_ma_diff'] = df['close'] - df['close'].rolling(5).mean()
    
    # 변동성 비율 (안전값 추가)
    close_mean = df['close'].rolling(10).mean()
    df['volatility_ratio'] = df['close'].rolling(10).std() / (close_mean + 1e-8)
    
    # 가격 위치 (안전값 추가)
    close_min = df['close'].rolling(20).min()
    close_max = df['close'].rolling(20).max()
    df['price_position'] = (df['close'] - close_min) / (close_max - close_min + 1e-8)
    
    # 모멘텀
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # VWAP (안전값 추가)
    volume_sum = df['volume'].rolling(20).sum()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (volume_sum + 1e-8)
    df['price_vs_vwap'] = df['close'] / (df['vwap'] + 1e-8) - 1
    
    # 계절성 변수
    if 'datetime' in df.columns:
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['datetime']).dt.day
        df['month'] = pd.to_datetime(df['datetime']).dt.month
    else:
        # datetime 컬럼이 없으면 인덱스 기반으로 생성
        df['hour'] = df.index % 24
        df['day_of_week'] = (df.index // 24) % 7
        df['day_of_month'] = (df.index // (24 * 30)) % 30
        df['month'] = (df.index // (24 * 30 * 12)) % 12
    
    # 삼각함수 변환
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def calculate_rsi(prices, window=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def calculate_obv(close, volume):
    """OBV (On-Balance Volume) 계산"""
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
    """ATR 계산"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_sma(prices, period=20):
    """SMA (Simple Moving Average) 계산"""
    return prices.rolling(window=period).mean()

def create_sequences_with_lookback(features, targets, look_back):
    """look_back을 적용한 시계열 시퀀스 생성"""
    if look_back <= 1:
        return features, targets
    
    X, y = [], []
    for i in range(look_back, len(features)):
        # 과거 N개 시점의 모든 변수를 평면화
        sequence = features[i-look_back:i].flatten()  # (look_back, n_features) -> (look_back * n_features,)
        X.append(sequence)
        y.append(targets[i])
    
    # CuPy 사용 가능하면 GPU 배열로, 아니면 NumPy 배열로
    if CUPY_AVAILABLE:
        X_result = cp.array(X)
        y_result = cp.array(y)
    else:
        X_result = np.array(X)
        y_result = np.array(y)
    
    return X_result, y_result

def main():
    """메인 함수"""
    # 최적 파라미터 설정
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
        'tree_method': 'hist',  # GPU 가속 (최신 방식)
        'device': 'cuda'  # GPU 사용
    }
    
    # look_back 파라미터 (XGBoost 전용)
    look_back = 1
    
    print("=== XGBoost 최적 파라미터 테스트 ===")
    print(f"최적 파라미터: {optimal_params}")
    
    # 데이터 로드
    CSV_FILE_PATH = r"C:\Users\INEEJI\Desktop\ske_1_dt\CC\2023317.csv"
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"데이터 로드 완료: {df.shape}")
    except FileNotFoundError:
        print(f"데이터 파일을 찾을 수 없습니다. '{CSV_FILE_PATH}' 파일이 존재하는지 확인하세요.")
        return
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return
    
    # 파생 변수 생성
    df = add_derived_features(df)
    
    # NaN/Inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # 종속변수 생성 (다음 시점 예측)
    df['returns'] = df['close'].pct_change() * 100
    df['target'] = np.where(df['returns'] > 0, 1, 0)
    df['target'] = df['target'].shift(-1)  # 다음 시점 예측
    df = df.dropna(subset=['target'])
    
    # 기본 특성 컬럼 (단일 시점)
    base_feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'close_diff', 'high_diff', 'low_diff', 'volume_diff',
        'close_change_ratio', 'high_change_ratio', 'low_change_ratio', 'volume_change_ratio',
        'close_ma_diff', 'volatility_ratio', 'price_position',
        'momentum_5', 'momentum_10', 'vwap', 'price_vs_vwap',
        'RSI', 'MACD', 'MACD_Signal', 'OBV', 'ATR', 'SMA',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    
    # 특성과 타겟 분리 - GPU 배열 사용
    if CUPY_AVAILABLE:
        # GPU 배열로 변환
        features = cp.array(df[base_feature_columns].values)
        targets = cp.array(df['target'].values)
    else:
        features = np.array(df[base_feature_columns].values)
        targets = np.array(df['target'].values)
    
    # 데이터 분할 (6:2:2) - GPU 배열로 분할
    total_size = len(features)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    val_features = features[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    test_features = features[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    # 스케일링 적용 (훈련 데이터로만 fit)
    print("특성 스케일링 적용 중...")
    scaler = StandardScaler()
    
    # GPU 데이터를 NumPy로 변환하여 스케일링
    if CUPY_AVAILABLE:
        train_features_np = cp.asnumpy(train_features)
        val_features_np = cp.asnumpy(val_features)
        test_features_np = cp.asnumpy(test_features)
    else:
        train_features_np = train_features
        val_features_np = val_features
        test_features_np = test_features
    
    # 훈련 데이터로만 스케일러 학습
    train_features_scaled = scaler.fit_transform(train_features_np)
    # 훈련된 스케일러로 검증/테스트 데이터 변환
    val_features_scaled = scaler.transform(val_features_np)
    test_features_scaled = scaler.transform(test_features_np)
    
    print(f"스케일링 전 통계 (훈련 데이터):")
    print(f"평균: {train_features_np.mean():.4f}")
    print(f"표준편차: {train_features_np.std():.4f}")
    print(f"스케일링 후 통계 (훈련 데이터):")
    print(f"평균: {train_features_scaled.mean():.4f}")
    print(f"표준편차: {train_features_scaled.std():.4f}")
    
    # 스케일링된 데이터를 GPU 배열로 변환
    if CUPY_AVAILABLE:
        train_features = cp.array(train_features_scaled)
        val_features = cp.array(val_features_scaled)
        test_features = cp.array(test_features_scaled)
    else:
        train_features = train_features_scaled
        val_features = val_features_scaled
        test_features = test_features_scaled
    
    print(f"데이터 분할: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")
    
    # 최종 모델 학습을 위해 훈련+검증 데이터 결합
    # 훈련 데이터와 검증 데이터를 합쳐서 최종 모델 학습
    combined_train_features = np.vstack([train_features, val_features])
    combined_train_targets = np.hstack([train_targets, val_targets])
    
    # 조기 종료를 위한 내부 검증 분할 (9:1)
    internal_train_size = int(0.9 * len(combined_train_features))
    internal_train_features = combined_train_features[:internal_train_size]
    internal_train_targets = combined_train_targets[:internal_train_size]
    internal_val_features = combined_train_features[internal_train_size:]
    internal_val_targets = combined_train_targets[internal_train_size:]
    
    print(f"최종 학습 데이터: {len(combined_train_features)} (훈련+검증 데이터 결합)")
    print(f"내부 훈련: {len(internal_train_features)}, 내부 검증: {len(internal_val_features)}")
    print(f"테스트 데이터: {len(test_features)}")
    
    # 최종 모델 학습 (look_back 적용)
    if look_back > 1:
        # look_back 적용된 데이터 생성
        train_X, train_y = create_sequences_with_lookback(internal_train_features, internal_train_targets, look_back)
        val_X, val_y = create_sequences_with_lookback(internal_val_features, internal_val_targets, look_back)
        test_X, test_y = create_sequences_with_lookback(test_features, test_targets, look_back)
    else:
        # look_back = 1인 경우 원본 데이터 사용
        train_X, train_y = internal_train_features, internal_train_targets
        val_X, val_y = internal_val_features, internal_val_targets
        test_X, test_y = test_features, test_targets
    
    # GPU 데이터를 NumPy로 변환
    if CUPY_AVAILABLE:
        train_X = cp.asnumpy(train_X) if hasattr(train_X, 'get') else train_X
        train_y = cp.asnumpy(train_y) if hasattr(train_y, 'get') else train_y
        val_X = cp.asnumpy(val_X) if hasattr(val_X, 'get') else val_X
        val_y = cp.asnumpy(val_y) if hasattr(val_y, 'get') else val_y
        test_X = cp.asnumpy(test_X) if hasattr(test_X, 'get') else test_X
        test_y = cp.asnumpy(test_y) if hasattr(test_y, 'get') else test_y
    
    # 최종 모델 생성 및 훈련
    final_model = xgb.XGBClassifier(**optimal_params)
    
    print("\n=== 모델 훈련 시작 ===")
    print("검증 데이터까지 포함하여 최종 모델 학습 (조기 종료 포함)")
    
    if CUPY_AVAILABLE:
        # CuPy 배열로 변환하여 GPU 장치 일치
        train_X_gpu = cp.array(train_X)
        train_y_gpu = cp.array(train_y)
        val_X_gpu = cp.array(val_X)
        val_y_gpu = cp.array(val_y)
        test_X_gpu = cp.array(test_X)
        test_y_gpu = cp.array(test_y)
        
        # XGBClassifier에 CuPy 데이터 전달
        final_model.fit(
            train_X_gpu, train_y_gpu,
            eval_set=[(val_X_gpu, val_y_gpu)],
            verbose=False
        )
        
        # CuPy 데이터로 예측 (GPU 장치 일치)
        test_predictions = final_model.predict(test_X_gpu)
        test_y_cpu = test_y
    else:
        final_model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            verbose=False
        )
        
        test_predictions = final_model.predict(test_X)
        test_y_cpu = test_y
    
    print("훈련 완료!")
    
    # 테스트 성능 평가
    print("\n=== 테스트 성능 평가 ===")
    test_accuracy = accuracy_score(test_y_cpu, test_predictions)
    test_precision = precision_score(test_y_cpu, test_predictions)
    test_recall = recall_score(test_y_cpu, test_predictions)
    test_f1 = f1_score(test_y_cpu, test_predictions)
    
    print(f"정확도: {test_accuracy:.4f}")
    print(f"정밀도: {test_precision:.4f}")
    print(f"재현율: {test_recall:.4f}")
    print(f"F1 점수: {test_f1:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(test_y_cpu, test_predictions)
    print(f"\n=== 혼동 행렬 ===")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    print(f"\n=== 클래스별 성능 ===")
    print(f"하락 예측 정확도: {cm[0,0] / (cm[0,0] + cm[0,1]):.4f}")
    print(f"상승 예측 정확도: {cm[1,1] / (cm[1,0] + cm[1,1]):.4f}")
    
    # 모델 저장 디렉토리 생성
    MODEL_SAVE_PATH = "models/model_xgb_optimal_test"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 최종 모델 저장
    final_model.save_model(f"{MODEL_SAVE_PATH}/model.json")
    print(f"\n모델이 {MODEL_SAVE_PATH}에 저장되었습니다.")
    
    # 파라미터 저장
    import json
    # look_back을 포함한 전체 파라미터 저장
    full_params = optimal_params.copy()
    full_params['look_back'] = look_back
    with open(f"{MODEL_SAVE_PATH}/optimal_params.json", 'w') as f:
        json.dump(full_params, f, indent=4)
    print(f"최적 파라미터가 {MODEL_SAVE_PATH}/optimal_params.json에 저장되었습니다.")
    
    # 스케일러 저장
    import joblib
    joblib.dump(scaler, f"{MODEL_SAVE_PATH}/scaler.pkl")
    print(f"스케일러가 {MODEL_SAVE_PATH}/scaler.pkl에 저장되었습니다.")

if __name__ == "__main__":
    main()
