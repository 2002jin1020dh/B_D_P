"""
dataset.py
===========
데이터 로드 및 train / val / test 분할
저장 위치: src/data/dataset.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── 경로 설정
DATA_PATH   = 'data/processed/yelp_preprocessed.csv'
OUT_DIR     = 'data/features'
SCALER_PATH = 'data/features/scaler.pkl'
os.makedirs(OUT_DIR, exist_ok=True)

# ── MLP에 사용할 수치형 특성
FEATURE_COLS = [
    'rating', 'is_extreme_rating', 'rating_dev_from_product',
    'product_avg_rating', 'product_std_rating', 'product_review_count',
    'product_unique_users', 'product_rating_bias', 'product_active_span',
    'year', 'month', 'day_of_week', 'quarter', 'is_weekend', 'days_since_start',
    'token_count', 'product_monthly_burst',
]

TARGET_COL = 'label_binary'   # 0=진짜, 1=가짜

# ── 언더샘플링 설정
UNDERSAMPLE      = True
UNDERSAMPLE_SIZE = 100_000     # 가짜/진짜 각각 10만건


def load_and_split(test_size=0.1, val_size=0.1, random_state=42):
    print('[1/5] 데이터 로드 중...')
    df = pd.read_csv(DATA_PATH, usecols=FEATURE_COLS + [TARGET_COL])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    print(f'  → {df.shape[0]:,}건 / 특성 {len(FEATURE_COLS)}개')
    print(f'  → 가짜(1): {df[TARGET_COL].sum():,}건 / 진짜(0): {(df[TARGET_COL]==0).sum():,}건')

    # ── 언더샘플링 (가짜/진짜 각각 10만건)
    if UNDERSAMPLE:
        print(f'\n[2/5] 언더샘플링 중...')
        df_fake    = df[df[TARGET_COL] == 1].sample(n=UNDERSAMPLE_SIZE, random_state=random_state)
        df_genuine = df[df[TARGET_COL] == 0].sample(n=UNDERSAMPLE_SIZE, random_state=random_state)
        df = pd.concat([df_fake, df_genuine]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f'  → 가짜(1): {(df[TARGET_COL]==1).sum():,}건')
        print(f'  → 진짜(0): {(df[TARGET_COL]==0).sum():,}건')
        print(f'  → 합계    : {len(df):,}건')
    else:
        print('[2/5] 언더샘플링 스킵')

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # ── 분할
    print('\n[3/5] 데이터 분할 중...')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state, stratify=y_temp
    )
    print(f'  → train : {len(X_train):,}건')
    print(f'  → val   : {len(X_val):,}건')
    print(f'  → test  : {len(X_test):,}건')

    # ── 정규화
    print('\n[4/5] 정규화 중...')
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f'  → scaler 저장: {SCALER_PATH}')

    # ── 저장
    print('\n[5/5] 분할 데이터 저장 중...')
    for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        np.save(f'{OUT_DIR}/{name}.npy', arr)
        print(f'  → {name}.npy 저장')

    print('\n  완료!')
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    load_and_split()