"""
Yelp 가짜 리뷰 탐지 - 특성 보완 스크립트
==========================================
실행 방법:
  python feature_engineering.py

입력: data/processed/yelp_preprocessed.csv
출력: data/processed/yelp_preprocessed.csv (덮어쓰기)

추가되는 컬럼:
  - label_binary
  - is_extreme_rating
  - product_monthly_burst
  - rating_dev_from_product
"""

import pandas as pd
import numpy as np

# ── 경로 설정
IN_FILE  = 'data/processed/yelp_preprocessed.csv'
OUT_FILE = 'data/processed/yelp_preprocessed.csv'

# ════════════════════════════════
# 1. 로드
# ════════════════════════════════
print('[1/3] 데이터 로드 중...')
df = pd.read_csv(IN_FILE)
df['review_date'] = pd.to_datetime(df['review_date'])
print(f'  → {df.shape[0]:,}건 / {df.shape[1]}개 컬럼')

# ════════════════════════════════
# 2. 특성 추가
# ════════════════════════════════
print('[2/3] 특성 추가 중...')

# ── label_binary
# 원본 label: 1=가짜, -1=진짜 → 0/1로 변환
df['label_binary'] = (df['label'] == 1).astype(int)
print(f'  → label_binary | 가짜(1): {df["label_binary"].sum():,}건 '
      f'({df["label_binary"].mean()*100:.1f}%) / '
      f'진짜(0): {(df["label_binary"]==0).sum():,}건 '
      f'({(df["label_binary"]==0).mean()*100:.1f}%)')

# ── is_extreme_rating
# 평점이 1점 또는 5점 (극단적 평점) → 가짜 리뷰 신호
df['is_extreme_rating'] = df['rating'].isin([1, 5]).astype(int)
print(f'  → is_extreme_rating | 극단 평점 비율: {df["is_extreme_rating"].mean()*100:.1f}%')

# ── product_monthly_burst
# 같은 상품(식당)에 같은 달에 리뷰가 얼마나 몰렸는지
# → 갑자기 리뷰가 폭발적으로 늘어나는 패턴이 가짜 리뷰의 특징
df['year_month'] = df['review_date'].dt.to_period('M').astype(str)
burst = df.groupby(['product_id', 'year_month'])['review_id'].transform('count')
df['product_monthly_burst'] = burst
df.drop(columns=['year_month'], inplace=True)
print(f'  → product_monthly_burst | 평균: {df["product_monthly_burst"].mean():.1f} / '
      f'최대: {df["product_monthly_burst"].max():,}')

# ── rating_dev_from_product
# 이 리뷰의 평점 - 해당 상품의 평균 평점
# → 평균에서 크게 벗어난 평점은 가짜 리뷰 의심
product_avg = df.groupby('product_id')['rating'].transform('mean')
df['rating_dev_from_product'] = (df['rating'] - product_avg).round(4)
print(f'  → rating_dev_from_product | 평균 편차: {df["rating_dev_from_product"].abs().mean():.3f}')

# ════════════════════════════════
# 3. 저장
# ════════════════════════════════
print('[3/3] 저장 중...')
df.to_csv(OUT_FILE, index=False)

print()
print('=' * 55)
print('  특성 보완 완료')
print('=' * 55)
print(f'  총 리뷰 수         : {df.shape[0]:>12,}')
print(f'  전체 컬럼 수       : {df.shape[1]:>12}')
print(f'  가짜 리뷰 비율     : {df["label_binary"].mean()*100:>11.1f}%')
print(f'  극단 평점 비율     : {df["is_extreme_rating"].mean()*100:>11.1f}%')
print(f'  월 최대 리뷰 폭발  : {df["product_monthly_burst"].max():>12,}건')
print('=' * 55)
print()
print('  추가된 컬럼 (4개):')
print('  - label_binary           ← 모델 타겟 (0=진짜, 1=가짜)')
print('  - is_extreme_rating      ← 가짜 리뷰 탐지 특성')
print('  - product_monthly_burst  ← 가짜 리뷰 탐지 특성')
print('  - rating_dev_from_product← 가짜 리뷰 탐지 특성')
print('=' * 55)
