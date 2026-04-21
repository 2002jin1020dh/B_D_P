"""
Yelp 리뷰 데이터 전처리 스크립트 (NLTK)
실행: python preprocess.py
"""

import re, gc
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# NLTK 리소스 다운로드
for r in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(r, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
TOKENIZER  = RegexpTokenizer(r'[a-z]{2,}')  # 2글자 이상 영문만

# ── 경로 설정 (필요 시 수정)
REVIEW_FILE = 'data/raw/Yelp dataset.xlsx'
META_FILE   = 'data/raw/Yelp Metadata.xlsx'
OUT_DIR     = 'data/processed'

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════
# 1. 두 파일 로드 → 데이터프레임 하나로 병합
# ════════════════════════════════
print('[1/5] 두 파일 로드 및 병합...')
df_review = pd.read_excel(REVIEW_FILE)
df_meta   = pd.read_excel(META_FILE)

# Review_id 기준으로 inner join → 컬럼 정리
df = pd.merge(df_review, df_meta[['Review_id', 'Rating', 'Label']], on='Review_id', how='inner')
del df_review, df_meta
gc.collect()

df.columns = ['review_id', 'product_id', 'review_date', 'review_text', 'rating', 'label']
df['review_date'] = pd.to_datetime(df['review_date'])

print(f'  → {df.shape[0]:,}건 / 컬럼: {df.columns.tolist()}')

# ════════════════════════════════
# 2. 텍스트 정제 및 NLTK 토큰화
# ════════════════════════════════
print('[2/5] 텍스트 정제 및 토큰화...')

def process(text):
    if not isinstance(text, str) or not text.strip():
        return '', '', 0
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # URL 제거
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = TOKENIZER.tokenize(text)                    # 영문 토큰 추출
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens   # 표제어 추출
              if t not in STOP_WORDS]                    # 불용어 제거
    return text, ' '.join(tokens), len(tokens)

results = [process(t) for t in tqdm(df['review_text'], desc='  토큰화')]
c, ts, tc = zip(*results)
df['clean_text']  = c    # 정제된 텍스트
df['tokens_str']  = ts   # 토큰 문자열 (모델 입력용)
df['token_count'] = tc   # 토큰 수
del results; gc.collect()

print(f'  → 평균 토큰 수: {df["token_count"].mean():.1f}')

# ════════════════════════════════
# 3. 시간 파생 특성
# ════════════════════════════════
print('[3/5] 시간 파생 특성 생성...')
ref = df['review_date'].min()
df['year']             = df['review_date'].dt.year
df['month']            = df['review_date'].dt.month
df['day_of_week']      = df['review_date'].dt.dayofweek  # 0=월 ~ 6=일
df['quarter']          = df['review_date'].dt.quarter
df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['days_since_start'] = (df['review_date'] - ref).dt.days

# ════════════════════════════════
# 4. 사용자 / 상품 단위 통계 특성
# ════════════════════════════════
print('[4/5] 통계 특성 구축...')
global_avg = df['rating'].mean()

# 사용자 단위 (review_id = user_id 대리)
user_stats = df.groupby('review_id').agg(
    user_review_count    = ('review_id',       'count'),
    user_avg_rating      = ('rating',           'mean'),
    user_std_rating      = ('rating',           'std'),
    user_min_rating      = ('rating',           'min'),
    user_max_rating      = ('rating',           'max'),
    user_unique_products = ('product_id',       'nunique'),
    user_avg_token_count = ('token_count',      'mean'),
    user_first_day       = ('days_since_start', 'min'),
    user_last_day        = ('days_since_start', 'max'),
).reset_index().rename(columns={'review_id': 'user_id'})

user_stats['user_active_span'] = user_stats['user_last_day'] - user_stats['user_first_day']
user_stats['user_review_freq'] = (user_stats['user_review_count']
                                   / user_stats['user_active_span'].replace(0, 1))
user_stats['user_rating_bias'] = user_stats['user_avg_rating'] - global_avg
user_stats['user_std_rating']  = user_stats['user_std_rating'].fillna(0)

# 상품 단위
product_stats = df.groupby('product_id').agg(
    product_review_count    = ('review_id',       'count'),
    product_avg_rating      = ('rating',           'mean'),
    product_std_rating      = ('rating',           'std'),
    product_min_rating      = ('rating',           'min'),
    product_max_rating      = ('rating',           'max'),
    product_unique_users    = ('review_id',        'nunique'),
    product_avg_token_count = ('token_count',      'mean'),
    product_first_day       = ('days_since_start', 'min'),
    product_last_day        = ('days_since_start', 'max'),
).reset_index()

product_stats['product_active_span'] = product_stats['product_last_day'] - product_stats['product_first_day']
product_stats['product_rating_bias'] = product_stats['product_avg_rating'] - global_avg
product_stats['product_std_rating']  = product_stats['product_std_rating'].fillna(0)

# 메인 DF에 병합
df['user_id'] = df['review_id']
df = df.merge(user_stats.rename(columns={'user_id': 'review_id'}), on='review_id', how='left')
df = df.merge(product_stats, on='product_id', how='left')

# 리뷰 레벨 편차 특성
df['rating_dev_from_user']    = df['rating'] - df['user_avg_rating']
df['rating_dev_from_product'] = df['rating'] - df['product_avg_rating']
df['rating_category'] = pd.cut(df['rating'], bins=[0,2,3,5],
                                labels=['negative', 'neutral', 'positive'])

# ════════════════════════════════
# 5. 저장
# ════════════════════════════════
print('[5/5] 저장...')

df.drop(columns=['review_text']).to_csv(f'{OUT_DIR}/yelp_preprocessed.csv', index=False)
user_stats.to_csv(f'{OUT_DIR}/user_stats.csv', index=False)
product_stats.to_csv(f'{OUT_DIR}/product_stats.csv', index=False)

print()
print('=' * 50)
print('  전처리 완료 요약')
print('=' * 50)
print(f'  총 리뷰 수        : {df.shape[0]:>10,}')
print(f'  고유 상품 수      : {df["product_id"].nunique():>10,}')
print(f'  평균 평점         : {df["rating"].mean():>10.2f}')
print(f'  평균 토큰 수      : {df["token_count"].mean():>10.1f}')
print(f'  긍정(4~5) 비율    : {(df["rating"] >= 4).mean()*100:>9.1f}%')
print(f'  부정(1~2) 비율    : {(df["rating"] <= 2).mean()*100:>9.1f}%')
print(f'  전체 컬럼 수      : {df.shape[1]:>10}')
print('=' * 50)
print(f'  저장 위치: {OUT_DIR}/')
print('  - yelp_preprocessed.csv  ← 병합 + 전처리 완료 데이터')
print('  - user_stats.csv         ← 사용자 단위 통계')
print('  - product_stats.csv      ← 상품 단위 통계')
print('=' * 50)
