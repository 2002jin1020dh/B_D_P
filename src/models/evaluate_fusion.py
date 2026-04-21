"""
evaluate_fusion.py
===================
저장된 best_fusion.pt 불러와서 테스트 평가만 수행
저장 위치: src/models/evaluate_fusion.py

실행: python src/models/evaluate_fusion.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

sys.path.append('src/models')
from fusion_model import FusionModel

# ── 설정 (train_fusion.py와 동일하게 유지)
BERT_MODEL       = 'bert-base-uncased'
MAX_LEN          = 128
BATCH_SIZE       = 32
UNDERSAMPLE_SIZE = 100_000
THRESHOLD        = 0.5
MODEL_PATH       = 'outputs/checkpoints/best_fusion.pt'
DATA_PATH        = 'data/processed/yelp_preprocessed.csv'
SCALER_PATH      = 'data/features/scaler.pkl'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = [
    'rating', 'is_extreme_rating', 'rating_dev_from_product',
    'product_avg_rating', 'product_std_rating', 'product_review_count',
    'product_unique_users', 'product_rating_bias', 'product_active_span',
    'year', 'month', 'day_of_week', 'quarter', 'is_weekend', 'days_since_start',
    'token_count', 'product_monthly_burst',
]


class FusionDataset(Dataset):
    def __init__(self, texts, behavior_features, labels, tokenizer, max_len):
        self.texts             = texts
        self.behavior_features = behavior_features
        self.labels            = labels
        self.tokenizer         = tokenizer
        self.max_len           = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':         encoding['input_ids'].squeeze(0),
            'attention_mask':    encoding['attention_mask'].squeeze(0),
            'behavior_features': torch.tensor(self.behavior_features[idx], dtype=torch.float32),
            'label':             torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ── 데이터 로드 (train_fusion.py와 동일한 random_state로 동일한 test셋 재현)
print('[1/3] 테스트 데이터 로드 중...')
df = pd.read_csv(DATA_PATH, usecols=['tokens_str', 'label_binary'] + FEATURE_COLS)
df['tokens_str'] = df['tokens_str'].fillna('')
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

df_fake    = df[df['label_binary'] == 1].sample(n=UNDERSAMPLE_SIZE, random_state=42)
df_genuine = df[df['label_binary'] == 0].sample(n=UNDERSAMPLE_SIZE, random_state=42)
df = pd.concat([df_fake, df_genuine]).sample(frac=1, random_state=42).reset_index(drop=True)

texts    = df['tokens_str'].tolist()
features = df[FEATURE_COLS].values.astype(np.float32)
labels   = df['label_binary'].tolist()

idx = list(range(len(texts)))
_, tmp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
val_labels_tmp = [labels[i] for i in tmp_idx]
_, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=val_labels_tmp)

X_test_t  = [texts[i] for i in te_idx]
X_test_b  = features[te_idx]
y_test    = [labels[i] for i in te_idx]

# 정규화 (저장된 scaler 사용)
scaler   = joblib.load(SCALER_PATH)
X_test_b = scaler.transform(X_test_b)
print(f'  → 테스트 데이터: {len(y_test):,}건')

# ── 토크나이저 & DataLoader
print('[2/3] 모델 로드 중...')
tokenizer   = BertTokenizer.from_pretrained(BERT_MODEL)
test_ds     = FusionDataset(X_test_t, X_test_b, y_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 모델 로드
model = FusionModel(
    bert_model_name=BERT_MODEL,
    behavior_input_dim=len(FEATURE_COLS),
    dropout=0.3
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ── 평가
print('[3/3] 평가 중...')
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids         = batch['input_ids'].to(DEVICE)
        attention_mask    = batch['attention_mask'].to(DEVICE)
        behavior_features = batch['behavior_features'].to(DEVICE)
        logits = model(input_ids, attention_mask, behavior_features)
        probs  = torch.sigmoid(logits).cpu().numpy()
        test_preds.extend(probs)

test_preds = np.array(test_preds)
test_bin   = (test_preds >= THRESHOLD).astype(int)

f1  = f1_score(y_test, test_bin, average='macro')
auc = roc_auc_score(y_test, test_preds)

print('\n' + '=' * 55)
print('  Fusion 테스트 결과 (Yelp)')
print('=' * 55)
print(f'  Macro F1-Score : {f1:.4f}')
print(f'  AUC-ROC        : {auc:.4f}')
print('=' * 55)
print('\n  분류 리포트:')
print(classification_report(y_test, test_bin,
                            target_names=['진짜(0)', '가짜(1)']))
print('  혼동 행렬:')
print(confusion_matrix(y_test, test_bin))
print('\n' + '=' * 55)
print('  모델별 성능 비교')
print('=' * 55)
print(f'  MLP    → F1: 0.7500 / AUC: 0.8252')
print(f'  BERT   → F1: 0.8731 / AUC: 0.9360')
print(f'  Fusion → F1: {f1:.4f} / AUC: {auc:.4f}')
print('=' * 55)