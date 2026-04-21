"""
train_fusion.py
================
BERT + MLP Fusion 모델 학습 및 평가
저장 위치: src/models/train_fusion.py

실행: python src/models/train_fusion.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

sys.path.append('src/models')
from fusion_model import FusionModel

# ── 경로 설정
DATA_PATH  = 'data/processed/yelp_preprocessed.csv'
SCALER_PATH = 'data/features/scaler.pkl'
MODEL_DIR  = 'outputs/checkpoints'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 하이퍼파라미터
BERT_MODEL       = 'bert-base-uncased'
MAX_LEN          = 128
BATCH_SIZE       = 32
EPOCHS           = 5
LR               = 2e-5
UNDERSAMPLE_SIZE = 100_000
THRESHOLD        = 0.5
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── MLP 특성 컬럼 (dataset.py와 동일하게 유지)
FEATURE_COLS = [
    'rating', 'is_extreme_rating', 'rating_dev_from_product',
    'product_avg_rating', 'product_std_rating', 'product_review_count',
    'product_unique_users', 'product_rating_bias', 'product_active_span',
    'year', 'month', 'day_of_week', 'quarter', 'is_weekend', 'days_since_start',
    'token_count', 'product_monthly_burst',
]


# ════════════════════════════════
# 1. Dataset 클래스
# ════════════════════════════════
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


# ════════════════════════════════
# 2. 데이터 로드 및 분할
# ════════════════════════════════
def load_data():
    print('[1/5] 데이터 로드 중...')
    df = pd.read_csv(DATA_PATH, usecols=['tokens_str', 'label_binary'] + FEATURE_COLS)
    df['tokens_str']   = df['tokens_str'].fillna('')
    df[FEATURE_COLS]   = df[FEATURE_COLS].fillna(0)
    print(f'  → 전체: {len(df):,}건')

    # 언더샘플링
    df_fake    = df[df['label_binary'] == 1].sample(n=UNDERSAMPLE_SIZE, random_state=42)
    df_genuine = df[df['label_binary'] == 0].sample(n=UNDERSAMPLE_SIZE, random_state=42)
    df = pd.concat([df_fake, df_genuine]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'  → 언더샘플링 후: {len(df):,}건 (가짜 {UNDERSAMPLE_SIZE:,} / 진짜 {UNDERSAMPLE_SIZE:,})')

    texts    = df['tokens_str'].tolist()
    features = df[FEATURE_COLS].values.astype(np.float32)
    labels   = df['label_binary'].tolist()

    # train / val / test 분할
    idx = list(range(len(texts)))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    val_labels_tmp  = [labels[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42,
                                       stratify=val_labels_tmp)

    def subset(idx_list):
        return ([texts[i] for i in idx_list],
                features[idx_list],
                [labels[i] for i in idx_list])

    X_train_t, X_train_b, y_train = subset(tr_idx)
    X_val_t,   X_val_b,   y_val   = subset(val_idx)
    X_test_t,  X_test_b,  y_test  = subset(te_idx)

    # 행동 특성 정규화
    scaler    = StandardScaler()
    X_train_b = scaler.fit_transform(X_train_b)
    X_val_b   = scaler.transform(X_val_b)
    X_test_b  = scaler.transform(X_test_b)
    joblib.dump(scaler, SCALER_PATH)

    print(f'  → train: {len(y_train):,} / val: {len(y_val):,} / test: {len(y_test):,}')
    return (X_train_t, X_train_b, y_train,
            X_val_t,   X_val_b,   y_val,
            X_test_t,  X_test_b,  y_test)


# ════════════════════════════════
# 3. 학습
# ════════════════════════════════
def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS
    )

    best_f1 = 0.0
    best_epoch = 0

    print(f'\n[3/5] 학습 시작 (device: {DEVICE})')
    print(f'  Epochs: {EPOCHS} / Batch: {BATCH_SIZE} / LR: {LR}')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids         = batch['input_ids'].to(DEVICE)
            attention_mask    = batch['attention_mask'].to(DEVICE)
            behavior_features = batch['behavior_features'].to(DEVICE)
            labels            = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, behavior_features)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids         = batch['input_ids'].to(DEVICE)
                attention_mask    = batch['attention_mask'].to(DEVICE)
                behavior_features = batch['behavior_features'].to(DEVICE)
                logits = model(input_ids, attention_mask, behavior_features)
                probs  = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['label'].numpy())

        val_preds  = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_bin    = (val_preds >= THRESHOLD).astype(int)
        val_f1     = f1_score(val_labels, val_bin, average='macro')
        val_auc    = roc_auc_score(val_labels, val_preds)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f'{MODEL_DIR}/best_fusion.pt')

        print(f'  Epoch {epoch:02d} | '
              f'Loss: {train_loss/len(train_loader):.4f} | '
              f'Val F1: {val_f1:.4f} | '
              f'Val AUC: {val_auc:.4f} '
              f'{"★ best" if epoch == best_epoch else ""}')

    print(f'\n  최고 Val F1: {best_f1:.4f} (Epoch {best_epoch})')


# ════════════════════════════════
# 4. 평가
# ════════════════════════════════
def evaluate(model, test_loader, y_test):
    print('\n[5/5] 테스트 평가 중...')
    model.load_state_dict(torch.load(f'{MODEL_DIR}/best_fusion.pt', map_location=DEVICE))
    model.eval()

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

    # ── 최종 비교 요약
    print('\n' + '=' * 55)
    print('  모델별 성능 비교')
    print('=' * 55)
    print(f'  MLP    → F1: 0.7500 / AUC: 0.8252')
    print(f'  BERT   → F1: 0.8731 / AUC: 0.9360')
    print(f'  Fusion → F1: {f1:.4f} / AUC: {auc:.4f}')
    print('=' * 55)


# ════════════════════════════════
# 5. 메인
# ════════════════════════════════
if __name__ == '__main__':
    (X_train_t, X_train_b, y_train,
     X_val_t,   X_val_b,   y_val,
     X_test_t,  X_test_b,  y_test) = load_data()

    print('\n[2/5] 토크나이저 로드 중...')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def make_loader(texts, behaviors, labels, shuffle=False):
        ds = FusionDataset(texts, behaviors, labels, tokenizer, MAX_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4)

    train_loader = make_loader(X_train_t, X_train_b, y_train, shuffle=True)
    val_loader   = make_loader(X_val_t,   X_val_b,   y_val)
    test_loader  = make_loader(X_test_t,  X_test_b,  y_test)

    print('[4/5] Fusion 모델 초기화 중...')
    model = FusionModel(
        bert_model_name=BERT_MODEL,
        behavior_input_dim=len(FEATURE_COLS),
        dropout=0.3
    ).to(DEVICE)

    train(model, train_loader, val_loader)
    evaluate(model, test_loader, y_test)