"""
train_bert.py
==============
BERT 기반 가짜 리뷰 탐지 모델 학습 및 평가
저장 위치: src/models/train_bert.py

실행: python src/models/train_bert.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from text_encoder import BertClassifier

# ── 경로 설정
DATA_PATH  = 'data/processed/yelp_preprocessed.csv'
MODEL_DIR  = 'outputs/checkpoints'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 하이퍼파라미터
BERT_MODEL       = 'bert-base-uncased'
MAX_LEN          = 128       # 토큰 최대 길이 (리뷰 평균 76토큰이라 128이면 충분)
BATCH_SIZE       = 32        # 4070 16GB 기준 적정값
EPOCHS           = 5         # BERT는 3~5 epoch으로도 충분
LR               = 2e-5      # BERT 파인튜닝 권장 학습률
UNDERSAMPLE_SIZE = 100_000   # 가짜/진짜 각각 10만건
THRESHOLD        = 0.5
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════
# 1. Dataset 클래스
# ════════════════════════════════
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ════════════════════════════════
# 2. 데이터 로드 및 분할
# ════════════════════════════════
def load_data():
    print('[1/5] 데이터 로드 중...')
    df = pd.read_csv(DATA_PATH, usecols=['tokens_str', 'label_binary'])
    df['tokens_str'] = df['tokens_str'].fillna('')
    print(f'  → 전체: {len(df):,}건')

    # 언더샘플링 (가짜/진짜 각각 10만건)
    df_fake    = df[df['label_binary'] == 1].sample(n=UNDERSAMPLE_SIZE, random_state=42)
    df_genuine = df[df['label_binary'] == 0].sample(n=UNDERSAMPLE_SIZE, random_state=42)
    df = pd.concat([df_fake, df_genuine]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'  → 언더샘플링 후: {len(df):,}건 (가짜 {UNDERSAMPLE_SIZE:,} / 진짜 {UNDERSAMPLE_SIZE:,})')

    texts  = df['tokens_str'].tolist()
    labels = df['label_binary'].tolist()

    # train 80% / val 10% / test 10%
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f'  → train: {len(X_train):,} / val: {len(X_val):,} / test: {len(X_test):,}')
    return X_train, X_val, X_test, y_train, y_val, y_test


# ════════════════════════════════
# 3. 학습
# ════════════════════════════════
def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS
    )

    best_f1    = 0.0
    best_epoch = 0

    print(f'\n[3/5] 학습 시작 (device: {DEVICE})')
    print(f'  Epochs: {EPOCHS} / Batch: {BATCH_SIZE} / LR: {LR}')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                logits = model(input_ids, attention_mask)
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
            torch.save(model.state_dict(), f'{MODEL_DIR}/best_bert.pt')

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
    model.load_state_dict(torch.load(f'{MODEL_DIR}/best_bert.pt', map_location=DEVICE))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)

    test_preds = np.array(test_preds)
    test_bin   = (test_preds >= THRESHOLD).astype(int)

    f1  = f1_score(y_test, test_bin, average='macro')
    auc = roc_auc_score(y_test, test_preds)

    print('\n' + '=' * 55)
    print('  BERT 테스트 결과 (Yelp)')
    print('=' * 55)
    print(f'  Macro F1-Score : {f1:.4f}')
    print(f'  AUC-ROC        : {auc:.4f}')
    print('=' * 55)
    print('\n  분류 리포트:')
    print(classification_report(y_test, test_bin,
                                target_names=['진짜(0)', '가짜(1)']))
    print('  혼동 행렬:')
    print(confusion_matrix(y_test, test_bin))


# ════════════════════════════════
# 5. 메인
# ════════════════════════════════
if __name__ == '__main__':
    # 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 토크나이저
    print('\n[2/5] 토크나이저 로드 중...')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # DataLoader
    def make_loader(texts, labels, shuffle=False):
        ds = ReviewDataset(texts, labels, tokenizer, MAX_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # 모델
    print('[4/5] 모델 초기화 중...')
    model = BertClassifier(BERT_MODEL, dropout=0.3).to(DEVICE)

    # 학습
    train(model, train_loader, val_loader)

    # 평가
    evaluate(model, test_loader, y_test)