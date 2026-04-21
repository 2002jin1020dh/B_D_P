"""
behavior_mlp.py
================
행동 특성 기반 MLP 모델 학습 및 평가
저장 위치: src/models/behavior_mlp.py

실행 방법:
  1. python src/data/dataset.py     (데이터 분할 먼저)
  2. python src/models/behavior_mlp.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import os

# ── 경로 설정
FEATURE_DIR  = 'data/features'
MODEL_DIR    = 'outputs/checkpoints'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 하이퍼파라미터
INPUT_DIM    = 17       # FEATURE_COLS 개수
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3
BATCH_SIZE   = 2048
EPOCHS       = 100
LR           = 1e-3
THRESHOLD    = 0.46      # 이진 분류 임계값
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════
# 1. MLP 모델 정의
# ════════════════════════════════
class BehaviorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))   # 이진 분류 출력
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)            # (batch,) 형태 출력


# ════════════════════════════════
# 2. 데이터 로드
# ════════════════════════════════
def load_data():
    print('[1/3] 데이터 로드 중...')
    X_train = np.load(f'{FEATURE_DIR}/X_train.npy')
    X_val   = np.load(f'{FEATURE_DIR}/X_val.npy')
    X_test  = np.load(f'{FEATURE_DIR}/X_test.npy')
    y_train = np.load(f'{FEATURE_DIR}/y_train.npy')
    y_val   = np.load(f'{FEATURE_DIR}/y_val.npy')
    y_test  = np.load(f'{FEATURE_DIR}/y_test.npy')
    print(f'  → train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}')
    return X_train, X_val, X_test, y_train, y_val, y_test


# ════════════════════════════════
# 3. 학습
# ════════════════════════════════
def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_f1    = 0.0
    best_epoch = 0

    print(f'\n[2/3] 학습 시작 (device: {DEVICE})')
    print(f'  Epochs: {EPOCHS} / Batch: {BATCH_SIZE} / LR: {LR} / Threshold: {THRESHOLD}')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(DEVICE))
                probs  = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(y_batch.numpy())

        val_preds  = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_bin    = (val_preds >= THRESHOLD).astype(int)
        val_f1     = f1_score(val_labels, val_bin, average='macro')
        val_auc    = roc_auc_score(val_labels, val_preds)

        scheduler.step(val_f1)

        # 최고 F1 모델 저장
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f'{MODEL_DIR}/best_mlp.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:02d} | '
                  f'Loss: {train_loss/len(train_loader):.4f} | '
                  f'Val F1: {val_f1:.4f} | '
                  f'Val AUC: {val_auc:.4f} '
                  f'{"★ best" if epoch == best_epoch else ""}')

    print(f'\n  최고 Val F1: {best_f1:.4f} (Epoch {best_epoch})')
    return best_f1


# ════════════════════════════════
# 5. 평가
# ════════════════════════════════
def evaluate(model, test_loader, y_test):
    print('\n[3/3] 테스트 평가 중...')
    model.load_state_dict(torch.load(f'{MODEL_DIR}/best_mlp.pt', map_location=DEVICE))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)

    test_preds = np.array(test_preds)
    test_bin   = (test_preds >= THRESHOLD).astype(int)

    f1  = f1_score(y_test, test_bin, average='macro')
    auc = roc_auc_score(y_test, test_preds)

    print('\n' + '=' * 55)
    print('  최종 테스트 결과')
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
# 6. 메인
# ════════════════════════════════
if __name__ == '__main__':
    # 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # DataLoader 생성
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # 모델 초기화
    model = BehaviorMLP(INPUT_DIM, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    print(f'\n  모델 구조:')
    print(model)

    # 학습
    train(model, train_loader, val_loader)

    # 평가
    evaluate(model, test_loader, y_test)