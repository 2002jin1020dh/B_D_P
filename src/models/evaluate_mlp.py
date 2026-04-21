"""
evaluate_mlp.py
================
저장된 best_mlp.pt 불러와서 테스트 평가만 수행
저장 위치: src/models/evaluate_mlp.py

실행: python src/models/evaluate_mlp.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix

# ── 설정 (behavior_mlp.py와 동일하게 맞춰야 함)
FEATURE_DIR = 'data/features'
MODEL_PATH  = 'outputs/checkpoints/best_mlp.pt'
INPUT_DIM   = 17
HIDDEN_DIMS = [256, 128, 64]
DROPOUT     = 0.3
BATCH_SIZE  = 2048
THRESHOLD   = 0.5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── 데이터 로드
print('[1/2] 테스트 데이터 로드 중...')
X_test = np.load(f'{FEATURE_DIR}/X_test.npy')
y_test = np.load(f'{FEATURE_DIR}/y_test.npy')
print(f'  → {len(X_test):,}건')

test_ds     = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ── 모델 로드 및 평가
print('[2/2] 모델 평가 중...')
model = BehaviorMLP(INPUT_DIM, HIDDEN_DIMS, DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        logits = model(X_batch.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs)

preds    = np.array(preds)
preds_bin = (preds >= THRESHOLD).astype(int)

f1  = f1_score(y_test, preds_bin, average='macro')
auc = roc_auc_score(y_test, preds)

print('\n' + '=' * 55)
print('  MLP 테스트 결과 (Yelp)')
print('=' * 55)
print(f'  Macro F1-Score : {f1:.4f}')
print(f'  AUC-ROC        : {auc:.4f}')
print('=' * 55)
print('\n  분류 리포트:')
print(classification_report(y_test, preds_bin,
                            target_names=['진짜(0)', '가짜(1)']))
print('  혼동 행렬:')
print(confusion_matrix(y_test, preds_bin))