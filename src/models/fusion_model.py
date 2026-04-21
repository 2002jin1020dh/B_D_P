"""
fusion_model.py
================
BERT + MLP Fusion 모델 정의
저장 위치: src/models/fusion_model.py
"""

import torch
import torch.nn as nn
from transformers import BertModel


class FusionModel(nn.Module):
    """
    BERT (텍스트) + MLP (행동 특성) Fusion 모델
    
    구조:
      BERT → [CLS] (768차원) ──┐
                                → concat (768+64) → Linear → 가짜/진짜
      MLP  → 행동 특성 (64차원) ──┘
    """
    def __init__(self, bert_model_name='bert-base-uncased',
                 behavior_input_dim=17, dropout=0.3):
        super().__init__()

        # ── BERT 인코더
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size  # 768

        # ── 행동 특성 MLP 인코더
        self.behavior_encoder = nn.Sequential(
            nn.Linear(behavior_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ── Fusion 분류 헤드
        fusion_input_dim = bert_hidden + 64   # 768 + 64 = 832
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, behavior_features):
        # BERT → [CLS] 토큰
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
        cls_output = bert_out.last_hidden_state[:, 0, :]   # (batch, 768)

        # MLP → 행동 특성 인코딩
        behavior_out = self.behavior_encoder(behavior_features)  # (batch, 64)

        # Concat → 분류
        fused = torch.cat([cls_output, behavior_out], dim=1)     # (batch, 832)
        logits = self.classifier(fused)                          # (batch, 1)
        return logits.squeeze(1)                                 # (batch,)