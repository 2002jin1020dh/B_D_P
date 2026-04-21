"""
text_encoder.py
================
BERT 기반 텍스트 인코더 + 분류 모델
저장 위치: src/models/text_encoder.py
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT 인코더 위에 분류 헤드를 붙인 모델
    - bert-base-uncased 사용
    - [CLS] 토큰 임베딩 → Dropout → Linear → 이진 분류
    """
    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # BERT 파라미터 고정 여부 (freeze_bert=True면 BERT는 학습 안 함)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)   # 이진 분류 출력
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] 토큰 임베딩 사용 (문장 전체 의미 요약)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.classifier(cls_output)             # (batch, 1)
        return logits.squeeze(1)                         # (batch,)