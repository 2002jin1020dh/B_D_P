# 가짜 리뷰 탐지 모델 (Yelp Dataset)

Yelp 리뷰 데이터셋을 활용한 가짜 리뷰 탐지 프로젝트입니다.  
MLP(행동 특성), BERT(텍스트), Fusion(결합) 세 가지 모델을 비교 실험했습니다.

## 📊 성능 보고서

> 모델별 성능 비교, 학습 이력, 혼동 행렬을 인터랙티브 UI로 확인하세요.

**[→ 성능 보고서 보기](https://2002jin1020dh.github.io/B_D_P/)**

## 결과 요약

| 모델 | Macro F1 | AUC-ROC |
|------|----------|---------|
| MLP (Baseline) | 0.7513 | 0.8298 |
| BERT | 0.8731 | 0.9360 |
| **Fusion (Best)** | **0.8739** | **0.9411** |

## 모델 구조

- **MLP** : 행동 특성 17개 → 256 → 128 → 64 → 1
- **BERT** : `bert-base-uncased` 파인튜닝 → Linear(768→1)
- **Fusion** : BERT [CLS](768) + MLP 행동 인코더(64) → concat → Linear(832→1)

## 데이터셋

**Yelp**
- 규모 : 3,540,818건 / 923개 식당
- 클래스 불균형 : 가짜 97.1% / 진짜 2.9% → 언더샘플링(각 10만건) 적용
- 파일 : Yelp dataset.xlsx + Yelp Metadata.xlsx 병합

**Amazon Review** (예정)
- 규모 : 21,000건 / 30개 도메인 (PC, Baby, Books 등)
- 클래스 균형 : 50% / 50%
- 목적 : 다중 도메인 일반화 성능 검증용

## 전처리

- NLTK RegexpTokenizer 기반 텍스트 정제 및 토큰화
- 시간 파생 특성, 상품 단위 통계 특성 생성
- 가짜 리뷰 탐지 특화 특성 17개 추출

## 실행 순서

```bash
pip install -r requirements.txt

python src/data/preprocess.py          # 전처리
python src/data/feature_engineering.py # 특성 생성
python src/models/behavior_mlp.py      # MLP 학습
python src/models/train_bert.py        # BERT 학습
python src/models/train_fusion.py      # Fusion 학습
python src/models/evaluate_fusion.py   # 최종 평가
```

## 프로젝트 구조

```
yelp-rating-model/
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── feature_engineering.py
│   │   └── dataset.py
│   └── models/
│       ├── fusion_model.py       # Fusion 모델 정의
│       ├── text_encoder.py       # BERT 분류기
│       ├── behavior_mlp.py       # MLP 학습
│       ├── train_bert.py         # BERT 학습
│       ├── train_fusion.py       # Fusion 학습
│       └── evaluate_fusion.py    # 최종 평가
├── data/
│   ├── processed/
│   └── features/
├── outputs/
│   └── checkpoints/
├── requirements.txt
└── index.html                    # 성능 보고서 (GitHub Pages)
```
