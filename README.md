목적: 리뷰 데이터에서 가짜/진짜 리뷰 이진 분류
타겟: label_binary (0=진짜, 1=가짜)

<데이터셋>

scale : 3,540,818건 / 923개 식당
클래스 불균형 심함 : 가짜 97.1% / 진짜 2.9%
파일: Yelp dataset.xlsx + Yelp Metadata.xlsx → 병합 후 사용

amazon_review (테스트 대상)
scale : 21,000건
클래스 불균형 x : 50% / 50%
도메인 수 : 30개(PC, Baby, Books)
각 도메인 별 리뷰 수 : 각 700건 씩

특정 도메인이 아니라 여러 도메인에서도 유의미한 결과를 도출할 수 있는 모델을 만드는 게 목표 -> 
현재는 Yelp 데이터 셋만 가지고 있으므로 검증 불가 -> amazon review 데이터 셋으로 테스트

<전처리>

NLTK RegexpTokenizer 기반 텍스트 정제 및 토큰화
시간 파생 특성, 상품 단위 통계 특성 생성
가짜 리뷰 탐지 특화 특성 추가

<모델 구조>

MLP: 행동 특성 17개 기반 이진 분류
BERT/RoBERTa: 텍스트 임베딩 (BERT 수행)
Fusion: MLP + BERT 결합 

<클래스 불균형 대응 : Yelp>

BCEWithLogitsLoss pos_weight 적용 
임계값 0.5으로 조정
평가지표: Macro F1-Score + AUC-ROC

-> 언더 샘플링으로 10만 건 : 10만 건 불균형 해소 함
