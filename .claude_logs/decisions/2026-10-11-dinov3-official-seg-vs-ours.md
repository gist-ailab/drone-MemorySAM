# DINOv3 공식 분할 모델 vs 우리 모델(E1 구조 + Q2 레시피) — 백본·어댑터 결선 대조 (2026-10-11)

목적: 얇은 4클래스(Pole·TrafficLight·Pedestrian·Static) 손실 원인 중 "백본에 어댑터를 붙인 방식" 쪽 결함 후보를 코드 수준에서 특정한다. 판정이 아니라 후보 목록이다. 원인 판정은 v2 재덤프 해부(thin_error_anatomy) 결과와 합쳐서 한다.

출처: DINOv3 arXiv 2508.10104 §6.3.2·App. D.10, 공식 코드 `facebookresearch/dinov3` `dinov3/eval/segmentation/models/__init__.py`·`backbone/dinov3_adapter.py`(2026-10-11 열람). 우리 쪽: `semseg/models/reliadino/encoder.py:384-415`(SimpleFPN), `model.py:48-70`(FPNSegHead), `model.py:578-623,1241-1262`(탭 결선), `m2f_head.py`, `configs/eval/q2_eval1024_yeon.yaml`.

## 1. 대조표

| 항목 | DINOv3 공식(ADE20k 63.0, 7B) | 우리(E1 구조, ViT-L) | 차이가 얇은 객체에 주는 의미 |
|---|---|---|---|
| 백본 | 동결, **완전 무수정**(injector 제거 이유 = "백본 특징을 바꾸지 않기 위해") | 동결 + 24블록 Q/V 에 센서별 LoRA r8 | 비-RGB 센서는 LoRA 필요. RGB 특징이 LoRA 로 무뎌지는지는 미측정 |
| 탭 | 4층 [5,12,18,24](1-indexed), 최종 LN + 학습 BN | 4층 [6,12,18,24], 센서별 LN + 1×1 conv(1024→256) 후 센서 합산 | 층 선택은 거의 같음. 채널을 256 으로 먼저 줄임 |
| 고해상 공간 정보 | **SPM conv 스템(RGB)** 이 stride 4/8/16/32 실제 conv 특징 생성 | **없음**. stride-4/8 레벨 = 융합 stride-16 맵의 ConvTranspose 업샘플 + 탭 bilinear 업샘플 | 우리 stride-4 특징은 전부 16px 토큰의 보간이다. 16px 보다 얇은 물체 윤곽 정보가 경로에 존재하지 않는다 |
| ViT↔고해상 결합 | **extractor**: conv 쿼리(1/8·1/16·1/32)가 각 탭 토큰을 deformable attention 으로 샘플(+ConvFFN), 마지막 블록은 extractor 3개. stride-4 = conv c1 + up(c2) + bilinear 탭 | 1×1 투영 탭을 bilinear 로 키워 **더하기** | 공식은 고해상 위치마다 "어느 토큰 의미를 가져올지"를 학습한다. 우리는 같은 토큰 값을 4×4 픽셀에 고르게 퍼뜨린다 |
| 어댑터 폭 | embed_dim(7B 4096) 유지, M2F hidden 2048 | FPN 256, M2F 256 | 우리 쪽이 훨씬 좁다 |
| 헤드 | 표준 Mask2Former(multi-scale deformable pixel decoder, 쿼리가 1/32·1/16·1/8 를 돌며 masked attention) | M2F-lite: 쿼리가 **stride-16 단일 맵**만 봄, 마스크 = embed·stride-4 FPN 특징. 최종 = SemanticFPN 식 conv 헤드 + softplus(λ)·쿼리 의미맵(arbiter, path dropout 0.25) | 쿼리가 고해상 맵을 직접 보지 않는다 |
| 마스크 손실 점 | 12544 점, 불확실 점 중요도 샘플(oversample 3, importance 0.75) | 12544 점 **균일** + 클래스 마스크당 최소 256점(POINT_QUOTA) | 경계(불확실 점)에 손실을 몰아 주지 않는다 |
| 해상도 | 학습 896(ADE 원본 약 1.75배 확대), sliding 추론; Cityscapes TTA 1.0/1.5/2.0 | 학습 crop 768(스케일 0.5~2.0, 평균 약 0.92×원본), 평가 1024(≈원본) | 공식은 사실상 패치가 원본 약 9px 를 덮는다. 우리는 16px. DINOv3 저자도 "16px 패치는 분할에 거칠다" 고 명시 |
| 입력 | RGB 1개 | RGB·depth·event·LiDAR 4개 | SPM 을 어느 센서에 걸지 설계 필요 |

## 2. 우리 어댑터 결선의 결함 후보와 수정안

1. **고해상 경로 부재(가장 유력)**: stride-4 특징에 16px 미만 정보가 없다. 수정안 = 공식과 같은 SPM + extractor(injector 없음) 를 SimpleFPN 대신 주 경로에 둔다. 영(0) 초기화 게이트 잔차로 얹지 않는다(실패-키 1). E17(stride-4/8 conv 스템 29만 파라미터를 tanh 게이트 0.1 로 FPN 에 더함)과의 차이 = (a) conv 쿼리가 deformable attention 으로 ViT 의미를 끌어온다, (b) 게이트 잔차가 아니라 주 경로, (c) 폭 256 이 아니라 수백~1024. 근거 한계: ViT-Adapter 의 합산→extractor 이득(APb 41.6→44.7)은 백본 미세조정 검출 수치이고, 원문에선 합산만으로도 +1.4 였는데 E17 은 얇은 객체 0 이었다. 따라서 "결합 방식만 바꾸면 된다"는 확신은 없다.
2. **채널 병목**: 1024→256 을 탭마다 즉시 투영. 수정안 = 어댑터 내부 폭을 512~1024 로 유지하고 헤드 직전에만 줄인다. 단독 효과 근거 없음(미확인).
3. **쿼리가 stride-16 만 봄**: 수정안 = 쿼리 cross-attn 을 1/8·1/16·1/32 순환(표준 M2F). 얇은 객체 근거 미확인.
4. **마스크 손실 균일 점 샘플**: 수정안 = 중요도 샘플(불확실 점 75%) 로 바꾼다. 코드 몇 줄, 학습 1회로 검증 가능. 근거 = Mask2Former/PointRend 관례, 얇은 클래스 이득 수치는 미확인.
5. **해상도**: 학습 없이 1280/1536/2048 평가로 먼저 확인(재덤프 진행 중). 이득이 있으면 학습 스케일 상향.
6. **RGB LoRA 가 동결 특징을 무디게 하는지**: E0 토글(LoRA delta off)은 학습된 헤드와 맞지 않아 공정 비교가 아니다. 측정하려면 "RGB LoRA 없음" 학습 1회가 필요. 우선순위 낮음.

## 3. 판정 대기 조건
- thin_error_anatomy(v2 재덤프)의 FN 경계 거리·패치 두께 분해에서 "0.5 패치 미만 두께 성분의 찾았지만 못 그린 비율"이 기준선보다 크게 나쁘면 1번(고해상 경로)이 1순위.
- 해상도 변형 덤프에서 2048 이 얇은 4클래스를 크게 올리면 1·5번이 같은 원인(토큰 해상도)을 가리킨다.
- FP 가 경계 1~3px 에 몰려 있으면 4번(경계 점 감독)·SegFix 축이 먼저.
