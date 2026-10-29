<!-- 생성물: 손편집 금지. lab-plan render로 다시 만든다. -->

# 실험 계획 · DRN

## running
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-261019-02 | P1 | MUSES 4모달 A2 — E1 구조 + Q2 레시피(두 패스+E1M4 교사 증류) + 고해상 경로, 40ep 스크린 s2·s3 | 4모달 E1M 스크린에 TRAIN.QAF(Q2 레시피) + MODEL.HIRES_ADAPTER | - | muses, 4modal, A2 |
| DRN-261020-02 | P1 | 노벨티 감사 + 구조 후보 수색(문헌만): Q2 레시피 최근접 선행과 차별점 · '세부-의미 결합' 실패 기제에 대한 2025~2026 대응 구조 | 문헌 조사만. 산출 = research/ 아래 감사 문서: 표1(Q2 vs 선행 4편: 교사·패스 수·열화 정의·라벨 필요 여부·게이트 유무), 표2(세부-의미 결합 구조 후보 ≥5, 각 arXiv ID·우리 기각 축(게이트·attn bias·라우팅·합성 열화 라벨) 저촉 여부·DELIVER/MUSES 적용 가능성) | - | novelty-audit, survey, structural-novelty |
| DRN-261027-01 | P1 | IEEE Access 원고 v1 — 분석형+성능 틀, 10쪽, 초고 동결 2026-11-10 (2026-10-29 IEEE Sensors Journal에서 변경) | 집필 작업(학습 없음). 브리프 .claude_logs/paper/00_brief.md 의 절 구성·표 출처·인용 규칙을 따른다 | - | paper, ieee-sensors, writing |
| DRN-261027-03 | P1 | MCubeS 로더 학습 전용 크롭 옵션(DATASET.TRAIN_CROP, 기본 1024) + 평가 하네스 재동결 — A2 를 24GB 카드에서 돌리기 위함 | semseg/datasets/mcubes.py __init__ 에 train_crop/train_base 인자(기본 1024), train_reliadino.py ds_kwargs 에 DATASET.TRAIN_CROP 전달, 스모크 tools/smoke_mcubes_train_crop.py. 이후 tools/eval_harness_guard.py --freeze 재동결 + 대표 ckpt(P39.1 MCubeS) 재채점으로 수치 불변 확인 | - | mcubes, harness, A2, build |
| DRN-261016-02 | P2 | MUSES 평가 해상도 반응 — PhysAug-off 3시드 공식 val @1024/1536/2048 (레터박스 0.53배 → 기준선 0.95배 정렬) |  | - | muses, resolution |
| DRN-261016-03 | P2 | 평가 해상도 상향이 왜 우리 모델만 크게 올리나 — 클래스·두께·조건 분해(DELIVER 우리 vs DGFusion, MUSES 1024/1536/2048) |  | - | deliver, muses, resolution, analysis |
| DRN-261020-14 | P2 | Ablation L6 · A2 − 교사 증류(→ 열화 두 패스만, KD_W 0) | TRAIN.QAF.KD_W 0, 그 외 A2 동일 | - | ablation, leave-one-out, paper |
| DRN-260926-54 | P101 | Q0 · E1 확정 강건성 전 프로토콜(EMM 15조합·RMM 4부분집합×3비율·NM) | E1 확정 시드821·902·903 강건성 측정(학습 0) | - | deliver, robustness, p54 |
| DRN-260926-55 | P102 | Q2→Q3 강건성 전 프로토콜(EMM15·RMM4×3·NM3, Q2 먼저) | Q2·Q3 체크포인트 강건성 측정 | - | deliver, robustness, p54 |
| DRN-260926-56 | P103 | Q2noDeg · 원천분리: DEGRADE_P=0, KD 유지(증류만의 순기여) | DEGRADE_P=0 (KD 유지) | - | deliver, p54, screen40 |
| DRN-260926-233 | P369 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed2026090{2,3}_screen40_Q2 | Q2 시드 재현(교사 고정, 학생 시드만 변경) | - | registry, deliver, seg |

## evaluating
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-261019-01 | P1 | MUSES 4모달 교사 E1M4 — 4탭 clean 200ep(+radar), A2-4모달의 동결 교사 | E1M full200 에 radar 추가(DATASET.MODALS) | - | muses, 4modal, teacher, A2 |
| DRN-261020-13 | P2 | Ablation L5 · A2 − 열화 패스(→ clean 패스 + 동결 E1 교사 증류만, DEGRADE_P 0) | TRAIN.QAF.DEGRADE_P 0 · MISSING_FRAC 0 (KD 유지), 그 외 A2 동일 | yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L5nodegrade_20261022_1920 | ablation, leave-one-out, paper |
| DRN-261020-07 | P3 | 백본 대조 1 · '우리 방법을 뺀 DINOv3': 동결 DINOv3-L + 센서별 LoRA + 단순 융합(센서 토큰 평균, 교차주의 없음) + 같은 헤드, clean 40ep 2시드(s821·s902) | 기준 config = E1 스크린(bengio-deliver_rgbdel_P46_c3only_seed2026082{1,0902}_screen40_E1.yaml, clean 학습·QAF 없음·고해상 없음). 바꾸는 키 = MODEL.FUSION.TRUNK: mean · MODEL.P39.TRUNK_EXP: false (L2 와 같은 토글, 단 기준이 A2 가 아니라 E1). SAVE_DIR 만 행 이름 | - | backbone-control, ladder, paper |
| DRN-261005-01 | P15 | Q2noKD · Q2 원천 분리: 교사 증류 끔(KD_W 0), 열화 두 패스만 — 시드821 40ep 스크린(yeon GPU1) | Q2 시드821 config(jarvis-…screen40_Q2.yaml)에서 TRAIN.QAF.KD_W 0.5 → 0.0 하나만(그 외 SAVE_DIR·서버 경로). 09-26 yeon 3차 시도는 /SSDb 디스크 0바이트로 ep5 저장 중 크래시 — 현재 yeon /SSDb 72% 사용(975G 여유) | - | deliver, q2, ablation |

## queued
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-261017-02 | P1 | A2 이식 · MCubeS — E1Mc(4탭) + 고해상 경로, 시드 20260827·20260828 | MODEL.HIRES_ADAPTER on(그 외 E1Mc 동일) | - | mcubes, A2, transfer |
| DRN-261020-01 | P1 | A2 실패-키 분석(학습 0): 구조 제안의 선행 조건 — TrafficLight FP 장면 단위·세부-의미 분리 증거·열화+야간 혼동행렬 | 학습 없음. 기존 A2 2시드 ckpt(s821·s902) + Q2 2시드 ckpt 로 (1) TrafficLight FP 상위 20장면 시각화·FP 픽셀의 경로별 기여(dense vs query, hires on/off 추론 시 ablation) (2) TrafficLight 단독 두께 구간(thin_error_anatomy 클래스 단일 모드 추가) (3) MUSES 조건별 혼동행렬(night 3조건 분리) 및 DELIVER night/fog 조건별 혼동행렬 | - | A2, failure-keys, structural-novelty |
| DRN-261022-01 | P1 | 벤치 교차 확인(학습 0): 객체 단위 클래스 오라클 상한 + 객체 풀링 선형 프로브 — DELIVER(A2·Q2) 와 MUSES(E1M 1024/1536, 4모달 A2 완주 후 재측정) | 학습 없음. (a) 객체 클래스 오라클: 예측 맵의 연결 성분마다 GT 다수 클래스로 라벨 교체 후 mIoU 재계산(DELIVER test A2 s821·s902 / Q2 s821 덤프, MUSES val E1M 3시드 1024·1536 — 예측 덤프 없으면 공식 val 평가기로 1회 재추론해 pred 저장) + 클래스별 Δ 와 혼동쌍 기여 (b) 선형 프로브: A2 val 에서 GT 객체별로 fused 토큰 평균과 고해상 경로 s4 특징의 마스크 내 평균을 추출(모델 eval 스태시 훅, 학습 영향 0), 혼동쌍 2진 로지스틱 회귀 5-fold AUROC (c) MUSES 4모달 A2(DRN-261019-02) 완주 후 (a)(b) 재측정 | - | cross-bench, class-oracle, probe, structural-novelty |
| DRN-261027-02 | P2 | 논문 비용표(학습 0): A2·Q2·E1·DGFusion·CAFuser 의 params·FLOPs·추론 지연·GPU 메모리, DELIVER 1024 같은 조건 | 학습 없음. 같은 GPU(4090 1장)·같은 입력(1×4모달 1024²)·bf16, 100회 평균 지연·최대 메모리·fvcore FLOPs. 기준선은 재학습 킷 체크아웃 그대로 | - | paper, cost |
| DRN-261020-06 | P4 | S2 구현: 센서별 고해상 임베딩 F_s + 마스크 로짓 전용 결선(MODEL.HIRES_ADAPTER.USE fpn_sum|mask_only|mask_only_fused) + 스모크 | hires_adapter.py 센서별 출력 모드(s4 레벨만 반환) · model.py USE 분기(_apply_hires_adapter 우회, [F_s] 전달) · m2f_head.py sensor_s4 인자 + 센서별 W_s 선형 + masks += Σ_s einsum · tools/smoke_hires_adapter.py 확장(등가 assert 2종·grad 흐름·메모리 프로파일) · 기제 판독용 센서별 마스크 항 기여 덤프 옵션 | - | S2, build, structural-novelty |

## idea
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-261020-03 | P1 | S2-sum · DELIVER 40ep 스크린 3시드(s821·s902·s903) — 고해상 특징을 센서별 마스크 로짓 전용으로(합산 제거), A2 같은 시드 짝 비교 | MODEL.HIRES_ADAPTER.USE: fpn_sum → mask_only (그 외 A2 config 동일) | - | S2, screen40, structural-novelty |
| DRN-261029-02 | P1 | 구조 의존성 · DGFusion 에 두 패스 학습 전체 적용(clean 패스 + 열화 패스 + 동결 DGFusion clean 교사 증류) — clean 이득이 우리 구조(동결 DINOv3 + LoRA)에서만 생기는가 | DGFusion (b) 레시피(열화 커리큘럼, third_party/dgfusion_train_restore) → 두 패스(clean 패스 CE + 열화 패스 CE) + 동결 clean DGFusion 교사(DELIVER (a) 가중치, 우리 프로토콜 clean 55.68)의 로짓 KL 증류(α 0.5, T 2, A2 와 동일). 구조·스케줄·증강은 (b) 와 동일. 주의: (b) 대비 '두 패스 구조'와 '교사 증류' 두 요소가 함께 바뀐다(레시피 단위 전이 실험) | - | baseline, transfer, paper, contribution |
| DRN-261029-03 | P1 | 구조 의존성 · CAFuser 에 두 패스 학습 전체 적용(clean 패스 + 열화 패스 + 동결 CAFuser clean 교사 증류) — clean 이득이 우리 구조에서만 생기는가(DRN-261029-02 짝) | CAFuser (a) clean 학습 레시피 → 두 패스(clean 패스 CE + 열화 패스 CE, 열화 커리큘럼 = third_party/dgfusion_train_restore 의 (b) 와 동일) + 동결 CAFuser (a) clean 가중치(170k) 로짓 KL 증류(α 0.5, T 2, A2 와 동일). 구조·스케줄·증강은 (a)(b) 와 동일 | - | baseline, structure-dependence, paper, contribution |
| DRN-261020-04 | P2 | S2-fused 대조 · DELIVER 40ep 3시드 — 센서 합산 고해상 임베딩 하나로 마스크 전용(센서별 분리의 효과를 분리) | MODEL.HIRES_ADAPTER.USE: mask_only_fused (센서 합산 F 하나, W_s 없음) | - | S2, control, structural-novelty |
| DRN-261029-01 | P2 | Ablation L7 · A2 − 센서별 물리 열화(→ 결측 0채움만, 결측 노출량은 A2와 동일) — 두 패스 학습이 RobustSeg 'basic distillation'과 다른가 | TRAIN.QAF.DEGRADE_P 0.5→0.1 + MISSING_FRAC 0.2→1.0 (센서당 결측 확률 0.1 은 A2 와 동일하게 유지하고 물리 열화만 0으로; 교사 증류·clean 패스·고해상 경로 등 나머지 A2 config 동일) | - | ablation, leave-one-out, paper, contribution |
| DRN-261020-05 | P3 | S2-x 탐색 1시드 — F_s 생성 전 s4 창 단위 센서 간 교차주의 1층(게이트 없음, 잔차 합) | S2-sum + HIRES_ADAPTER.XSENSOR_ATTN: window 8, 1층, 표준 초기화 | - | S2, explore, structural-novelty |
| DRN-260926-05 | P5 | N-E18 · E18 = E1 레시피 + Lovász-Softmax 보조손실 40ep 스크린 시드821 | E1 레시피에 Lovász-Softmax 보조손실 추가(OHEM CE 유지, 픽셀 로짓, 배치 내 존재 클래스만, 가중 0.5) | - | deliver, screen40 |
| DRN-260926-06 | P6 | N-E19 · E19 = E1 레시피 + 가중치 EMA(0.9995) val 평가·저장 | E1 레시피에 가중치 EMA(0.9995)로 val 평가·저장 | - | deliver |
| DRN-260926-07 | P7 | N-D0 · D0 = LoRA 진단 2종(학습 0) | (i) 4모달 LoRA A 행렬 층별 코사인 (ii) 조건별 미니배치 그래디언트 코사인 | - | deliver, diagnostic |
| DRN-260926-08 | P8 | N-E23 · E23 = E1 레시피 + DGFusion식 depth 보조 감독 + 로컬 depth 토큰 | E1 레시피 + 학습 시 depth 헤드(보조 감독) + 로컬 depth 토큰(융합 cross-attn K/V) | - | deliver |
| DRN-260926-09 | P9 | N-USER · 사용자 결정 대기(잔여): 토큰 밀도 상향 · 마스크 분류 주 헤드 | 토큰 밀도 상향(실측 반대) · 마스크 분류 주 헤드(비용 4×) | - | deliver, user-decision |
| DRN-260926-13 | P13 | N3 · C3 진단-구동 검출기 (분석, 학습 0) | 기존 ckpt val confusion에서 class-transfer 붕괴 지표(비대각 집중도) 정량화 → C3 on/off 효과와 상관 검증 | - | deliver, muses, diagnostic |
| DRN-260926-19 | P19 | N9 · 정성 증명 패키지 | ①VICReg 전후 유효랭크 스펙트럼 ②C3 전후 confusion(RailTrack·rubber) ③per-class 예측맵 패널 | - | analysis, figure |
| DRN-260926-20 | P20 | P52-감사 · P52 기동 전 재설계(P52.1) | ① [C3-ADPT] 신호 즉검 ② 게이트 재등록(동등성 [−1.0,∞) 3페어 + G5 고정가중 스윕 −0.5 이내 + λ 궤적) ③ 병리 조작 1런 ④ MUSES 3모달 기준선 병기 | - | p52, user-decision |
| DRN-260926-21 | P21 | P50-재현 · P50 Phase1 init 유/무 매칭 페어 +2(총 3페어) | P50 Phase1 init 유/무 | - | deliver, p50 |
| DRN-260926-22 | P22 | E-BB A1 · 백본 적응 깊이 ablation 1순위: LoRA 전 선형층(QKVO+MLP) r32·α=64, 3페어 | LoRA를 전 선형층(QKVO+MLP) r32·α=64로 | - | deliver, backbone |
| DRN-260926-24 | P24 | DELIVER-perclass-3way · 클래스별 3자 비교표(학습 0) | 우리 P46 C3 5-seed / DGFusion 재현 / MM-SA 클래스별 비교 + 혼동형/공통붕괴형 재분류 + C3 on/off 클래스별 비용 | - | deliver, analysis |
| DRN-260930-01 | P25 | N-RGBX · 확정 구조의 RGB+X 2모달 ablation(MM SAM-adapter 2509.10408 과 같은 센서 조합으로 비교) | 학습 센서 집합만 축소(RGB+X 3조합), 구조·학습 레시피는 확정본 그대로. 교사 = 같은 2모달로 학습한 E1 교사(user 승인 2026-09-30, 선택지 a): 조합당 E1 2모달 40에폭 교사 1런을 먼저 돌린 뒤 Q2 형 학생을 학습 | - | ablation, rgbx, positioning |
| DRN-260926-27 | P27 | DAILY-CARDS · 일일 사이클 실험 카드 프로그램(40ep 스크린 → 200ep 3페어 확정) | 카드 순서 B0 → E0 → E9 → E7 → E1 → E2 → E3 → E4 → E8 → E5 → E6 → E10, EPOCHS 40 | - | deliver, program, status-unknown |
| DRN-260926-29 | P29 | S3-block-exchange · 매 블록 모달 간 교환 어댑터(백본 frozen) | 매 블록 모달 간 교환 어댑터(StitchFusion-MoA/CrossWeaver-MIB식) | - | deliver |
| DRN-260926-30 | P30 | P50-진단 · 학습0 진단 D1~D5(LogME·모달 Fisher·intruder-dim·CKA ID/OOD·WiSE-FT α=0.5) | 학습 0 진단 D1~D5로 Phase2 역전 원인 판정 → P50.1 처방 1개 등록 | - | p50, diagnostic |
| DRN-260926-31 | P31 | P52 · RxDINO 개정: 단일-config 자기-적응 처방(C3-adaptive + UniBal-adaptive + P50 init) | C3-adaptive(per-class λ) + UniBal-adaptive(per-modal λ_u) + P50 init, 벤치별 config 차이 0 | - | p52, deliver, muses, mcubes, status-unknown |
| DRN-260926-33 | P33 | N5 · TTA-on 실측 (참고용 ablation 행) | TTA on | - | deliver, ablation, filler |
| DRN-260926-34 | P34 | A100 ① · P47-2 UniBal (MUSES 4모달 역전 유일 레버) | P47-2 UniBal(보조 head) | - | muses, a100 |
| DRN-260926-35 | P35 | A100 ② · P51 후속 (F 추가 재프로브 or MUSES 비회귀) | P51 Δ 판정에 따른 후속 | - | a100 |
| DRN-260926-36 | P36 | E4b · 혼동쌍 명시(RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain) 40ep 스크린 — 재개 대기 | 혼동쌍 명시 margin | - | deliver, screen40, resume-wait |
| DRN-260926-37 | P37 | B0s2 · 기준선 시드2 40ep 스크린 — 재개 대기 | 기준선 시드 20260902 | - | deliver, screen40, resume-wait |
| DRN-260926-38 | P38 | E1s2 · 4탭 시드2 40ep 스크린 — 재개 대기 | E1 4탭 시드 20260902 | - | deliver, screen40, resume-wait |
| DRN-261001-04 | P45 | MUSES-Q2 · DELIVER 에서 채택한 Q2 레시피(열화 두 패스 + 동결 교사 증류)를 MUSES 3모달에 적용 — val→test 전이 손실(−3.52) 축소 카드 후보 | 학습 레시피만 Q2(두 패스 clean CE + 열화 CE, 교사 = 같은 시드 E1M PhysAug-off 완주본 동결). 구조·모달(RGB+Event+LiDAR)·에폭은 현행 헤드라인 레시피 그대로. 판정은 공식 test 제출이 있어야 성립(user 2026-09-29: 새 4모달 모델까지 제출 보류) — 제출 경로가 정해지기 전에는 queued 로 올리지 않음 | - | muses, q2, transfer |
| DRN-261002-01 | P60 | Q0-기준선-커리큘럼 · (옵션 B, 리뷰 대응용) 워커 step 공유를 파일 기반으로 고친 뒤 DGFusion 을 진짜 커리큘럼(0.3→0.6→1.0)으로 재학습 — 기준선에 커리큘럼을 주면 clean 손실이 사라지는가 | degradation.py 의 step 공유를 파일 기반으로 교체(토글, 기본 off) 후 DGFusion 재학습 200k, 그 외 (b) 와 동일 | - | deliver, baseline, robust, curriculum |
| DRN-260926-66 | P120 | E5 · 변형 가능 다중스케일 픽셀 디코더(MSDeformAttn 3층) | SimpleFPN → MSDeformAttn 픽셀 디코더(3층) | - | deliver, daily-cards |
| DRN-260926-67 | P121 | E6 · 매 블록 센서 간 교환 어댑터(S3) | 각 블록 attn·FFN 뒤 StitchFusion-MoA식 양방향 저랭크 MLP, 백본 frozen | - | deliver, daily-cards |
| DRN-260926-68 | P122 | E8 · 클래스 표적 copy-paste 증강 | RailTrack·Wall·Water·Bridge 인스턴스를 4센서 동시 copy-paste | - | deliver, daily-cards |
| DRN-260926-69 | P123 | E10 · 상위 절반 블록 부분 FT(S4) | 블록 13~24 unfreeze, LLRD 0.9, top lr 1e-5 | - | deliver, daily-cards |
| DRN-260926-70 | P124 | E17x · E17 통과 시 ViT-Adapter extractor 결합 | E17 합산 주입 → deformable cross-attn extractor ×4 | - | deliver, strategy-0917 |
| DRN-260926-71 | P125 | E20 · 공유 A + 모달 B + 공유 LoRA (D0-i ≥ 0.9일 때만) | ΔW_m = B_m A + B_s A, 라우터 없음 | - | deliver, strategy-0917 |
| DRN-260926-72 | P126 | E21 · 확률적 분할-B LoRA + 병합 | 모달 LoRA B를 2~3조각 분할, 배치마다 1조각, KL 일관성, 추론 시 병합 | - | deliver, strategy-0917 |
| DRN-260926-73 | P127 | E22 · 라벨 없는 조건 게이트 | 고정 tanh(γ_m) → RGB 최상위 특징 풀링 조건 토큰 소프트맥스 모달 가중치 | - | deliver, strategy-0917 |
| DRN-260926-74 | P128 | 마지막 2~4블록 MLP LoRA / 부분 해동 | 마지막 2~4블록 MLP LoRA 또는 부분 해동 | - | deliver, user-decision, strategy-0917 |
| DRN-260926-75 | P129 | Q3+R · QAF 위에 depth 경계 refinement + 연결 성분 손실 결합 카드 | Q3 + R(경계 prior refinement + 연결 성분 손실) | - | deliver, p54 |
| DRN-260926-76 | P130 | Q4 · depth-anchor 비대칭 D (Q3 통과 후) | QAF.DEPTH_ANCHOR | - | deliver, p54 |
| DRN-260926-77 | P131 | Q5 · 품질 조건부 잔차 LoRA M (E-LoRA 판정 + Q3 통과 후) | QAF.RESIDUAL_LORA | - | deliver, p54 |
| DRN-260926-287 | P140 | E2s3 · E2(전 선형층 LoRA r32) 시드3 40ep 스크린 | E2 시드 20260903 | - | deliver, screen40, daily-cards, status-unknown |
| DRN-260926-288 | P141 | E2b · E2 변형 카드 40ep 스크린 | E2 변형(config 참조) | - | deliver, screen40, daily-cards, status-unknown |
| DRN-260926-289 | P142 | E3 확정 200ep (센서별 prototype 단독) | E3 레시피 200ep 확정 런 | - | deliver, confirm200, daily-cards, status-unknown |
| DRN-260926-112 | P248 | [registry] jarvis_muses_rgbel_P44_bmr | P44 (BMR) | - | registry, muses, seg, status-unknown |
| DRN-260926-149 | P285 | [registry] hpca100_muses_rgbelr_P52_seed20260901 | P52 RxDINO 본런 seed1 | - | registry, muses, seg, hold |
| DRN-260926-153 | P289 | [registry] hpca100_muses_rgbelr_P52_seed20260902 | P52 RxDINO 본런 seed2 | - | registry, muses, seg, hold |
| DRN-260926-218 | P354 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_screen40_E3s2 | E3 시드2 페어(E3 test +1.40 최고의 시드 판별) | - | registry, deliver, seg, status-unknown |
| DRN-260926-219 | P355 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_E13 | E13 = E1(4탭)+E3(센서별 prototype) 결합 — test 1·2위 축 결합 | - | registry, deliver, seg, status-unknown |
| DRN-260926-224 | P360 | [registry] E2_legal_rescore_seed821 | E2 legal 재채점 ckpt epoch40_68.65_top1 | - | registry, deliver, 평가(학습 0), hold |
| DRN-260926-225 | P361 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260902_screen40_B0s2 / E1s2 | 시드 20260902 매칭 페어(B0·E1) | - | registry, deliver, seg, status-unknown |
| DRN-260926-234 | P370 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD / Q2noDeg | Q2 이득 원천 분리: noKD = KD_W 0(열화 두 패스만), noDeg = DEGRADE_P 0(증류만, clean CE 사실상 2회) | yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD_20261006_0227, yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD_20261016_0548 | registry, deliver, seg, status-unknown |
| DRN-260926-264 | P615 | [세대] P15: Calibrated Spatial Entropy Fusion (설계 단계) | P15: Calibrated Spatial Entropy Fusion (설계 단계) | - | generation, status-unknown |
| DRN-260926-272 | P623 | [세대] P23: MoE DeBA-BB (구현 완료, 학습 대기) (실험 M) | P23: MoE DeBA-BB (구현 완료, 학습 대기) (실험 M) | - | generation, status-unknown |
| DRN-260926-274 | P625 | [세대] P25: Unified Spatial Quality Fusion — Quality Map으로 UAMM + AMF + Memory 통합 (설계 중) | P25: Unified Spatial Quality Fusion — Quality Map으로 UAMM + AMF + Memory 통합 (설계 중) | - | generation, status-unknown |
| DRN-260926-275 | P626 | [세대] P26: Per-Modality SQG + Multi-Scale + Per-Modality Decoder + Modal-Cond MoE + UAMM Softmax (설계 v5, 2026-03-23) | P26: Per-Modality SQG + Multi-Scale + Per-Modality Decoder + Modal-Cond MoE + UAMM Softmax (설계 v5, 2026-03-23) | - | generation, status-unknown |
| DRN-260926-284 | P645 | [세대] P45: FogStyle (2026-07-25) — 미실행 | P45: FogStyle (2026-07-25) — 미실행 | - | generation, status-unknown |

## done
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-260926-01 | P1 | DELIVER 확정 3쌍 판정 · E1(중간층 4탭 카드)·E13(4탭+센서별 클래스 prototype) 확정 3쌍 | E1·E13 확정 3쌍(시드 821·902·903) vs 같은 코드 C3-only 기준선 | - | deliver, daily-cards, review:method |
| DRN-261012-01 | P1 | A1 · 고해상 학습 스케일(TRAIN_SCALE_RANGE [1,3], 평균 1536) — Q2 레시피 40ep 스크린 2시드 | DATASET.TRAIN_SCALE_RANGE [0.5,2.0]→[1.0,3.0](crop 768 유지). 부수: 학습 중 val-best 선택 해상도 768→1536 | - | deliver, thin-class, resolution, A1 |
| DRN-261016-01 | P1 | 기준선 평가 해상도 반응(원본 1042 GT 축) — DGFusion 공식·CAFuser(b) val@1024/1280/1536 + 선택 해상도 test |  | - | deliver, resolution, baseline, fairness |
| DRN-261018-01 | P1 | A2 3번째 시드 20260903 — Q2 + 고해상 경로 40ep 스크린(DRN-261012-02 inconclusive 후속) | TRAIN.SEED 20260903(구조·레시피는 A2 s821·s902 와 동일) | - | deliver, A2, seed3 |
| DRN-261019-03 | P1 | A2 강건성 — 2시드(s821·s902) DELIVER test EMM15·RMM·NM + val EMM15·RMM(공개 벤치 정의), Q2 와 같은 프로토콜 |  | - | deliver, A2, robustness |
| DRN-261028-01 | P1 | 고해상 특징이 실제 열화를 토큰 단위로 알아보는가(학습 0) — P56-C 기각의 원인('신호 없음')이 고해상 경로에서 해소되는지 재검 | 학습 없음. A2 s821 ckpt, DELIVER test 전수. 센서별 고해상 경로 출력(hires_adapter 의 센서별 스템 출력 또는 s4 레벨, 센서 분리 가능한 지점에서 추출)과 블록 6 토큰 특징을 토큰별 통계 3종으로 요약 → 파티션(clean vs 열화) 이진 AUROC(이미지 단위·토큰 단위 둘 다), 센서별. 기존 도구 tools/oracle_headroom_probe.py·Q1d 품질 헤드 프로브 코드를 확장(새로 짜지 않음) | - | probe, hires, routing-signal, P56C-recheck |
| DRN-260926-02 | P2 | MUSES 풀 런 4종 · E1M 3407 · E7 3407 · E13M 3407 · 시드 20260902 페어 | MUSES 200ep 풀 런 4종 공식 재채점 → test 제출 후보 확정 | - | muses, review:method |
| DRN-261012-02 | P2 | A2 · 고해상 경로(SPM conv 스템 + deformable extractor, injector 없음) — 1024 유지, Q2 레시피 40ep 스크린 2시드 | SimpleFPN+탭 합산 → SPM(센서별 경량 conv 스템, 합산) + extractor(conv 쿼리 1/8·1/16·1/32 ← 4탭 deformable attn) + c1=up(c2)+conv1/4. 백본·LoRA·헤드·레시피 불변, 입력 1024 | - | deliver, thin-class, adapter, A2 |
| DRN-261020-09 | P2 | Ablation L1 · A2 − 센서별 LoRA(→ 완전공유 LoRA 1개, r 동일) | A2 s821 config 대비 MODEL.LORA_MODE: shared · MODEL.LORA_R: 16 (E1shared 팔 B 와 동일 용량 맞춤, 09-21), 그 외 동일 | yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L1shared_20261021_1644 | ablation, leave-one-out, paper |
| DRN-261020-10 | P2 | Ablation L2 · A2 − cross-modal attention 융합(→ 센서 토큰 평균 MeanFusionTrunk) | A2 s821 config 대비 MODEL.FUSION.TRUNK: mean · MODEL.P39.TRUNK_EXP: false (코드 제약: mean 은 TRUNK_EXP false 필수, model.py:664-674; N2 평균융합 팔과 같은 결선), 그 외 동일 | yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L2meanfusion_20261021_2342 | ablation, leave-one-out, paper |
| DRN-261020-12 | P2 | Ablation L4 · A2 − 질의 헤드+arbiter(→ dense conv 헤드만) | A2 s821 config 대비 MODEL.M2F.ENABLE: false · MODEL.P39.ARBITER: false (dense conv 헤드만; PATH_DROPOUT_P 는 무효가 됨), 그 외 동일 | yeon_yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L4nom2f_20261022_1920 | ablation, leave-one-out, paper |
| DRN-260926-03 | P3 | E-LoRA · LoRA 구조 ablation 3갈래(A 센서별 r16 / B 완전공유 r16 / C 공유 r8+센서별 잔차 r8) | LoRA 공유 구조(A per-modal r16 vs B 완전공유 r16 vs C 공유 r8 + per-modal 잔차 r8), 시드 20260821 매칭 | - | deliver, e-lora, review:method |
| DRN-261020-08 | P3 | 백본 대조 0 · DINOv3-L + LoRA RGB 단독(1센서), clean 40ep 1시드(s821) | 기준 config = E1 스크린 s821. 바꾸는 키 = DATASET.MODALS: [img] (1센서; 참고 선례 configs/jarvis-muses_rgbel_P47_2_unibal_img.yaml). 융합·탭·헤드는 그대로(1모달에서 융합은 항등). SAVE_DIR 만 행 이름. 기동 검증: 센서별 LoRA 가 1세트만 생성돼 trainable 이 E1 보다 작아야 정상 | jarvis_jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_B2rgbonly_20261023_0220 | backbone-control, ladder, paper |
| DRN-260926-04 | P4 | N-P53 · E17 = E1 레시피 + 고해상도 세부 가지(DETAIL_BRANCH) 40ep 스크린 시드821 | E1 레시피 + DETAIL_BRANCH(입력 합성곱 스템 stride-4/8 특징을 FPN 에 게이트 잔차로 더함) | - | deliver, screen40, review:method |
| DRN-260926-10 | P10 | N-MC · 확정 레시피의 MCubeS 이식 3시드(200ep) | DELIVER 확정 레시피(E13/E1)를 MCubeS 통일 레시피 3시드(3407·20260827·20260828)에 매칭 | - | mcubes, review:method |
| DRN-260930-02 | P10 | Q0-기준선 · DGFusion (a)·(b) 강건 프로토콜 측정(EMM 14조합+clean · RMM 14조합×3비율 · NM S&P 3밀도+Gaussian, test 1897장) | 학습 변경 없음. 평가만: tools/baseline_failure/robust_bench_eval.py(한 순회에서 61케이스 forward, 케이스·시드는 tools/missing_modality_eval.py build_cases 를 그대로 import) | - | deliver, baseline, robust |
| DRN-260926-11 | P11 | N1 · MUSES 시드 분산 ×2 | MUSES 시드 2개 추가(s824·s825) | - | muses, seed, review:method |
| DRN-260926-12 | P12 | N2 · MLE-SAM 평균융합 baseline | 융합 믹서를 평균융합(MLE-SAM)으로 교체 | - | deliver, mixer, review:method |
| DRN-260926-14 | P14 | N4 · MCubeS 이식 파일럿 | MCubeS 첫 진입 3-seed | - | mcubes, review:method |
| DRN-260926-15 | P15 | N4b · MCubeS C3-on 페어 | MCubeS에서 C3 on | - | mcubes, c3, review:method |
| DRN-260926-16 | P16 | N6 · DELIVER legal-val ckpt 재선택 | ckpt 선택을 legal-val 기준으로 재선택(5시드) | - | deliver, protocol, review:method |
| DRN-260926-17 | P17 | N7 · VICReg-off 격리 토글 1런 (seed821 매칭) | MODEL.P39.VICREG.ENABLE off(그 외 동일) | - | deliver, ablation |
| DRN-260926-18 | P18 | N8 · P50 런 legal-val 재선택 | P50 런 6개 ckpt legal-val 스윕 | - | deliver, protocol, review:method |
| DRN-260929-01 | P20 | P56-A · 모달 충돌 학습(한 센서 영역에 틀린 내용을 넣고 나머지 센서와 정답을 따르게) | Q2 의 열화 패스에 모달 충돌 증강 추가: 한 센서의 임의 영역을 다른 장면 내용·포화 얼룩으로 바꾸고 정답은 그대로(나머지 센서가 본 장면) | - | P56, fusion, robust |
| DRN-260929-02 | P21 | P56-B · 거리 조건화 attention(depth·LiDAR 입력의 거리값을 융합 attention 의 위치 정보로) | CrossModalAttentionLayer 로짓에 −λ_h·|log r_i − log r_j|/σ_h 편향 추가(λ 초기 0). r = depth 토큰 평균 → 없으면 LiDAR 유효 화소 평균 → 둘 다 없으면 0 | - | P56, fusion, range |
| DRN-260929-03 | P22 | P56-C · 환경 조건 LoRA 전문가 혼합(공유 LoRA 와 센서별 LoRA 를 얕은 블록 통계로 토큰마다 섞음) | SharedLoRAQKV 확장: ΔW x = α(x)·shared_r8(x) + (1−α)·sensor_r8[m](x), α = sigmoid(MLP(LN(block6 특징 stop-grad) + 센서 임베딩)), 블록 7~24 에만 적용, 라우터 약 7만 파라미터 | - | P56, lora, moe |
| DRN-260926-23 | P23 | MUSES-PhysAug-off · MUSES 최선 레시피(P39.1-rank)에서 PhysAug만 off, 시드 매칭 3페어 | DATASET.PHYSAUG.ENABLE true→false (+ TRAIN.SEED · SAVE_DIR) | - | muses, fairness |
| DRN-260926-25 | P25 | S0-feature-probe · 사다리 출발점(학습 = 선형 헤드만), 원 특징 vs 어댑터 후 특징 프로브 | DINOv3 중간층 원 특징 vs 어댑터 후 특징에서 오라클 센서 선택·클래스 예측 정확도 비교 | - | deliver, diagnostic, review:method |
| DRN-261001-02 | P30 | N-RGBX-T · E1 레시피 2모달 교사 3런(RGB+Depth → RGB+LiDAR → RGB+Event, 시드821 40ep 스크린) — N-RGBX 학생의 교사이자 MM SAM-adapter 와 같은 센서 조합의 1차 비교선 | 센서 집합만 축소(DATASET.MODALS: [img,depth] / [img,lidar] / [img,event]). 그 외는 bengio E1 시드821 40ep 스크린 config(bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_E1.yaml) 그대로. QAF 없음(교사 자체가 clean 학습). 새 config 3개 = bengio-deliver_rgbd|rgbl|rgbe_P46_c3only_seed20260821_screen40_E1.yaml(감시 세션이 파생·스모크) | - | deliver, rgbx, teacher, screen |
| DRN-260926-32 | P32 | P50-EXT · P50 사전학습 확장(코퍼스/에폭 스케일업) | P50 사전학습 코퍼스 500k·등-스텝 확장, EXT-init 파인튠 | - | p50, deliver, review:method |
| DRN-261001-03 | P32 | Q0-기준선-공식 · DGFusion 공식 DELIVER 가중치(게시 val 66.51/test 56.71)로 clean 재측정 + 같은 강건 프로토콜 61케이스 — (a) 재현본(test 55.68) 대신 발표 가중치를 기준선으로 | 학습 없음. dgfusion 세션이 보존한 공식 가중치(NAS ckpts/dgfusion_official_20261001/ DELIVER clde)로 ①README 명령 그대로 clean val·test 재측정(약 15분) ②tools/baseline_failure/robust_bench_eval.py 로 61케이스 측정(약 7 GPU-시간). 등가검증은 DRN-260930-02 와 같은 세 케이스 | - | deliver, baseline, robust |
| DRN-261001-01 | P35 | MUSES-기준선 · DGFusion 공식 MUSES 가중치(4모달, radar 포함)로 val 250장 재현 → 우리 E1M(P39.1-rank PhysAug-off) val 덤프와 클래스·조건·거리 비교 | 학습 없음. ①bengio 에 MUSES val 250장(RGB·event·lidar·radar 투영본) 스테이징 ②DGFusion 저장소 README 의 공식 MUSES 가중치로 val 추론 → 클래스별 IoU·조건별·거리 구간별 덤프 ③우리 E1M val 덤프(같은 250장)와 표로 대조. 공식 가중치 재현이 실패(README 수치 대비 −1.0 초과)할 때만 DGFusion MUSES 재학습(180k iter, 4090×4 약 28시간) | - | muses, baseline, analysis |
| DRN-260926-58 | P40 | CAFuser (b) · 열화 커리큘럼 재학습 재개(iter 90k→267k, detectron2) + 같은 강건 프로토콜 61케이스 측정 — 두 번째 융합 기준선 | 학습: CAFuser 발표 레시피(bs8, MAX_ITER 200,000) + DGFusion (b) 와 같은 열화 재학습 조건 — **정정(ISSUE-041, 2026-10-02): 커리큘럼이 아니라 학습 내내 최대 강도 열화(PER_MODAL_PROB 0.5, severity 상한 1.0)이며 재개 config 에 CURRICULUM [1.0,1.0,1.0] 을 명시**. iter 90k 체크포인트(bengio /SSDb/jemo_maeng/dgfusion_train/output/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade/model_0089999.pth, md5 b3d1e2ed…)에서 재개, 잔여 110k. 평가: tools/baseline_failure/robust_bench_eval.py 로 test 1897장 61케이스, 시드0 | - | deliver, baseline, resume-wait |
| DRN-260926-45 | P45 | A/B 격리 (Arm A/B) | A/B 격리 (Arm A/B) | - | review:method |
| DRN-260926-46 | P46 | TTA 판정 | TTA 판정 | - | review:method |
| DRN-260926-47 | P47 | 투영 정합 | 투영 정합 | - | review:method |
| DRN-260926-48 | P48 | module ablation | module ablation | - | review:method |
| DRN-260926-49 | P49 | det 붕괴 진단 | det 붕괴 진단 | - | det, review:method |
| DRN-260926-50 | P50 | seg-P37a/b (bengio분) | seg-P37a/b (bengio분) | - | review:method |
| DRN-260926-51 | P51 | MUSES 시드 20260825 test 제출본 | MUSES 시드 20260825 test 제출본 | - | review:method |
| DRN-260926-52 | P52 | P43-PanopticDual (MUSES, hpca100) | P43 panoptic dual head | - | muses, review:method |
| DRN-260926-53 | P53 | jarvis_muses_rgbl_P39_1_rank_2modal · MUSES RGB-L 2모달 (2026-08-06 완주) | MUSES 2모달(RGB-LiDAR) | - | muses, review:method |
| DRN-260926-57 | P104 | E1-shared · E1 레시피 + 전 센서 공유 LoRA r16 40ep 스크린 3시드 | 센서별 LoRA → 전 센서 공유 LoRA r16(E1 4탭 위) | - | deliver, e-lora, screen40, p54 |
| DRN-260926-59 | P110 | D4/D6 · 기준선 실패 분석(DGFusion·CAFuser·우리 모델 모달 zero-out·조건별 Δ·깊이 구간·셀 지도·원거리 원인) | 학습 0 분석 | - | deliver, analysis, baseline, review:method |
| DRN-260926-60 | P111 | Q1 · 품질 헤드 실현성 프로브(Q1 5ep·Q1b-1 15ep·Q1b-2 연산자별 분해) | 동결 E1 특징 위 품질 헤드만 학습 | - | deliver, p54, probe, review:method |
| DRN-260926-61 | P112 | R1 · depth 경계 prior refinement 40ep 스크린 3시드 | depth 불연속(HHA 에지) 경계 prior stride-4 refinement | - | deliver, screen40, p54, review:method |
| DRN-260926-62 | P113 | R2 · 연결 성분 soft-IoU 손실 40ep 스크린 | 연결 성분 단위 soft-IoU 손실(원거리 가중) | - | deliver, screen40, p54, review:method |
| DRN-260926-63 | P114 | E1 40ep 스크린 3시드 짝 기준선 | E1 40ep 스크린 시드 821·902·903 | - | deliver, screen40, review:method |
| DRN-260926-64 | P115 | E3b · E3 + 센서 간 prototype 일치 항(AGREE_LAMBDA 0.1) | AGREE_LAMBDA 0.1 | - | deliver, screen40, daily-cards, review:method |
| DRN-260926-65 | P116 | E4c · 혼동 쌍 margin(RailTrack 3쌍·MARGIN 0.25) | 혼동 쌍을 RailTrack 3쌍으로 좁히고 MARGIN 0.25 | - | deliver, screen40, daily-cards, review:method |
| DRN-260926-290 | P150 | D1 · det 인증 배포 계열(ViT-S/S+/B 백본, poongsan) | D1 검출 백본(ViT-S+ 등) | - | det, poongsan, review:method, status-unknown |
| DRN-260926-292 | P151 | det 백본 이식 계열 P34~P39-Det·RF-DETR(poongsan) | seg 세대 백본의 검출 이식 | - | det, review:method, status-unknown |
| DRN-260926-304 | P160 | [누락] P39.1 MUSES 3모달 시드4 (yeon) | P39.1-rank 시드4 | - | muses, review:method, status-unknown |
| DRN-260926-305 | P161 | [누락] P44-BMR DELIVER 시드2 (yeon) | P44-BMR 시드2 | - | deliver, review:method, status-unknown |
| DRN-260926-306 | P162 | [누락] P34 MUSES 4모달 DGF 투영 변형 | P34 + DGFusion 투영(4모달) | - | muses, review:method, status-unknown |
| DRN-260926-307 | P163 | [누락] MUSES Codabench 공식 test 제출 이력(11건) | 제출본 zip | - | muses, submission, review:method |
| DRN-260926-308 | P164 | [누락] DGFusion 기준선 재현 평가(DELIVER test·MUSES val·P39.1 대비 비교) | 공개 DGFusion 가중치/재학습본 평가 | - | deliver, muses, baseline, review:method |
| DRN-260926-87 | P218 | [registry] levine_multiaqua_rgbtl_P9_hardaug8_physaug (ep131) | P9 | - | registry, multiaqua, seg, review:method |
| DRN-260926-88 | P219 | [registry] levine_multiaqua_rgbtl_P22_hardaug8_physaug (ep120) | P22 | - | registry, multiaqua, seg, review:method |
| DRN-260926-89 | P220 | [registry] levine_multiaqua_rgbtl_P21_hardaug8_physaug (ep94) | P21 | - | registry, multiaqua, seg, review:method |
| DRN-260926-90 | P226 | [registry] dgfusion_swin_tiny_bs8_200k_deliver_clde | DGFusion Swin-T (공개 SOTA 계열 베이스라인 직접 학습, 학습코드 복원본) | - | registry, deliver, seg-baseline, review:method |
| DRN-260926-91 | P227 | [registry] dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade | DGFusion Swin-T 재학습 (b) — 모달 드롭 p=0.2 + 열화 커리큘럼 | - | registry, deliver, seg-baseline, review:method |
| DRN-260926-92 | P228 | [registry] cafuser_swin_tiny_bs6_267k_deliver_clde_lecun | CAFuser Swin-T (공식 학습코드, DGFusion 대조군) | - | registry, deliver, seg-baseline, review:method |
| DRN-260926-93 | P229 | [registry] b200_deliver_rgbdel_P31_physaug | P31 (P31.1) | - | registry, deliver, seg, status-unknown |
| DRN-260926-94 | P230 | [registry] ANALYSIS: P29·P31·P32·P34 표준분석 (lecun) | P29 ep146 + P31 ep182 + P32 ep158 + P34 ep40(스냅샷) | - | registry, deliver, seg-analysis, review:method |
| DRN-260926-95 | P231 | [registry] b200_deliver_rgbdel_P29_physaug | P29 (SDC) | - | registry, deliver, seg, review:method |
| DRN-260926-96 | P232 | [registry] b200_deliver_rgbdel_P30_physaug | P30 | - | registry, deliver, seg, review:method |
| DRN-260926-98 | P234 | [registry] ANALYSIS: P37a-CEFR MUSES 출력분석 (yeon) | P37a ep110 (val-best 81.16) | - | registry, muses, seg-analysis, review:method |
| DRN-260926-99 | P235 | [registry] ANALYSIS: P37a MUSES 표준분석 + 종합 실패-키 (yeon) | P37a ep110 (val-best 81.16) | - | registry, muses, seg-analysis, review:method |
| DRN-260926-100 | P236 | [registry] hpca100_deliver_rgbdel_P39_dpc | P39-DPC (V1~V5) | - | registry, deliver, seg, status-unknown |
| DRN-260926-101 | P237 | [registry] jarvis_muses_rgbel_P39_dpc | P39-DPC (V1~V5) | - | registry, muses, seg, status-unknown |
| DRN-260926-102 | P238 | [registry] ANALYSIS: P39 조기 즉검 + DELIVER 3시점 + 모듈 시각리포트 (yeon) | P39 ep38/ep60/ep64 · ep66(MUSES) | - | registry, deliver, muses, seg-analysis, review:method |
| DRN-260926-103 | P239 | [registry] ANALYSIS: P39-MUSES 3모달 표준분석 (yeon) | P39-DPC ep146 (val 81.52 / 공식 test 78.881) | - | registry, muses, seg-analysis, review:method |
| DRN-260926-104 | P240 | [registry] SAM3-RBMA (DELIVER 25cls) | SAM3-RBMA | - | registry, deliver, seg, status-unknown |
| DRN-260926-105 | P241 | [registry] hpca100_deliver_rgbdel_P38_m2f | P38 (MaskQueryLite) | - | registry, deliver, seg, status-unknown |
| DRN-260926-106 | P242 | [registry] ANALYSIS: P38-m2f 표준분석 (yeon) | P38 ep62 (test-best 55.05) | - | registry, deliver, seg-analysis, review:method |
| DRN-260926-107 | P243 | [registry] jarvis_muses_rgbel_P39_1_rank_seed2 | P39.1-rank (seed2) | - | registry, muses, seg, review:method |
| DRN-260926-108 | P244 | [registry] hpca100_muses_rgbelr_P39_1_rank_4modal_seed2 | P39.1-rank (4-modal, seed2 — radar 추가 3seed race) | - | registry, muses, seg, review:method |
| DRN-260926-109 | P245 | [registry] hpca100_muses_P47_d1_dgfproj_4modal | P47-MUB D-1(LiDAR 투영 밀도화, projected_to_rgb_dgf, (7,7)+motion-comp 32.6%=4.99×) — base=4모달 seed2(val 82.35) 그대로, PROJ_DIR 1건만 변경 | - | registry, muses, seg, review:method |
| DRN-260926-110 | P246 | [registry] yeon_muses_rgbel_P39_1_seed3 | P39.1-rank (seed3) | - | registry, muses, seg, review:method |
| DRN-260926-111 | P247 | [registry] hpca100_muses_rgbel_P43_pdual | P43 (PanopticDual) | - | registry, muses, seg, review:method |
| DRN-260926-113 | P249 | [registry] hpca100_muses_rgbel_P44_bmr | P44 (BMR) | - | registry, muses, seg, review:method |
| DRN-260926-115 | P251 | [registry] jarvis_deliver_rgbdel_P46_ctr_c1c3 | P46-CTR C1+C3 변형 (C2_MCC off, C3.CROSS_VIEW off — all-on OOM 회피, P39.1 base) | - | registry, deliver, seg, review:method |
| DRN-260926-116 | P252 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only 🔴 헤드라인 ckpt 정본 = NAS ckpts/P46_c3only_base_ep70_test5699_20260730/epoch70_67.79_top1_checkpoint.pth(md5 d340e3fe…, 2026-09-17 이관 — ISSUE-035) | P46-CTR C3-only ablation (C1_RCS off, C2_MCC off, C3_PROTO만 유지·cross_view off — c1c3 대비 C3 단독기여 격리, P39.1 base) | - | registry, deliver, seg, review:method |
| DRN-260926-118 | P254 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_seed2 | P46-CTR C3-only 재현성 검증(seed2, C1 off/C2 off/C3 on·cross_view off, c3only와 동일 구성) | - | registry, deliver, seg, review:method |
| DRN-260926-119 | P255 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_res1024_seedB | P46 C3 @1024 학습 진짜시드 B(TRAIN.SEED) | - | registry, deliver, seg, review:method |
| DRN-260926-120 | P256 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_res1024_seedC | P46 C3 @1024 학습 진짜시드 C | - | registry, deliver, seg, review:method |
| DRN-260926-121 | P257 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_768_seed15/16 | @768 대표구성(λ0.1) 진짜시드 2런 — 논문 mean±std. ⚠️ 1차 기동은 discussion 세션의 λ 지정 착오(λ0.2)로 ep120대에서 kill→λ0.1 재기동(2026-08-16) — 대표 구성 = 접미사 없는 c3only config(λ0.1, 본run 67.79@70의 레시피)로 확정 | - | registry, deliver, seg, status-unknown |
| DRN-260926-123 | P259 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam02 | P46-CTR C3-only λ=0.2(C1 off/C2 off/C3 on·cross_view off, λ 스윕 상단) | - | registry, deliver, seg, review:method |
| DRN-260926-124 | P260 | [registry] hpca100_muses_rgbel_P46_c3only_lam02 | P46-CTR C3-only λ=0.2 (C1_RCS off, C2_MCC off, C3_PROTO on·cross_view off — DELIVER SOTA-breaking 기제를 MUSES로 이식, P39.1-rank seed2 base 동결) | - | registry, muses, seg, review:method |
| DRN-260926-125 | P261 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam03 | P46-CTR C3-only λ=0.3(C1 off/C2 off/C3 on·cross_view off, λ 스윕 상단탐색) | - | registry, deliver, seg, review:method |
| DRN-260926-126 | P262 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam02_seed2 | P46-CTR C3-only λ=0.2 seed2(SOTA 재현성 검증, C1 off/C2 off/C3 on·cross_view off, lam02와 동일 구성) | - | registry, deliver, seg, review:method |
| DRN-260926-127 | P263 | [registry] hpca100_muses_rgbelr_P47_2_unibal_4modal | P47-2 UniBal (Uni-modal Balance, 구 D-2 — 모달별 독립 aux head + uni-modal CE, λ_u 0.4/MODALS all/HEAD linear/WARMUP 0, OGM_GE off. P39.1-rank 4모달 seed2 base 동결) | - | registry, muses, seg, review:method |
| DRN-260926-128 | P264 | [registry] eliceb200_deliver_rgbdel_P46_c3only_1024 | P46-CTR C3-only @1024²(본run, BS8·eff-batch16) | - | registry, deliver, seg, status-unknown |
| DRN-260926-129 | P265 | [registry] jarvis_deliver_rgbdel_P46_c3only_1024_gpu67 | P46-CTR C3-only @1024² | - | registry, deliver, seg, status-unknown |
| DRN-260926-130 | P266 | [registry] jarvis_deliver_rgbdel_P39_1_rank_1024_ctrl | P39.1-rank @1024²(해상도 교란 대조군, C3 미적용) | - | registry, deliver, seg, review:method |
| DRN-260926-131 | P267 | [registry] jarvis_muses_rgbl_P39_1_rank_2modal | P39.1-rank 레시피 − event (RGB-L) | - | registry, muses, seg, review:method |
| DRN-260926-132 | P268 | [registry] jarvis_deliver_P46_c3only_xattn_trunk | cross-attn 트렁크 A/B(#12) — XAttnTrunk(대칭 모달간 attn, LayerScale 1.0) vs gated-MLP, 나머지 byte-동일 | - | registry, deliver, seg, review:method |
| DRN-260926-133 | P269 | [registry] yeon_mcubes_rgbadn_P39_1_rank | N4 통일 레시피(P39.1-rank, C3 off) 이식 파일럿 | - | registry, mcubes, seg, review:method |
| DRN-260926-134 | P270 | [registry] jarvis_muses_P39_1_rank_seed20260824 | N1 진짜-시드 분산 측정(TRAIN.SEED 20260824) | - | registry, muses, seg, review:method |
| DRN-260926-135 | P271 | [registry] yeon_muses_P39_1_rank_seed20260825_testsubmit | N1 시드 20260825 의 Codabench test 제출본 생성(2026-09-17) — MUSES 1위 수치가 시드2 단일 제출이라 시드 평균을 만들기 위한 둘째 제출 후보 | - | registry, muses, seg, review:method |
| DRN-260926-136 | P272 | [registry] yeon_p50_align_pretrain | P50-MAP 정렬 사전학습(#11) — frozen DINOv3-L + LoRA+트렁크 cross-modal masked recon(MultiMAE식), trainable 50.4M/frozen 309.6M | - | registry, seg-pretrain, status-unknown |
| DRN-260926-137 | P273 | [registry] yeon_p50_pseudomodal_gen | P50 pseudo-모달 생성(depth/lidar/event) — depth 백엔드 = Depth-Anything-V2-Small(Omnidata ckpt 미공개로 폴백, meta.json 기록) | - | registry, infra, status-unknown |
| DRN-260926-138 | P274 | [registry] yeon_muses_rgbelr_P49_1_air_4modal_g01 | P49.1-AIR 4모달 — G-4M 쌍의 두 번째 팔 | - | registry, muses, seg, status-unknown |
| DRN-260926-139 | P275 | [registry] yeon_muses_rgbel_P49_1_air_g01 | P49.1-AIR MUSES 본런 — C3 off(H8)·EPOCHS 300·γ0.1 | - | registry, muses, seg, status-unknown |
| DRN-260926-140 | P276 | [registry] hpca100_deliver_rgbdel_P46_ctr_c2c3 (재기동) | C2-MCC 순기여 측정 — C2(masked consistency)+C3(λ0.1) vs C3-only(λ0.1) A/B | - | registry, deliver, seg, review:method |
| DRN-260926-141 | P277 | [registry] yeon_deliver_rgbdel_P49_1_air_768_g01 | P49.1 = P49-AIR + γ init 0.1(키1 처방, H13 반증 반영) | - | registry, deliver, seg, review:method |
| DRN-260926-142 | P278 | [registry] yeon_deliver_rgbdel_P49_air_768 | P49-AIR 본런(비대칭 주입, @768 학습, 150ep) | - | registry, deliver, seg, review:method |
| DRN-260926-143 | P279 | [registry] yeon_deliver_rgbd_P46_c3only_lam005_2modal_eval1024 | RGB-D 완주본(ep66 val-best) @1024 fair-eval | - | registry, deliver, seg, review:method |
| DRN-260926-144 | P280 | [registry] P48 인스턴스 감독 제안 | P48 (쿼리 경로 instance supervision — 클래스단위 타깃→connected-component 인스턴스단위 전환 제안) | - | registry, muses, seg, review:method |
| DRN-260926-145 | P281 | [registry] jarvis_muses_rgbl_P39_1_rank_2modal | P39.1-rank(gated_mlp trunk+VICReg+P36 router+M2F, RGB-L 직접비교용) | - | registry, muses, seg, review:method |
| DRN-260926-146 | P282 | [registry] yeon_deliver_rgbd_P46_c3only_lam005_2modal_eval1024 | P46-CTR C3-only(λ0.05) @1024 재평가(학습 0, 완주 ckpt 재사용) | - | registry, deliver, seg-eval, status-unknown |
| DRN-260926-147 | P283 | [registry] hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 | P46-C3only + P50-EXT Phase2 init 채택 게이트 파인튠(seed20260821 매칭) | - | registry, deliver, seg, review:method |
| DRN-260926-148 | P284 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_permodal_r16 | E-LoRA arm A(per-modal) r16(seed20260821 매칭) | - | registry, deliver, seg, status-unknown |
| DRN-260926-150 | P286 | [registry] yeon_deliver_rgbdel_P52_seed20260901 | P52 RxDINO 본런 seed1 | - | registry, deliver, seg, status-unknown |
| DRN-260926-151 | P287 | [registry] yeon_deliver_rgbdel_P52_seed20260902 | P52 RxDINO 본런 seed2 | - | registry, deliver, seg, status-unknown |
| DRN-260926-152 | P288 | [registry] yeon_mcubes_rgbadn_P52_seed20260901 | P52 RxDINO 본런 seed1 | - | registry, mcubes, seg, review:method |
| DRN-260926-154 | P290 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_vicregoff | P46-CTR C3-only(λ0.1) N7 VICReg-off 격리(seed20260821 매칭) | - | registry, deliver, seg, review:method |
| DRN-260926-155 | P291 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_B0 | 일일 카드 B0 기준선 스크린(현행 P46 C3-only λ0.1 레시피, 변경 0) | - | registry, deliver, seg, review:method |
| DRN-260926-156 | P292 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1 | E1 4탭 읽기(MODEL.TAPS per_modal [6,12,18,24], B0 대비 변경 1) | - | registry, deliver, seg, review:method |
| DRN-260926-157 | P293 | [registry] E0_probe_feature_info_deliver_seed821 | E0 특징 정보 프로브(raw 4탭 vs adapted vs fused, 클래스 선형 프로브) — ckpt jarvis seed821 val-best epoch90_67.3_top1 | - | registry, deliver, 분석(학습 0), review:method |
| DRN-260926-158 | P294 | [registry] E9_logit_adjust_deliver_seed821 | E9 검증셋 사전 로짓 보정 τ∈{0,0.5,1.0} test — τ=0 = 하네스 재동결용 대표 재채점(정본 기준 54.21) | - | registry, deliver, 평가(학습 0), review:method |
| DRN-260926-159 | P295 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E2 | 일일 카드 E2 전 선형층 LoRA(S2, LORA_TARGETS qkv_full 등으로 확장, B0 대비 변경) | - | registry, deliver, seg, status-unknown |
| DRN-260926-160 | P296 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_screen40_E7 | 일일 카드 E7 MUSES PhysAug-off 기준선(PHYSAUG.ENABLE False, B0 대비 변경) | - | registry, muses, seg, review:method |
| DRN-260926-161 | P297 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugon_screen40_E7c | 일일 카드 E7c PhysAug-on 대조군(E7과 PHYSAUG만 상이, B0 대비 변경 없음 = 정본 레시피) | - | registry, muses, seg, review:method |
| DRN-260926-162 | P298 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E3 | E3 센서별 prototype 손실(P46.C3_PROTO.SRC permodal, AGREE 0) | - | registry, deliver, seg, review:method |
| DRN-260926-163 | P299 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E4 | E4 혼동 쌍 margin(auto_val_k5, m 0.5, λ 0.05, proto) | - | registry, deliver, seg, status-unknown |
| DRN-260926-164 | P300 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E4b | E4b 혼동 쌍 명시(RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain) | - | registry, deliver, seg, status-unknown |
| DRN-260926-165 | P301 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E12 | E12 = E1(4탭)+E2(전 선형층 LoRA r32) 결합 | - | registry, deliver, seg, status-unknown |
| DRN-260926-166 | P302 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_E13_confirm200 | E13 확정 런 시드1(200ep) = E1(4탭) + E3(센서별 prototype) | - | registry, deliver, seg, review:method |
| DRN-260926-167 | P303 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E13_confirm200_s2 | E13 확정 런 시드2(200ep), SEED 20260902 | - | registry, deliver, seg, review:method |
| DRN-260926-168 | P304 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260903_E13_confirm200_s3 | E13 확정 런 시드3(200ep), SEED 20260903 — 3페어 게이트의 세 번째 | - | registry, deliver, seg, review:method |
| DRN-260926-169 | P305 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200 | E1 확정 런 시드1(200ep) = 중간층 4탭 읽기 단독 | - | registry, deliver, seg, status-unknown |
| DRN-260926-170 | P306 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E1_confirm200_s2 | E1 확정 런 시드2(200ep), SEED 20260902 | - | registry, deliver, seg, review:method |
| DRN-260926-171 | P307 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260903_E1_confirm200_s3 | E1 확정 런 시드3(200ep), SEED 20260903 | - | registry, deliver, seg, review:method |
| DRN-260926-172 | P308 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E13s2 | E13 시드2 판별 = E1(4탭)+E3(센서별 prototype), SEED 20260902 | - | registry, deliver, seg, review:method |
| DRN-260926-173 | P309 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260903_screen40_E13s3 | E13 시드3 판별, SEED 20260903 | - | registry, deliver, seg, status-unknown |
| DRN-260926-174 | P310 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E14 | 탭 깊이 어블레이션 = E13 레시피 + TAPS.LAYERS 만 [6,12,18,24]→[4,8,12,24] | - | registry, deliver, seg, status-unknown |
| DRN-260926-175 | P311 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E15 | 탭 투영 어블레이션 = E13 레시피 + TAPS.MODE 만 per_modal→mean(투영 16개→4개, 약 4.3M→1.1M) | - | registry, deliver, seg, status-unknown |
| DRN-260926-176 | P312 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_screen40_E1M | E1(4탭)의 MUSES 이식 — E7(PhysAug-off 기준선) 대비 유일 변수 MODEL.TAPS on | - | registry, muses, seg, review:method |
| DRN-260926-177 | P313 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_c3permodal_screen40_E13M | E13 의 MUSES 이식 = E1M + P46.C3_PROTO on(SRC permodal, DELIVER E3/E13 과 동일 값) | - | registry, muses, seg, review:method |
| DRN-260926-178 | P314 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_screen40_E13M_s2 | E13M 시드2 — 시드1 의 야간 악조건 손실이 실재인지 시드 노이즈인지 가린다 | - | registry, muses, seg, review:method |
| DRN-260926-179 | P315 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_screen40_E1M_s2 | E1M 시드2 — E13M 의 야간 손실이 prototype 탓인지 시드 노이즈인지 가를 대조군 | - | registry, muses, seg, status-unknown |
| DRN-260926-180 | P316 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed3407 | E13Mc 시드 3407 — E13(4탭 + 센서별 prototype)의 MCubeS 이식 | - | registry, mcubes, seg, review:method |
| DRN-260926-181 | P317 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed20260827 | E13Mc 시드 20260827 — E13(4탭 + 센서별 prototype)의 MCubeS 이식 | - | registry, mcubes, seg, review:method |
| DRN-260926-182 | P318 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed20260828 | E13Mc 시드 20260828 — E13(4탭 + 센서별 prototype)의 MCubeS 이식 | - | registry, mcubes, seg, review:method |
| DRN-260926-183 | P319 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed3407 | E1Mc 시드 3407 — E1(4탭)의 MCubeS 이식 | - | registry, mcubes, seg, review:method |
| DRN-260926-186 | P322 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_shared | E-LoRA arm B = 완전공유 LoRA r16(1.57M). arm A(센서별 r16, 6.29M)·arm C(공유8+잔차8, 3.93M)와 3자 비교 | - | registry, deliver, seg, status-unknown |
| DRN-260926-187 | P323 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_sharedresidual | E-LoRA arm C = 공유 r8 + 센서별 잔차 r8(3.93M) | - | registry, deliver, seg, review:method |
| DRN-260926-188 | P324 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260902 | 확정 런 매칭 분모 P46 C3-only 200ep 시드 20260902(프로토타입 손실만 쓴 기준선) (E1·E13·E-LoRA 시드2 짝) | - | registry, deliver, seg, review:method |
| DRN-260926-189 | P325 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260903 | 확정 런 매칭 분모 P46 C3-only 200ep 시드 20260903(프로토타입 손실만 쓴 확정 매칭 분모) | - | registry, deliver, seg, review:method |
| DRN-260926-190 | P326 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_screen40_E7_s2 | E7 시드 20260902 — PhysAug-off 기준선의 시드 페어 | - | registry, muses, seg, review:method |
| DRN-260926-191 | P327 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_screen40_E7_s3 | E7 시드 20260903 | - | registry, muses, seg, review:method |
| DRN-260926-192 | P328 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_screen40_E1M_s3 | E1M 시드 20260903 — 중간층 4탭 읽기의 MUSES 이식, 셋째 시드 | - | registry, muses, seg, review:method |
| DRN-260926-193 | P329 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_full200_E1M | E1M 200ep 풀 런(시드 3407) — MUSES test 서버 제출 후보 | - | registry, muses, seg, review:method |
| DRN-260926-194 | P330 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_full200_E13M | E13M 200ep 풀 런(시드 3407) — MUSES test 제출 후보(E13 쪽) | - | registry, muses, seg, status-unknown |
| DRN-260926-195 | P331 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_full200_E13M_s2 | E13M 200ep 풀 런 시드2(20260902) — 풀 런 시드 페어 | - | registry, muses, seg, status-unknown |
| DRN-260926-196 | P332 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_full200_E1M_s2 | E1M 200ep 풀 런 시드2(20260902) — 풀 런 시드 페어 | - | registry, muses, seg, status-unknown |
| DRN-260926-197 | P333 | [registry] jarvis_muses_rgbel_P39_1_physaugoff_full200_E7 | E7 기준선 200ep 풀 런(시드 3407, MUSES 비교 기준 카드) — E1M 풀 런의 같은 길이 짝(4탭만 뺀 구성) | - | registry, muses, seg, review:method |
| DRN-260926-198 | P334 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_screen40_E13M_s3 | E13M 시드 20260903 — E13(4탭 + 센서별 prototype)의 MUSES 셋째 페어 | - | registry, muses, seg, review:method |
| DRN-260926-199 | P335 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed3407 | B0Mc 시드 3407 — MCubeS 같은 코드 기준선(E1Mc·E13Mc 재판정 분모) | - | registry, mcubes, seg, review:method |
| DRN-260926-200 | P336 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260827 | B0Mc 시드 20260827 | - | registry, mcubes, seg, review:method |
| DRN-260926-201 | P337 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260828 | B0Mc 시드 20260828 | - | registry, mcubes, seg, review:method |
| DRN-260926-202 | P338 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | E1Mc 시드 20260827 (hpca100, TAPS 적용) — yeon 무효 시드의 재실행 | - | registry, mcubes, seg, review:method |
| DRN-260926-203 | P339 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828 | E1Mc 시드 20260828 (hpca100, TAPS 적용) | - | registry, mcubes, seg, review:method |
| DRN-260926-206 | P342 | [registry] bengio_deliver_…_seed20260902_elora_permodal_r16 | E-LoRA arm A 시드 902(센서별 r16, 6.29M) — 생각정리 세션 기동 | - | registry, deliver, seg, status-unknown |
| DRN-260926-207 | P343 | [registry] bengio_deliver_…_seed20260902_elora_shared | E-LoRA arm B 시드 902(완전공유 r16, 1.57M) | - | registry, deliver, seg, status-unknown |
| DRN-260926-208 | P344 | [registry] bengio_deliver_…_seed20260903_elora_permodal_r16 | E-LoRA arm A 시드 903 | - | registry, deliver, seg, status-unknown |
| DRN-260926-209 | P345 | [registry] bengio_deliver_…_seed20260903_elora_shared | E-LoRA arm B 시드 903 | - | registry, deliver, seg, status-unknown |
| DRN-260926-212 | P348 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E1_confirm200_v2 | E1 확정 시드4 (새 체크아웃, TAPS 적용) | - | registry, deliver, seg, status-unknown |
| DRN-260926-213 | P349 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260904_v2 | C3-only 200ep 시드4 (새 체크아웃) — E1 시드4 v2 의 같은 코드 짝 | - | registry, deliver, seg, status-unknown |
| DRN-260926-214 | P350 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E13_confirm200 | E13 확정 시드4 (4탭 + 센서별 prototype) — E1 시드4 v2 와 같은 서버·체크아웃·시드의 넷째 페어 | - | registry, deliver, seg, status-unknown |
| DRN-260926-215 | P351 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260902_elora_sharedresidual | E-LoRA arm C 시드 902(공유 r8+잔차 r8, 3.93M) | - | registry, deliver, seg, status-unknown |
| DRN-260926-216 | P352 | [registry] jarvis_…_seed20260903_elora_sharedresidual | E-LoRA arm C 시드 903(공유 LoRA r8 + 센서별 잔차 r8) | - | registry, deliver, seg, review:method |
| DRN-260926-217 | P353 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E2s2 | E2 시드2 페어(회색지대 판별, B0s2와 매칭) | - | registry, deliver, seg, status-unknown |
| DRN-260926-220 | P356 | [registry] E3_legal_rescore_seed821 | E3 legal test ckpt epoch25_65.97_top1 | - | registry, deliver, 평가(학습 0), review:method |
| DRN-260926-221 | P357 | [registry] E1_legal_rescore_seed821 | E1 legal 재채점 ckpt epoch35_67.25_top1 | - | registry, deliver, 평가(학습 0), review:method |
| DRN-260926-222 | P358 | [registry] E4_legal_rescore_seed821 | E4 legal 재채점 ckpt epoch15_65.92_top1 | - | registry, deliver, 평가(학습 0), review:method |
| DRN-260926-226 | P362 | [registry] B0_legal_rescore_seed821 | B0 legal 재채점 ckpt epoch40_65.4_top1 | - | registry, deliver, 평가(학습 0), review:method |
| DRN-260926-228 | P364 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E17 | E17 = 4탭 읽기 카드에 고해상도 세부 가지(MODEL.DETAIL_BRANCH)를 더한 40에폭 스크린 — 얇은 객체 병목 직격 | - | registry, deliver, seg, status-unknown |
| DRN-260926-231 | P367 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2 | P54 Q2 = 증류 대조군: E1 40ep 스크린 레시피 + 두 패스 학습(clean CE 상시 + 열화 입력 CE + 동결 E1 확정 시드1 교사 KD), 품질 헤드·QAF 융합 없음 | - | registry, deliver, seg, review:method |
| DRN-260926-232 | P368 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q3 | P54 Q3 = QAF 본 카드: Q2 + 품질 헤드(η̂)·depth·LiDAR key 마스크·self-attn 1층·가중 평균 | - | registry, deliver, seg, review:method |
| DRN-260926-235 | P376 | [registry] det_P29_egofill_bengio | P29-Det | - | registry, poongsan, det, review:method |
| DRN-260926-236 | P377 | [registry] det_P29_event_bengio | P29-Det | - | registry, poongsan, det, review:method |
| DRN-260926-237 | P378 | [registry] det_P29_final_full | P29-Det | - | registry, poongsan, det, status-unknown |
| DRN-260926-238 | P379 | [registry] det_P29_v2 (재학습) | P29-Det | - | registry, poongsan, det, review:method |
| DRN-260926-239 | P380 | [registry] det_P31_v3clip_jarvis | P31.1-Det (calibrated rel. + decisive router + FCOS) | - | registry, poongsan, det, review:method |
| DRN-260926-240 | P381 | [registry] det_P30_v2 | P30-Det (router+query decoder) | - | registry, poongsan, det, review:method |
| DRN-260926-241 | P382 | [registry] YOLO11m RGB-only 기준점 (E1.1b/c) | YOLO11m (외부 head, RGB-only) | - | registry, poongsan, det, review:method |
| DRN-260926-242 | P383 | [registry] jarvis_muses_probea2_backbone_scaling | ProbeA2(frozen DINOv3 S+/B/L/H+/7B + 공용 경량 head, 비교용) | - | registry, muses, probe, review:method |
| DRN-260926-293 | P600 | [세대] MemorySAM 초기 기준선(SAM·LoRASam·P4~P7·재구성) | SAM2 기반 초기 기준선 | - | generation, review:method, status-unknown |
| DRN-260926-257 | P608 | [세대] P8: ConfidenceHeadV2 + Sigmoid UAMM | P8: ConfidenceHeadV2 + Sigmoid UAMM | - | generation, review:method |
| DRN-260926-258 | P609 | [세대] P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선) | P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선) | - | generation, review:method |
| DRN-260926-261 | P612 | [세대] P12: Input-Conditioned Soft MoE LoRA | P12: Input-Conditioned Soft MoE LoRA | - | generation, review:method |
| DRN-260926-262 | P613 | [세대] P13: Energy Score Fusion + Expert Collapse Fix | P13: Energy Score Fusion + Expert Collapse Fix | - | generation, review:method |
| DRN-260926-263 | P614 | [세대] P14: Per-Modality Separate Aux Decoders | P14: Per-Modality Separate Aux Decoders | - | generation, review:method |
| DRN-260926-265 | P616 | [세대] P16: Calibrated Spatial Entropy Fusion (P15 설계의 구현 버전) | P16: Calibrated Spatial Entropy Fusion (P15 설계의 구현 버전) | - | generation, review:method |
| DRN-260926-266 | P617 | [세대] P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy Fusion | P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy Fusion | - | generation, review:method |
| DRN-260926-267 | P618 | [세대] P18: Trainable ResNet-18 Aux Backbone + Configurable Fusion | P18: Trainable ResNet-18 Aux Backbone + Configurable Fusion | - | generation, review:method |
| DRN-260926-268 | P619 | [세대] P19: Learned Spatial Cross-Modal Fusion (SpatialCrossModalFusionHead) | P19: Learned Spatial Cross-Modal Fusion (SpatialCrossModalFusionHead) | - | generation, review:method |
| DRN-260926-269 | P620 | [세대] P20: Shared MLP Gate + Higher Rank MoE (실험 J-A) | P20: Shared MLP Gate + Higher Rank MoE (실험 J-A) | - | generation, review:method |
| DRN-260926-270 | P621 | [세대] P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid) (실험 K) | P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid) (실험 K) | - | generation, review:method |
| DRN-260926-271 | P622 | [세대] P22: Multi-Scale DeBA-FP (all FPN levels, Phase 1) (실험 L) | P22: Multi-Scale DeBA-FP (all FPN levels, Phase 1) (실험 L) | - | generation, review:method |
| DRN-260926-273 | P624 | [세대] P24: P9 + Quality-aware Memory Gating via Per-Modality Decoder Distillation (실험 N) | P24: P9 + Quality-aware Memory Gating via Per-Modality Decoder Distillation (실험 N) | - | generation, review:method |
| DRN-260926-276 | P627 | [세대] P27: Additive Attention Bias on Cross-Modal Memory Attention (RBMA 전구체, 2026-04-14) | P27: Additive Attention Bias on Cross-Modal Memory Attention (RBMA 전구체, 2026-04-14) | - | generation, review:method |
| DRN-260926-256 | P628 | [세대] P28: RBMA — Reliability-Biased Memory Attention (2026-06-15) | P28: RBMA — Reliability-Biased Memory Attention (2026-06-15) | - | generation, review:method |
| DRN-260926-255 | P629 | [세대] P29: SDC — Self-Derived Condition 라우팅 (2026-06-27) | P29: SDC — Self-Derived Condition 라우팅 (2026-06-27) | - | generation, review:method |
| DRN-260926-291 | P629 | [세대] P29-Det · RBMA 백본 + FPN/FCOS 검출(poongsan) | P29-Det | - | det, generation, review:method |
| DRN-260926-253 | P630 | [세대] P30-Det — P30 백본 detection 확장: Reliability-router 융합 + Object-Query decoder + FCOS aux (2026-06-30) | P30-Det — P30 백본 detection 확장: Reliability-router 융합 + Object-Query decoder + FCOS aux (2026-06-30) | - | generation, review:method |
| DRN-260926-254 | P630 | [세대] P30: Class-token decoder + Reliability-anchored learned router (2026-06-28) | P30: Class-token decoder + Reliability-anchored learned router (2026-06-28) | - | generation, review:method |
| DRN-260926-252 | P631 | [세대] P31: Calibrated Dual-Reliability RBMA + Multi-scale HR Class-Token Decoding (2026-07-02) | P31: Calibrated Dual-Reliability RBMA + Multi-scale HR Class-Token Decoding (2026-07-02) | - | generation, review:method |
| DRN-260926-251 | P632 | [세대] P32: CoRB — Corroboration-Biased Memory Attention (2026-07-06) | P32: CoRB — Corroboration-Biased Memory Attention (2026-07-06) | - | generation, review:method |
| DRN-260926-277 | P633 | [세대] P33: CG-MoD — Competence-Gated Hard Fusion + Modality Dropout (2026-07-07) | P33: CG-MoD — Competence-Gated Hard Fusion + Modality Dropout (2026-07-07) | - | generation, review:method |
| DRN-260926-278 | P634 | [세대] P34: ReliaDINO — DINOv3-L frozen + per-modal LoRA (계보 전환점, 2026-07-13 완주) | P34: ReliaDINO — DINOv3-L frozen + per-modal LoRA (계보 전환점, 2026-07-13 완주) | - | generation, review:method |
| DRN-260926-279 | P635 | [세대] P35: 공정 레시피 동결 (P34 − ATTN_BIAS − CONSISTENCY − PhysAug, 2026-07-15) | P35: 공정 레시피 동결 (P34 − ATTN_BIAS − CONSISTENCY − PhysAug, 2026-07-15) | - | generation, review:method |
| DRN-260926-280 | P636 | [세대] P36: Per-Class Reliability-Anchored Router (= P35 + router, 2026-07-15 완주) | P36: Per-Class Reliability-Anchored Router (= P35 + router, 2026-07-15 완주) | - | generation, review:method |
| DRN-260926-281 | P637 | [세대] P37a / P37b: CEFR-Head · ClassToken-lite-Learned (2026-07-17~18) | P37a / P37b: CEFR-Head · ClassToken-lite-Learned (2026-07-17~18) | - | generation, review:method |
| DRN-260926-247 | P638 | [세대] P38: MaskQueryLite — Mask2Former-lite Query Head (2026-07-18 launch) | P38: MaskQueryLite — Mask2Former-lite Query Head (2026-07-18 launch) | - | generation, review:method |
| DRN-260926-248 | P639 | [세대] P39: DPC — Dual-Path Compete (2026-07-20) | P39: DPC — Dual-Path Compete (2026-07-20) | - | generation, review:method |
| DRN-260926-249 | P639 | [세대] P39.1: Rank 수리 — gated_mlp trunk + VICReg (2026-07-21) ★ 현행 기준선 | P39.1: Rank 수리 — gated_mlp trunk + VICReg (2026-07-21) ★ 현행 기준선 | - | generation, review:method |
| DRN-260926-250 | P640 | [세대] P40: RCA-Fusion — Reliability-Conditioned Attenuation (2026-07-21) | P40: RCA-Fusion — Reliability-Conditioned Attenuation (2026-07-21) | - | generation, review:method |
| DRN-260926-282 | P641 | [세대] P41: FCR — Fusion Spectral Collapse / Fused Class-alignment Regularizer (2026-07-22~23) | P41: FCR — Fusion Spectral Collapse / Fused Class-alignment Regularizer (2026-07-22~23) | - | generation, review:method |
| DRN-260926-283 | P642 | [세대] P42: lidar-강제 — 조건부 균형 img 마스킹 (2026-07-23) | P42: lidar-강제 — 조건부 균형 img 마스킹 (2026-07-23) | - | generation, review:method |
| DRN-260926-245 | P643 | [세대] P43: PanopticDual — 독립 주손실 mask-classification 헤드 (2026-07-25) | P43: PanopticDual — 독립 주손실 mask-classification 헤드 (2026-07-25) | - | generation, review:method |
| DRN-260926-246 | P644 | [세대] P44: BMR — Balanced Multimodal Reliability (+ P45 FogStyle) (2026-07-25) | P44: BMR — Balanced Multimodal Reliability (+ P45 FogStyle) (2026-07-25) | - | generation, review:method |
| DRN-260926-244 | P646 | [세대] P46: CTR — Class-Transfer Recovery (RCS + MCC + Prototype) (2026-07-29 ~ 2026-08-05) | P46: CTR — Class-Transfer Recovery (RCS + MCC + Prototype) (2026-07-29 ~ 2026-08-05) | - | generation, review:method |
| DRN-260926-243 | P647 | [세대] P47-2: UniBal — Uni-modal Balance (구 D-2, 2026-08-04) | P47-2: UniBal — Uni-modal Balance (구 D-2, 2026-08-04) | - | generation, review:method |
| DRN-260926-285 | P647 | [세대] P47-1: LiDAR 투영 밀도화 (구 D-1, 2026-08-03) | P47-1: LiDAR 투영 밀도화 (구 D-1, 2026-08-03) | - | generation, review:method |
| DRN-260926-286 | P648 | [세대] P48: 쿼리 경로 인스턴스 감독 (2026-08-05) — 제안 단계 | P48: 쿼리 경로 인스턴스 감독 (2026-08-05) — 제안 단계 | - | generation, review:method |
| DRN-260926-78 | P649 | P49 · AIR 비대칭 주입 | P49-AIR | - | generation, review:method |
| DRN-260926-79 | P650 | P50 · MAP 모달 정렬 사전학습 | P50-MAP Phase1/2 | - | generation, review:method |
| DRN-260926-80 | P651 | P51 · CMLC 교차모달 LoRA 결합 | P51-CMLC | - | generation, review:method |
| DRN-260926-81 | P660 | condexpert 어댑터 프로브 | 조건 전문가 어댑터 프로브 | - | probe, status-unknown, review:method |
| DRN-260926-82 | P661 | H10 재판정 실험 요청 | H10 재판정 | - | probe, status-unknown, review:method |
| DRN-260926-83 | P662 | 공간 모달 오라클 프로브 | 공간 모달 오라클 프로브 | - | probe, status-unknown, review:method |
| DRN-260926-84 | P663 | P36 노벨티 비판 검토 | 문서 검토(실험 아님) | - | review, status-unknown, review:method |
| DRN-260926-294 | P700 | 실험 I · P9 Gamma TTA (MULTIAQUA) | 추론 시 감마 TTA | - | multiaqua, review:method |
| DRN-260926-295 | P701 | 실험 II·III · I2I Translation (MULTIAQUA) | 야간→주간 I2I 변환 입력 | - | multiaqua, review:method |
| DRN-260926-296 | P702 | 실험 IV · P9 CV Heuristic Enhancement | 휴리스틱 영상 보정 후 재추론 | - | multiaqua, review:method |
| DRN-260926-298 | P704 | 실험 VI · P9 PhysAug | PhysAug 증강 | - | multiaqua, review:method |
| DRN-260926-299 | P705 | P9 증강 ablation 계열(hardaug2~8·night2·noCRM·tta, MULTIAQUA) | NIGHT_AUG·hardaug 수준 | - | multiaqua, review:method |
| DRN-260926-300 | P706 | Tiled Inference 실험 (2026-03-17) | 타일 추론 | - | multiaqua, review:method |
| DRN-260926-301 | P707 | [진단] P25 Spatial Quality Gating 야간 Pred Q 분석 (2026-06-14) | 학습 0 진단 | - | multiaqua, analysis, review:method |
| DRN-260926-302 | P708 | P26 DELIVER AMP on + GC off 단일 GPU 메모리 프로브 (2026-04-10) | AMP on·GC off 메모리 측정 | - | deliver, probe, review:method, status-unknown |
| DRN-260926-303 | P709 | 아키텍처 심층 분석 A·B (UAMM 실효성, P9 vs MMSamBase, 2026-03-24) | 학습 0 분석 | - | multiaqua, analysis, review:method |

| id | 판정 | 요약 | 분석 |
| --- | --- | --- | --- |
| DRN-260926-01 | adopt | E1(중간층 4탭 읽기)·E13(4탭 + 센서별 클래스 prototype) 확정 3쌍 채택: legal v2 로 C3-only 기준선 대비 E1 Δ25 +1.21±0.81 / Δ24 +0.92±0.22, E13 +1.45±0.88 / +0.80±0.38 → 사전 게이트(Δ25 ≥ +1.0, Δ24 ≥ +0.5) 두 카드 통과. 헤드라인 카드 = E1, E13 병기. 단 SOTA 거리 게이트(우리 최고 단일 런 56.99·DGFusion 56.71)는 미달. 판정 대장 2026-09-20 행(34행) 확인 | .claude_logs/experiments/judgment-ledger.md:34 |
| DRN-261012-01 | reject | 시드821 legal v2(같은 평가 해상도 1536 대비 Q2): 24클래스 test −1.24 / val −1.02, 얇은4 test −1.01 / val −3.80. 손실은 맥락 의존 클래스(RailTrack −10.3·Wall −5.4·Terrain −4.9·Water −3.6). 학습 스케일 상향은 얇은 객체도 못 올리고 맥락만 잃음 — 평가 해상도 이득은 추론 시 토큰 밀도에서 나옴. 시드902 는 user 지시(10-15)로 1에폭에서 중단·A2 s902 로 교체, 2시드 미충족은 user 결정. 판정 대장 f3c539e·e35a0d6 | - |
| DRN-261016-01 | adopt | 같은 규칙(원본 1042 GT·val 선택 해상도 1536) 비교 확정: 1024 재현 ±0.02 통과. 기준선은 val 이 1536 을 고르지만 test 하락(DGFusion 56.71→55.57, CAFuser(b) 55.77→55.60), 우리는 상승(E1 56.24→57.48, Q2 56.59→58.12). val 선택 규칙에서 Q2 +2.55 vs DGFusion. 해상도 이득은 우리 구조에만 일반화 — 프로토콜 결과이지 노벨티 아님, 헤드라인 규칙 변경은 user 결정 | .claude_logs/tasks/DRN-261016-01/analysis.md |
| DRN-261018-01 | adopt | A2(E1 구조 + Q2 두 패스 학습 + 고해상 경로) 3번째 시드(903) legal v2 test 57.52·val 69.86. 사전 기준 충족: 3시드 평균 24클래스 Δ vs Q2 같은 시드 +1.50(기준 ≥ +0.5), 시드903 단독 +2.32(기준 > 0). 3시드 25클래스 test {58.61, 57.09, 57.52} 평균 57.74(DGFusion 56.71 +1.03), 24클래스 58.56. → A2 를 논문 모델로 확정(DRN-261012-02 inconclusive 의 후속 판정). 주의: TrafficLight 3시드 평균 −4.14(시드902 −10.51 특이), RailTrack 시드 요동 큼(3시드 평균 −7.16). | .claude_logs/tasks/DRN-261018-01/analysis.md |
| DRN-261019-03 | adopt | A2 강건성(같은 프로토콜 mm_eval_v2, 3시드): test EMM15 평균 48.66 vs Q2 48.06(Δ +0.61, 기준 ≥ −1.0 충족), val EMM15 평균 57.85(공개 최고 RMMSS 49.93 +7.92, 기준 충족), val RMM15 r.25/.5/.75 = 65.53/63.35/61.28(Q2 +0.6~1.5). 고해상 경로가 결측 강건성을 해치지 않음. 단 노이즈 S&P 0.2 에서 −0.94(Q2 2시드), depth·RGB 부분 결측 저하 0.15~0.50 큼 — 논문 한계 절에 기록. | .claude_logs/tasks/DRN-261019-03/analysis.md |
| DRN-261028-01 | reject | 고해상 신호 프로브(A2 s821, DELIVER test 1897장, 이미지 단위 AUROC 양방향 최대): 고해상 s4 토큰 통계 vs 블록 6 통계 = MB 0.655/0.617 · OE 0.788/0.756 · UE 0.942/0.963 · LJ 0.624/0.586 · EL 0.612/0.749. 사전 기준(고해상 ≥0.8 이 ≥3 파티션 그리고 블록 6 보다 ≥0.15) 미달: ≥0.8 은 1/5, +0.15 이상 0/5(최대 +0.04). 토큰 단위는 두 신호원 모두 ≤0.78. 결론: '블록 6 에 신호가 없어 P56-C 가 붕괴했다' 는 가설 기각 — 밝기 계열(UE·night·fog)은 두 신호원 모두 이미지 단위로 갈리고, 토큰 단위 열화 신호는 어느 쪽에도 약함. 토큰 적응 LoRA 축은 신호원과 무관하게 종결(반복 금지). | .claude_logs/tasks/DRN-261028-01/analysis.md |
| DRN-260926-02 | reject | MUSES 200에폭 풀 런(E1M·E7·E13M 시드3407, 시드902 짝) 공식 재채점: E13M 3407 81.95 · E13M 902 81.68 · E1M 3407 81.70 · E1M 902 81.93 · E7(기준선 카드) 81.88 → 2시드 평균 E13M 81.81 ≈ E1M 81.82, 기준선이 그 사이. 40에폭 스크린에서 보였던 E13M 이득이 풀 런에서 사라져 MUSES 에서 4탭·prototype 카드는 이득 없음. 제출본 교체 근거 없음. 판정 대장 2026-09-20 행(36행) 확인 | .claude_logs/experiments/judgment-ledger.md:36 |
| DRN-261012-02 | inconclusive | 사전 성공 기준 3개 중 1개 충족·반증 조건 미발동. 2시드 legal v2 test@1024: 24클래스 Δ vs Q2 같은 시드 +1.08(+0.99/+1.18, 기준 +0.5 충족), 25클래스 57.85(DGFusion 56.71 대비 +1.14). 얇은4 Δ +1.39 — 사전 문턱('A1 얇은4 이득의 절반')이 A1 이득 −1.01 로 정의 불능(문자대로 −0.5 통과·대체 +1.5 미달·반증 +0.5 통과), 결과 확인 후 문턱을 고르지 않음. 두께 0.25~0.5 패치 '찾았지만 못 그림' .372/.374 로 기준 ≤.35 미달(감소 .072/.054 는 반증 .05 통과). TrafficLight −6.60(FP 증가). 성능 이득은 두 시드 일관 — 채택 여부·3번째 시드·MUSES/MCubeS 이식은 user 결정 | .claude_logs/tasks/DRN-261012-02/analysis.md |
| DRN-261020-09 | adopt | ablation 표 행 확정(측정값 채택, 1시드 s821 40ep legal v2 test@1024): A2 에서 센서별 LoRA 를 완전공유 LoRA(r16) 로 바꾸면 test 58.84 / val 70.36 — A2 같은 시드(58.61 / 68.93 학습기) 대비 test +0.23. 즉 clean 정확도에서 센서별 LoRA 의 기여는 없음(E-LoRA 09-21 관찰과 일치: 공유 LoRA 가 clean 동급·depth 열화 내성 우위). 논문 서술: 센서별 LoRA 는 '기여' 가 아니라 설계 선택으로 적고, 강건성 축(EMM) 비교를 추가 요청. | .claude_logs/tasks/DRN-261020-09/plan.md |
| DRN-261020-10 | adopt | ablation 표 행 확정(1시드 s821): A2 에서 교차 모달 attention 융합을 센서 토큰 평균(MeanFusionTrunk, TRUNK_EXP off)으로 바꾸면 test 57.76 / val 70.80 — A2 58.61 대비 test −0.85. 교차주의 융합(+트렁크 확장)의 clean 기여 약 +0.9(코드 제약으로 두 가지가 묶여 분리 불가). | .claude_logs/tasks/DRN-261020-10/plan.md |
| DRN-261020-12 | adopt | ablation 표 행 확정(1시드 s821): A2 에서 질의 헤드+arbiter 를 빼고 dense conv 헤드만 쓰면 test 57.89 / val 70.56 — A2 58.61 대비 test −0.72. 질의 헤드 결합의 clean 기여 약 +0.7. | .claude_logs/tasks/DRN-261020-12/plan.md |
| DRN-260926-03 | inconclusive | E-LoRA 3갈래(A 센서별 r16 / B 완전공유 r16 / C 공유 r8 + 센서별 잔차 r8): 축별로 결론이 다르다. clean·모달 결측에서는 구조 무관, depth 절반 열화(RMM r=0.5)에서는 B 가 손해를 절반 이하(−6.82 vs A −15.77)로 줄이고 시드 분산도 1/5, clean 대가는 legal v2 val 약 −1. C 는 둘을 합치지 못했다. 단독 채택·기각 없이 후속 카드 E1-shared(4탭 위 완전공유 LoRA) 로 넘김 → 그 결과는 DRN-260926-57 에서 판정. 판정 대장 2026-09-21 행(49행) 확인 | .claude_logs/experiments/judgment-ledger.md:49 |
| DRN-261020-08 | adopt | 백본 사다리 0단계 확정(1시드 s821 40ep clean): 동결 DINOv3-L + LoRA RGB 단독 test 55.14 / val 61.78. 같은 시드 사다리: RGB 단독 55.14 → E1(4센서 교차주의 융합, clean) 55.94 → Q2(+두 패스·증류) 57.73 → A2(+고해상) 58.61. 백본 단독·clean 융합 모두 DGFusion 56.71 미달 → '백본 덕' 반박 근거. 융합 자체의 clean 이득은 +0.8 로 작음(정직하게 서술). | .claude_logs/tasks/DRN-261020-08/plan.md |
| DRN-260926-04 | reject | E17(E1 레시피 + 고해상도 세부 가지 DETAIL_BRANCH) 40에폭 스크린 시드821: legal v2 Δtest +0.88(사전 게이트 +1.0 미달), Δval +1.47. 확정 200에폭 2런은 user 지시로 취소. 이후 얇은·원거리 객체 결함을 겨냥한 R1·R2 도 실패해 세 축 모두 기각. 판정 대장 2026-09-20 행(35행) 확인 | .claude_logs/experiments/judgment-ledger.md:35 |
| DRN-260926-10 | reject | 확정 레시피의 MCubeS 이식 3시드(200에폭): 4탭 단독 3짝 평균 +0.10, 4탭 + prototype 3시드 평균 −0.63 으로 둘 다 게이트(+0.5) 미달. 짝 간 폭 0.64 가 평균 효과의 여섯 배라 시드를 더 늘려도 판정이 바뀌지 않는다고 사전 종결 조건대로 종결. MCubeS 헤드라인은 C3-off 58.07±0.49 유지. plan.md N-MC 행(2026-09-16) 확인 | .claude_logs/experiments/plan.md (N-MC 행) |
| DRN-260930-02 | adopt | 측정 항목(채택 = 결과를 기록으로 확정) + 가설 판정: DGFusion (a) 발표 재현 vs (b) 열화 재학습, 같은 프로토콜(test 1,897장·시드0·61케이스). EMM 15조합 평균 31.53 → 45.52 로 가설 전반 채택(열화 학습이 결측 강건성을 줌). 후반(depth 의존 유지)은 (b) 에서 반증(depth 통째 결측 −33.3 → −4.7). 우리 Q2 3시드는 (b) 대비 clean +1.8, EMM +2.5(48.06), depth 부분 저하 약 절반(0.98/2.04/2.64 vs 2.51/2.88/3.47), 노이즈 저하 1/2~1/4 로 모든 축 우위 → 강건 우위 주장 성립(단일 런·백본 차이 각주). 원본 NAS analysis_logs/baseline_robust_dgfusion_{a,b}_20261001, 판정 대장 2026-10-01 행 | 정정(ISSUE-041, 2026-10-02): (b) 는 '같은 열화 커리큘럼' 이 아니라 학습 내내 최대 강도 열화(severity 상한 1.0 고정, spawn 워커가 커리큘럼 단계를 못 받는 코드 결함). 강건 측정값은 유효, 판정(Q2 전 축 우위) 유지·보수적. (b) 의 clean −4.80 은 최대 강도 열화 탓일 수 있어 기준선 clean 희생 서술은 금지 |
| DRN-260926-11 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 분산. 공식 val 3점 {82.13, 81.79, 81.47} 폭 0.66 으로 val 은 안정하나, 공식 test 2점 {79.788, 78.786} 폭 1.00 으로 val 폭이 test 에서 커진다. 결론: 단일 제출 방어 근거로 못 쓰며 MUSES 도 시드 평균 병기 필요. 원문 = analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md | .claude_logs/experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md |
| DRN-260926-12 | reject | MLE-SAM 식 평균 융합 기준선(믹서를 평균으로 교체): legal test 55.45 로 우리 gated-MLP trunk(54.2~55.4)와 동급 이상. 'trunk 가 우위'라는 주장은 철회하고 믹서 3점(평균 ≈ gated-MLP > 교차 attention 만) 을 기록. 기각 = 우리 trunk 의 우위 가설 기각. 원문 = analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md | .claude_logs/experiments/analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md |
| DRN-260926-14 | adopt | MCubeS 첫 진입 3시드(P39.1-rank 통일 레시피, C3 off): test {57.93, 57.67, 58.62} = 58.07±0.49, published 최고 54.65 대비 +3.42, StitchFusion 55.9 대비 +2.17 → MCubeS 헤드라인으로 채택(3시드 평균 병기). 원문 = analysis/2026-08-25-n4-mcubes-first-entry-verdict.md, plan.md N4 행 | .claude_logs/experiments/analysis/2026-08-25-n4-mcubes-first-entry-verdict.md |
| DRN-260926-15 | adopt | MCubeS C3(클래스 prototype 손실) on 짝: 사전 등록한 예측 2/2 적중 — rubber 클래스 +9.76(18.80→28.56, published 대역 복귀), 전체 mIoU Δ −0.10(허용 범위 안) → C3 의 용량-반응 3점(DELIVER·MUSES·MCubeS) 성립. 채택 = 기제 가설 확인. 헤드라인은 C3 off 3시드 유지. 원문 = analysis/2026-08-27-n4b-dose-response-confirmed.md | .claude_logs/experiments/analysis/2026-08-27-n4b-dose-response-confirmed.md |
| DRN-260926-16 | adopt | DELIVER 체크포인트 선택을 legal-val 기준으로 재선택(5시드): 평균 53.82 → 54.39±0.76, 선택 아티팩트가 2/5런에서 실재(816 +2.21, 821 +0.63), 최고 단일 런 = 시드816 55.29. 채택 = 프로토콜 확정: 이후 legal 수치는 하네스 가드 --check 필수, 체크포인트 = 학습기 val-best top1 규칙과 병기. 원문 = plan.md N6 행(2026-08-31) | .claude_logs/experiments/plan.md (N6 행) |
| DRN-260926-17 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): N7 VICReg-off 격리(P46 C3-only λ0.1, 시드821 짝, yeon 200에폭): val-best 66.56@ep32 로 본 런(67.79@70) 대비 학습기 val −1.23. legal test 는 val-best 시점 미측정(test-best 56.31 은 인용 금지). VICReg 기여는 val 축에서만 확인(DRN-260926-154 과 같은 런) | - |
| DRN-260926-18 | adopt | P50(정렬 사전학습 init) 런 6개 체크포인트 legal-val 스윕: ep30(학습기 top1) = legal-val-best 로 동일, 재선택 변화 없음 → P50 legal test 54.95 를 최종 확정하고 P52 게이트 G1 기준으로 채택. 원문 = plan.md N8 행(2026-09-01) | .claude_logs/experiments/plan.md (N8 행) |
| DRN-260929-01 | reject | P56-A 모달 충돌 학습(Q2 열화 패스 30% 에 '선명하지만 틀린 센서' 합성 표본) 40ep 스크린 2시드 legal v2: clean 24클래스 Δ vs Q2 같은 시드 짝 −0.98(시드821) / −0.56(시드902), 평균 −0.77 → 사전 기준 ①(≥ −0.3) 미달. 실제 과·저노출·모션블러 3조건 평균 Δ −1.55(시드821, 기준 ②-a ≥ +1.0 반대 방향, 10조건 전부 음수). 하락 클래스 = Static·Terrain·얇은 4. 합성 충돌 표본은 실제 열화로 전이되지 않음(Q1d 와 같은 축) → 합성 충돌·열화 라벨 기반 표본 조작 계열 재시도 금지(H33). ②-b 진단(부분집합 380장 강건 비교)은 진행 중이나 판정 불변. 원본 NAS ckpts/…P56A_20261004/(두 시드), analysis_logs/p56a_s{821,902}_eval_20261004/, 판정 대장 2026-10-04 행 | 판정 시 병기(user 승인 09-30): ①조건별 leave-one-out(oracle_headroom_probe --by_condition --emm_max_missing 1, Q2 같은 시드 짝 대비 RGB·depth 제거 Δ 이동) ②합성 충돌 영역에서 충돌 센서 key 에 두는 attention 질량 비율(도구 신설, Q2 대비) ③실제 케이스별 mIoU. 설계서 decisions/2026-09-30-p56-bc-modality-aware-design.md §3 |
| DRN-260929-02 | reject | P56-B 거리 조건화 attention(교차 attention 에 −λ·|Δlog r| 거리 편향) 시드821 40ep 스크린 legal v2: test 55.36 / 24클래스 56.43(Q2 시드821 대비 Δ24 −1.27, Q2 3시드 평균 대비 −0.63) → 사전 기준 ①(≥ −0.3) 미달. 기제 진단: 원거리 78~255m −0.32(기준 +1.0 미달), 근·중거리 −2.21/−3.95 → 거리 편향이 원거리를 올리지 못하고 근거리 문맥을 깎음(기제 미작동). 하락 클래스 Static·Terrain·Pole·Ground 는 P56-A 와 같은 집합. 시드902 는 user 승인으로 중단(2026-10-05, yeon GPU1 을 Q2noKD 에 재배치) → 단일 시드 + 기제 미작동으로 기각. 판정 대장 2026-10-05 행 | - |
| DRN-260929-03 | reject | P56-C 상태 조건부 LoRA 혼합 2시드 40ep: clean 24클래스 2시드 평균 Δ +0.04(① 통과, 3시드 평균 대비 +0.31, 시드 간 편차 0.03). 라우터 α 토큰 간 std ≈0(img·depth), event 0.078, LiDAR 0.00006, 조건별 변화 없음 → 사전 기준 ③(≥0.05) 미달, 라우터가 모달별 상수로 붕괴해 '상태 조건부' 기제가 작동하지 않음. 강건(같은 시드821 부분집합 짝): depth 부분 저하 비율 0.31~0.36, RGB 0.95/0.75/0.74, EMM −0.71 — 시드 잡음 커 판정 근거 아님. P56 A·B·C 전부 기각, 논문 모델 E1 구조 + Q2 레시피 고정. 판정 대장 2026-10-08 행, H34 | - |
| DRN-260926-23 | adopt | 측정 항목(채택 = 공정선 기록으로 확정): MUSES 최선 레시피(P39.1-rank 3모달)에서 PhysAug 만 끈 시드 3짝(824·825·826): 공식 val 82.463 / 82.19 / 82.256 = 82.30±0.14. PhysAug-on 같은 레시피 시드2 82.62 대비 −0.3 안. 공정선 헤드라인 후보이며 test 제출은 user 결정으로 보류(2026-09-29, 새 4모달 모델 전까지). 판정 대장 2026-09-24·25 행 | experiments/judgment-ledger.md 행 74,75,76 |
| DRN-260926-25 | reject | E0/S0 특징 프로브(학습 = 선형 헤드만, DINOv3 중간층 원 특징 vs 어댑터 후 특징 vs 융합 특징): 원 특징 27.8 < 어댑터 후 35.6 < 융합 45.1 로 '원 특징만으로 충분하다'는 가설 기각. depth 중간층에 Water 47.6·RailTrack 23.9 정보가 남아 있어 중간층 읽기(E1)·prototype(E3) 카드는 유지, E5·E6 은 우선순위 하향. 원문 = cards/2026-W37-verdicts.md E0 행, plan.md 09-07~08 절 | .claude_logs/experiments/plan.md (직전 완결 09-07~08 절) |
| DRN-261001-02 | adopt | 측정 확정 + 서사 판정: E1 레시피 2모달 교사 3런(RGB+Depth·RGB+LiDAR·RGB+Event, 시드821 40ep, bengio 4장 각 6시간) 완주·legal v2·NAS 보존(md5) 완료. legal v2 test RGB+Depth 56.56(24클래스 56.75, 4모달 E1 스크린 55.94/56.39 대비 +0.62/+0.36 = 동률) · RGB+LiDAR 55.31(Δ24 −1.42) · RGB+Event 55.68(Δ24 −1.36). 기준 ②(P46-CTR 54.14 +1.0) 충족. 반증 조건(2모달 > 4모달)은 잡음 범위 발동 → clean 축 센서 가산성 주장 폐기, 4모달 가치는 강건 축에서만. depth 가 핵심 보조 센서(없으면 기하 의존 클래스 −3.6~−11.3). 교사 3개 확보 → N-RGBX 학생(DRN-260930-01) 착수 조건 충족. 후속: RGB-D 결측 3조합으로 강건 축 가산성 확인(요청됨). 원본 NAS ckpts/bengio_deliver_rgb{d,l,e}_…_20261001/, 판정 대장 2026-10-02 행 | - |
| DRN-260926-32 | reject | P50-EXT(정렬 사전학습 코퍼스 500k·등-스텝 확장, Phase2 375,000 step 완주): EXT-init 파인튠 ep30 legal test 53.10 < 채택 게이트 55.25(= 프로브-init 54.95 + 0.3) → 기각, P52 init 은 Phase1(프로브) 로 확정. user 결정 2026-09-07. 원문 = registry hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 행, decisions/2026-08-31-p52-rxdino-adaptive-amendment.md | .claude_logs/experiments/registry.md (hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 행) |
| DRN-261001-03 | adopt | 측정 확정 + 가설 채택: DGFusion 공식 DELIVER 가중치(md5 0f40a641…) clean test 56.73(README 56.71)·val 66.51 재현, 61케이스 같은 프로토콜 EMM 15조합 평균 31.14(a 재현본 31.53, b 열화 재학습 45.52, Q2 3시드 48.06), depth 통째 결측 23.34, depth 부분 저하 7.22/11.72/15.78(Q2 0.98/2.04/2.64), NM S&P 저하 17.61/39.88/52.20(Q2 3.11/4.79/7.68). 반증 조건 해당 없음 → Q2 강건 우위는 공식 가중치 앞에서도 유지. 기준선 표의 (a) 열을 이 값으로 교체. 원본 NAS analysis_logs/baseline_robust_dgfusion_official_deliver_clde_20261001/, 판정 대장 2026-10-01 행 | - |
| DRN-261001-01 | adopt | 측정 확정: DGFusion 공식 MUSES 가중치 val 250장 재현 79.7183(README 79.72, 채점 축 동일) → 재학습 불필요. 우리 PhysAug-off 3시드(82.30±0.14) 대비 val +2.58 이나 공식 test 는 −0.21 이라 이 대조는 val 만 설명. 열세는 얇고 작은 4클래스(pole −6.20·person −5.37·traffic sign −4.34·traffic light −3.16, 3/3시드)와 야간 조건(clear/night −2.98·rain/night −3.01)에 국소 → '전 클래스 균일 열세' 반증 조건 불성립, 가설 채택. 거리 구간 표(LiDAR 희소 화소 한정)는 하지 않음. 다음 = test 측 증거(Codabench 클래스별 IoU vs DGFusion 논문 test 클래스별 표, GPU 불필요). 원본 NAS analysis_logs/muses_baseline_dgfusion_official_20261001/, 판정 대장 2026-10-01 행 | 후속 판정 4건(판정 대장 2026-10-01~02): ③' val→test 전이 손실 −3.52 · H-M1 val-best 선택 편향 기각 · H-M3 부분 채택 · **§10 최종: DGFusion 공식 가중치 Codabench test 79.494 대 우리 78.786(−0.71), 몸통 = 얇은 4클래스 −22.6(val·test 동일 폭의 실재 열세), 희귀 3클래스 val 우위는 test 동률(우리 고유 낙차 약 −12.5)**. H-M2(레시피 수준 val 과적합)는 미판정 보류. 추가 측정 없음 |
| DRN-260926-58 | adopt | 측정 확정 + 가설 채택: CAFuser (b)(최대 강도 열화 재학습, ISSUE-041, iter 200k final) 공식 채점 clean test 55.79(기준 ① 54.38 통과, (a) 대비 +0.41)·val 62.89, 61케이스 EMM 15조합 평균 46.50(DGFusion (b) 45.52, Q2 3시드 48.06), depth 부분 저하 2.59/3.06/3.54(Q2 0.98/2.04/2.64), RGB 부분 저하 2.52/3.39/4.79(Q2 3.27/3.90/4.90 — 동률·약간 우위), NM S&P 저하 7.07/17.62/34.48(Q2 3.11/4.79/7.68), 단일 LiDAR 26.48(Q2 22.9). 반증 조건(EMM ≥ 48.06 또는 depth 부분 ≤ Q2) 해당 없음 → 구조 결합 효과 주장 유지, 단 '모든 축 우위'는 '결측·depth·노이즈 축 우위, RGB 부분 축 동률'로 수정. 원본 NAS analysis_logs/baseline_robust_cafuser_b_20261003/, ckpts/cafuser_b_resume_20261003/ md5 fbae170d…, 판정 대장 2026-10-04 행 | - |
| DRN-260926-45 | adopt | 측정 항목(채택 = 진단 결과를 기록으로 확정): MUSES 성능 저하 원인 A/B 격리. radar(또는 4모달 구조)가 원인이고 lidar 재투영·event dilation·eff batch 는 무관. Arm A ep24 best 73.85@ep18 은 대조군 ep10 74.24 에 앞서지 않아 DGFusion 식 투영은 중립. 원문 = plan.md ✅ 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-46 | reject | TTA(다중 스케일·플립 시험 시 증강)를 헤드라인에 쓸 수 있는가: 경쟁 논문 3종(CMNeXt·CAFuser·DGFusion) 모두 미사용(CMNeXt 는 논문에 single-scale 명시) → 헤드라인 사용 불가로 기각. 우리 MSF config 는 쓰인 적 없어 과거 수치 오염 없음. 원문 = plan.md 완료·판정 표, monitor-log.md TTA 조사 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-47 | adopt | 측정·정합 항목(채택 = 결과를 기록으로 확정): DGFusion 전처리 파라미터 재현 완료(공개 PIXEL_MEAN 을 오라클로 −0.1% 안에서 적중). 실제 남은 차이는 lidar 투영뿐(radar·event 30ms 는 이미 동일). 성능 이득은 0 이고 공정 비교 조건만 확보. 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-48 | reject | SAM2 계보 제안 모듈 ablation: ATTN_BIAS(RBMA 간판 기제) 포함 제안 모듈 전부 효과 ≈0, gate+calibration 만 test +0.26. 성능 출처 = DINOv3 백본 + 센서별 LoRA 로 확정 → 제안 모듈들의 기여 가설 기각(이후 재시도 금지 축의 근거). 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-49 | adopt | 측정 항목(채택 = 진단 결과를 기록으로 확정): det(poongsan) 학습 붕괴 원인 = BS1 의 gradient 잡음(양성 앵커 1~3개), LR 은 원인 아님. 처방 = 배치 증가 + LR 유지, warmup 5에폭 완주로 검증. 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-50 | inconclusive | seg P37a/P37b(bengio 분): bengio 노드 CUDA 전체 장애로 ep1~2 에서 종료된 런 소실. 결과 없음이라 판정 불가. jarvis 재기동분(P37a→P37b 체인)이 계보를 승계함. 원문 = plan.md 완료·판정 표, monitor-log.md | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-51 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 20260825 공식 test 78.786(2026-09-18 수신). 시드2 79.788 대비 −1.00, 2시드 평균 79.29±0.71 < DGFusion 79.5 → '융합 계열 1위' 서술은 시드2 단일 런 한정으로 제한. 시드 20260824 는 체크포인트 부재로 제출 불가. 원문 = analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md | .claude_logs/experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md |
| DRN-260926-52 | inconclusive | P43-PanopticDual(MUSES, 분할 헤드에 panoptic 이중 헤드 추가): 학습기 val 82.51@ep156 로 완주했으나 MUSES 에 panoptic 정답이 없어 PQ 판정이 불가하고 공식 test 제출도 하지 않았다. 이후 MUSES 헤드라인은 P39.1-rank(공식 test 79.788)로 확정돼 P43 은 승계되지 않음. 판정 근거 부족으로 미확정 종결 | - |
| DRN-260926-53 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(P39.1-rank) 완주 val 82.00@ep136. 같은 레시피 3모달(RGB+Event+LiDAR) val 82.22 와 0.22 차이로, MUSES 에서 event 의 기여가 작다는 근거. 공식 test 제출은 하지 않았고 재실행 금지. 원문 = plan.md 2026-08-10 절, registry 행 | - |
| DRN-260926-57 | inconclusive | E1-shared(E1 4탭 레시피 + 전 센서 공유 LoRA r16) 40에폭 스크린 3시드: 게이트 1차(clean 24클래스 Δ ≥ −0.3) 평균 −0.24 로 통과(여유 0.06, 얇은 객체 4클래스 −2.10), 게이트 2차(depth 부분 열화 저하가 E1 의 절반 이하) 비율 0.47 로 통과. 그러나 clean 분산이 커지고(s902 붕괴형 54.46) 효과가 depth 한 모달에 한정돼 단독 채택하지 않음. P56-C(공유 + 센서별 LoRA 전문가 혼합)의 재료로 넘김. 판정 대장 2026-09-23·24 행 | experiments/judgment-ledger.md 행 56,66,68,69,77 |
| DRN-260926-59 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): 기준선(DGFusion·CAFuser)과 우리 모델의 실패 분석 D4/D6. 핵심 결과: 우리 depth 의존은 기준선의 절반(모달 zero-out −17.6 vs −31~−33), 기준선은 event·LiDAR 를 어떤 조건에서도 쓰지 않음(제거 시 ±0.4), 원거리 얇은 객체 결함(찾고도 못 그리는 비율 거리3 0.44 vs 기준선 0.25)이 우리 주 손실 요인. 이 결과가 Q2(두 패스 학습)·P56 설계의 출발점. 원문 = 판정 대장 2026-09-19~20 행 8개, analysis/2026-09-17-baseline-failure-analysis-plan.md | experiments/judgment-ledger.md 행 28,29,30,31,32,33,37,38 |
| DRN-260926-60 | reject | Q1(품질 헤드 실현성 프로브) 기각: Q1b-2 에서 RGB 광학 열화(흐림·감마·색 틀어짐) AUROC 0.67~0.73 으로 인식 실패, Q1d 에서 합성 열화로 학습한 헤드가 실제 열화로 전이되지 않음(다중 블록 혼합 헤드의 실제 야간 AUROC 0.002). 합성 열화 라벨로 감독하는 품질 헤드 계열은 재시도 금지. 판정 대장 2026-09-23·09-28·09-29 행 | experiments/judgment-ledger.md 행 43,50,61 |
| DRN-260926-61 | reject | R1(depth 경계 prior refinement) 종료: 40ep 스크린 3시드 legal v2 test 평균 56.61(E1 스크린 짝 56.58), 24클래스 평균 Δ −0.09, 얇은 객체 4클래스 Δ 시드별 +2.11/−1.57/−1.80 으로 부호 불일치. 기제 게이트(찾고도 못 그린 비율)도 미달. val +1.18 은 test 에서 사라짐. 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 52,54,57,60,65 |
| DRN-260926-62 | reject | R2(연결 성분 단위 soft-IoU 손실, 원거리 가중) 40에폭 스크린: 시드821 두 사본 legal v2 24클래스 Δ −0.03 / +0.06(효과 없음), 기제 게이트 미달(원거리 못 그림 비율은 줄었으나 찾음률도 하락, mIoU 이득은 근거리 구간에만). 시드902·903 은 재채점했다고 기록됐으나 수치가 대장·NAS 에 보존되지 않아 3시드 평균은 미확정 → 시드821 근거와 기제 게이트 미달로 종료. 체크포인트는 NAS ckpts/screen40_bengio_20260923/ 에 있어 필요 시 재채점 가능. 원문 = 판정 대장 2026-09-23·24 행(55·67·72행) | experiments/judgment-ledger.md 행 55,67,72 |
| DRN-260926-63 | adopt | E1 40ep 스크린 3시드(821·902·903) 를 40ep 스크린 카드의 짝 기준선으로 확정: legal v2 test 56.58±0.55, 24클래스 56.65±0.23, 얇은 객체 4클래스 49.60, val 67.51±0.91. 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 64 |
| DRN-260926-64 | reject | E3b(E3 센서별 클래스 prototype + 센서 간 prototype 일치 항, AGREE_LAMBDA 0.1) 40에폭 스크린: legal test 54.10 으로 E3 보다 낮음 → 센서 간 일치를 강제하면 해롭다는 결론으로 폐기. 원문 = cards/2026-W37-verdicts.md | - |
| DRN-260926-65 | reject | E4c(혼동 쌍 margin 을 RailTrack 3쌍으로 좁히고 MARGIN 0.25) 40에폭 스크린: legal test 55.32, 24클래스 Δ +0.13 으로 게이트(Δ25 ≥ +1.0, Δ24 ≥ +0.5) 둘 다 미달 → 폐기. 원문 = cards/2026-W37-verdicts.md §5-12 | - |
| DRN-260926-290 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): det 인증 배포 계열 D1(ViT-S/S+/B 백본, poongsan). ViT-S+ 가 drone-jemo(RTX 5090) 인증 배포본이며 GPU 19.43 fps, 야간 mAP50 0.8292 검증. 국책 목표 mAP50 0.85 는 D1 스윕에서 달성(status 표 0.9321). 원문 = det/det-cert-D1-realtime.md, 브랜치 26-drone-certificate | - |
| DRN-260926-292 | inconclusive | det 백본 이식 계열(P34~P39-Det, RF-DETR, P9 base; poongsan): config 12개만 남아 있고 registry·판정 대장에 결과 기록이 없어 판정 불가. det 트랙은 D1 계열로 목표(mAP50 0.85)를 달성했으므로 이 계열은 재실행하지 않고 미확정으로 종결 | - |
| DRN-260926-304 | inconclusive | P39.1-rank MUSES 3모달 시드4(yeon): monitor-log 에 val 81.92@ep198 정체·약 277/300 에폭에서 '저신뢰, 미완주'로만 남아 있고 config 도 레포에 없다. 완주 기록·체크포인트·재채점이 없어 판정 불가. MUSES 시드 통계는 시드2·824·825·826 으로 대체됨 | - |
| DRN-260926-305 | inconclusive | P44-BMR DELIVER 시드2(yeon): monitor-log 에 '미완주'로만 남아 있고 결과·체크포인트 기록 없음. 판정 불가. P44 계열은 이후 P46/E1 계보로 대체됨 | - |
| DRN-260926-306 | inconclusive | P34 MUSES 4모달 + DGFusion 투영 변형: monitor-log 에 이름만 1회 언급되고 결과 기록이 없어 판정 불가. 투영 정합 자체는 DRN-260926-47 에서 '성능 이득 0, 공정성만 확보'로 확정됨 | - |
| DRN-260926-307 | adopt | 기록 항목(채택 = 이력을 기록으로 확정): MUSES Codabench 공식 test 제출 11건의 이력. 최고 = P39.1-rank 시드2 3모달 79.788, 둘째 시드 20260825 78.786(2시드 평균 79.29). 제출 zip 규약 위치 = /ailab_mat2/.../submission/muses/. 새 제출은 user 만 하며 2026-09-29 user 결정으로 새 4모달 모델 전까지 보류 | - |
| DRN-260926-308 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): DGFusion 기준선 재현 평가. DELIVER 재학습 (a) val-best 80k val 66.54 / test 55.68(final-iter 65.62 / 55.56, 공개 66.51 / 56.71 대비 −1.15), MUSES 는 공개 가중치 val 재현이 감시 세션에서 진행 중. 우리 P39.1 과의 비교는 판정 대장 2026-09-19~23 행에 기록 | - |
| DRN-260926-87 | adopt | MULTIAQUA 챌린지(RGB+Thermal+LiDAR, 종료 트랙) P9(교차 모달 융합 헤드) hardaug8+PhysAug ep131: M-score 82.10(val 93.29 / test 70.91, 재제출 #16710) = 챌린지 공동 1위 제출본. 종료 트랙의 확정 기록으로 채택 | - |
| DRN-260926-88 | adopt | MULTIAQUA 챌린지 P22 hardaug8+PhysAug ep120: M-score 82.10(val 93.42 / test 70.77, #16932) = P9 와 공동 1위 제출본. 종료 트랙의 확정 기록으로 채택 | - |
| DRN-260926-89 | reject | MULTIAQUA 챌린지 P21(DEBA-FP) hardaug8+PhysAug ep94: M-score 81.77(val 93.17 / test 70.36, #16792) 로 P9·P22(82.10) 에 미달 → 최종 제출본으로 채택되지 않음. 종료 트랙 | - |
| DRN-260926-90 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): DGFusion Swin-T 재학습 (a), 발표 config 재현, DELIVER 4모달 200k(yeon). final-iter(공식 프로토콜) val 65.62 / test 55.56, val-best(80k) val 66.54 / test 55.68, 공개 발표값 66.51 / 56.71 대비 −0.89 / −1.15. test-best(100k 55.92)는 인용 금지. 이 값이 우리 모델과의 공정 비교 기준선 | - |
| DRN-260926-91 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): DGFusion Swin-T 재학습 (b) = (a) + 우리와 같은 모달 드롭 p=0.2 + 열화 커리큘럼(jarvis). val-best 90k val 61.74 / test 54.77 → (a) 대비 val −4.80 / test −0.91. 열화 커리큘럼이 DGFusion clean 을 깎는다는 근거(우리 Q2 는 clean 손해 없음). 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 40,58,63 |
| DRN-260926-92 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): CAFuser Swin-T 공식 학습 코드 재학습, DELIVER 4모달 267k(lecun, 2026-09-15 완주). 공식 test 55.38(final), 학습 로그 최종 val 66.04. 체크포인트 스윕 27개의 test 예측이 lecun 에 남아 있음(lecun 은 배치 금지 서버, 파일 회수만 가능). DGFusion (a) 와 함께 공정 비교 기준선 | - |
| DRN-260926-93 | reject | P31(SAM2 계보, PhysAug) DELIVER B200 런: 당시 최선이었으나 B200 상실(2026-07-15 마감) 뒤 P34~P36(ReliaDINO)·P46/E1 계보가 대체. registry 의 '학습 중' 표기는 낡은 것. 계보에서 승계되지 않음 | - |
| DRN-260926-94 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P29·P31·P32·P34 표준분석(lecun, 2026-07-12~13). SAM2 계보 실패-키 문서의 근거. 산출물 NAS analysis_logs | - |
| DRN-260926-95 | reject | P29(SDC) DELIVER B200 런: val 63.20@ep100 / test 54.34@ep146 로 목표(66.51 / 56.71) 대비 −3.31 / −2.37. SAM2 계보 비교 기준선으로만 남고 승계되지 않음 | - |
| DRN-260926-96 | reject | P30 DELIVER B200 런: val 49.76 / test 44.10 으로 P29 대비 −13.4 / −10.2 회귀 확정(dead). 승계되지 않음 | - |
| DRN-260926-98 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P37a-CEFR(클래스 기대 특징 라우팅) MUSES 출력 분석(yeon, 2026-07-18). 라우팅 분화 실패(19클래스 중 0개 committed) → CEFR 라우팅은 반증 축으로 등재 | - |
| DRN-260926-99 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P37a MUSES 표준분석 + 종합 실패-키(yeon, 2026-07-20). fog 62.7 최약, event·lidar 어댑터 기여 확인. 산출물 NAS analysis_logs/P37a_muses_std_20260719/ | - |
| DRN-260926-100 | reject | P39-DPC DELIVER hpca100 런(V1~V5 토글): registry 의 '학습 중' 표기는 낡은 것. 이후 P39.1-rank → P46 → E1 계보가 대체해 승계되지 않음 | - |
| DRN-260926-101 | reject | P39-DPC MUSES jarvis 런: 완주 ep146 val 81.52 / 공식 test 78.881(DRN-260926-103 분석 대상). 이후 P39.1-rank(공식 test 79.788)가 대체해 승계되지 않음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-102 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P39 조기 즉검 + DELIVER 3시점 + 모듈 시각 리포트(yeon, 2026-07-20). P39 V1~V5 토글 기여 분해의 근거 | - |
| DRN-260926-103 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P39-DPC MUSES 3모달 표준분석(yeon, 2026-07-21), ep146 val 81.52 / 공식 test 78.881. lidar 어댑터 기여와 조건별 약점의 근거 | - |
| DRN-260926-104 | reject | SAM3-RBMA(DELIVER 25클래스, SAM3 이식) 계열: '학습/디버깅 중'으로 남았고 완주·판정 기록 없음. B200 상실과 ReliaDINO(DINOv3) 전환으로 중단. 승계되지 않음 | - |
| DRN-260926-105 | reject | P38(MaskQueryLite 쿼리 헤드) DELIVER hpca100 런: val 조기 포화·test plateau 로 ep78+ 에서 중단, 표준분석(DRN-260926-106)에서 쿼리 헤드 추론 기여 no-op. MUSES 쪽 P38-m2f 는 공식 test 79.025 로 7월 기록. DELIVER 는 승계되지 않음 | - |
| DRN-260926-106 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P38-m2f DELIVER 표준분석(yeon, 2026-07-19). D1 평균 53.66(P36 −1.63·P34 −1.99), 쿼리 헤드 추론 기여 no-op → P38 DELIVER 기각의 근거 | - |
| DRN-260926-107 | adopt | P39.1-rank(라우터 랭크 변형) MUSES 3모달 시드2(jarvis): val-best epoch208 82.62, 공식 test 79.788 = 우리 MUSES 공식 최고 단일 런. 헤드라인(best 단일 런 병기)으로 채택. 2시드 평균 79.29 는 DGFusion 79.5 미만이라 '융합 계열 1위'는 단일 런 한정 | - |
| DRN-260926-108 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank MUSES 4모달(radar 추가) 시드2(hpca100): 공식 test 79.571 로 3모달 79.788 대비 −0.22. radar 추가는 이득 없음. 4모달 기록으로 보존 | - |
| DRN-260926-109 | reject | P47-MUB D-1(LiDAR 투영 밀도화, DGFusion 식 projected_to_rgb 7×7 + 모션 보정) MUSES 4모달: val 82.58(+0.23)이지만 공식 test 78.790 으로 4모달 base 79.571 대비 −0.78, 3모달 최고 대비 −1.00. val→test 낙차 3.79 = 밀도화 val 과적합 → 폐기(user 제출 2026-08-17) | - |
| DRN-260926-110 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank MUSES 3모달 시드3(yeon) 완주 val 81.89@ep204. 5시드 val 분산 자료의 일부(시드1 82.03 등). 공식 test 미제출 | - |
| DRN-260926-111 | reject | P43-PanopticDual MUSES 3모달(hpca100): val 82.51 → 공식 test 79.351(우리 2위, 시드2 79.788 대비 −0.44). PQ 판정 불가·test 미달로 승계되지 않음. DRN-260926-52 와 같은 런 | - |
| DRN-260926-113 | reject | P44-BMR MUSES 3모달(hpca100): val-best 80.71@ep156(외부 선점으로 중단), 공식 test 는 P39.1-rank 미달. 승계되지 않음 | - |
| DRN-260926-115 | reject | P46-CTR C1(희소 클래스 재샘플 RCS)+C3(prototype) DELIVER: ep40 대조에서 C3-only(RailTrack 64.13, 전체 55.64)가 C1+C3(59.10, 54.92) 를 앞서 C1 은 순유해 → 2026-07-30 kill. C1 축 폐기 | - |
| DRN-260926-116 | adopt | P46-CTR C3-only(λ0.1, 탭 없음) DELIVER 본 런(jarvis, 시드 고정 기능 없던 무작위 시드): val-best ep70 → 1024 평가 val 69.44 / test 56.99 = 우리 최고 단일 런(구기록). DGFusion 66.51 / 56.71 동시 상회, MM SAM-adapter(69.60 / 57.35)에는 미달. 체크포인트 정본 NAS ckpts/P46_c3only_base_ep70_test5699_20260730/(md5 d340e3fe). 헤드라인은 이후 E1 3시드 legal v2 로 교체됐고 이 값은 단일 런 최고로 병기 | experiments/judgment-ledger.md 행 16,17 |
| DRN-260926-118 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P46-CTR C3-only 재현성 검증 시드2: val-best ep62 → 1024 평가 val 68.90 / test 56.40(본 런 대비 test −0.59). 방향은 재현되나 단일 런 편차 0.6 존재 → 이후 3시드 평균 병기 규칙의 근거 | - |
| DRN-260926-119 | reject | P46 C3-only 를 1024 해상도로 학습한 시드B: legal test 54.85(학습 로그 test-best 57.43 은 비legal). 768 학습 + 1024 평가(56.99)보다 낮아 1024 학습은 폐기 | - |
| DRN-260926-120 | reject | P46 C3-only 1024 해상도 학습 시드C: legal test 54.55. 시드B 와 함께 1024 학습 폐기 확정 | - |
| DRN-260926-121 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): C3-only 768 대표 구성(λ0.1) 진짜 시드 2런(20260815·16, jarvis). registry '학습 중' 표기는 낡은 것 — N6(DRN-260926-16) 5시드 legal-val 재선택에 포함돼 완주·평가됨(5시드 평균 54.39±0.76, 최고 시드816 55.29) | - |
| DRN-260926-123 | reject | P46-CTR C3-only λ=0.2 DELIVER: 완주했으나 SOTA 미달(2026-08-04 정정). λ0.1 이 대표 구성으로 확정 | - |
| DRN-260926-124 | reject | P46-CTR C3-only λ=0.2 MUSES 3모달(hpca100): val 81.65@ep136 로 base(P39.1-rank 시드2 82.62) 대비 −0.97. 손해가 clear/day 에 집중(Δclear −1.72), fog/rain 은 +0.2. DELIVER 기제의 MUSES 이식 실패 → P47 D-2(단일 모달 균형) 근거로 전환 | - |
| DRN-260926-125 | reject | P46-CTR C3-only λ=0.3 DELIVER: λ 스윕 상단 탐색, λ0.1 대표 구성을 넘지 못함(registry 핵심수치 칸). λ 스윕 종결 | - |
| DRN-260926-126 | reject | P46-CTR C3-only λ=0.2 시드2 DELIVER: 재현성 검증 실패, SOTA 미달 확정(2026-08-05). λ0.2 폐기 | - |
| DRN-260926-127 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P47-2 UniBal(모달별 독립 보조 헤드 + 단일 모달 CE) MUSES 4모달, yeon 300에폭 완주 val-best 82.06@ep164, 하네스 가드 재평가 81.72 → P52 게이트 G2 = 81.42 로 확정. 공식 test 미제출. 캘리브레이션 문서 analysis/2026-09-04-p52-unibal-calibration.md | - |
| DRN-260926-128 | reject | P46 C3-only 1024 해상도 본 런(elice B200): registry '학습 중' 표기는 낡은 것. 1024 학습은 시드B·C(DRN-260926-119·120, legal test 54.85·54.55)로 폐기 확정, 768 학습 + 1024 평가가 대표 구성 | - |
| DRN-260926-129 | reject | P46 C3-only 1024 해상도 런(jarvis GPU6,7): 위 128 과 같은 계열. 1024 학습 폐기 확정, registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-130 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank 1024 해상도 대조군(C3 미적용, ep126 조기 종료): 해상도 순효과 val +1.15 / test +2.01 확정 → 1024 평가 규약(legal)의 근거 | - |
| DRN-260926-131 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(P39.1-rank − event) val 82.00@ep136. DRN-260926-53 과 같은 런 | - |
| DRN-260926-132 | reject | cross-attention trunk A/B(#12, XAttnTrunk 대칭 모달 간 attention vs gated-MLP trunk, 나머지 동일): 믹서 3점 비교에서 평균 ≈ gated-MLP > 교차 attention trunk → XAttnTrunk 기각(DRN-260926-12 N2 판정과 같은 근거) | - |
| DRN-260926-133 | adopt | MCubeS 첫 진입 파일럿(P39.1-rank 통일 레시피, C3 off, yeon): 3시드 58.07±0.49 의 첫 런. DRN-260926-14 와 같은 계열, 헤드라인 채택 | - |
| DRN-260926-134 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES P39.1-rank 진짜 시드 20260824: 공식 val 81.79(시드2 82.13 대비 −0.34). N1 시드 분산 자료. 체크포인트 부재로 test 제출 불가(DRN-260926-51 참조) | - |
| DRN-260926-135 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 20260825 공식 test 78.786(2026-09-18). DRN-260926-51 과 같은 항목. 2시드 평균 79.29±0.71 | experiments/judgment-ledger.md 행 15 |
| DRN-260926-136 | adopt | P50-MAP 정렬 사전학습(동결 DINOv3-L + LoRA + trunk 의 cross-modal masked reconstruction, Places365 200k 의사 모달, yeon): 완주해 Phase1(프로브) init 이 P52 에 채택됨(P50 legal test 54.95 = P52 게이트 G1). 확장 Phase2 는 DRN-260926-32 에서 기각. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-137 | adopt | 기록 항목(채택 = 완료 확정): P50 의사 모달 생성(depth = Depth-Anything-V2-Small, lidar·event 합성, Places365 200k). P50 사전학습 입력으로 사용 완료 | - |
| DRN-260926-138 | inconclusive | P49.1-AIR(비대칭 주입, γ init 0.1) MUSES 4모달 런(yeon): 기동 검증만 기록되고 완주·판정 기록이 없다. DELIVER 쪽 P49.1 이 '57.68 SOTA' 철회 후 폐기돼 MUSES 팔도 승계되지 않음. 판정 불가로 종결 | - |
| DRN-260926-139 | inconclusive | P49.1-AIR MUSES 3모달 본 런(yeon): 기동 검증만 기록되고 완주·판정 기록이 없다. DELIVER 쪽 폐기로 승계되지 않음. 판정 불가로 종결 | - |
| DRN-260926-140 | reject | C2-MCC(masked consistency)+C3 vs C3-only A/B(hpca100 재기동, ep134 조기 종료): 1024 평가 55.32 / 768 54.32 vs 본 런 56.99 / 55.91 → Δ −1.67 / −1.59 로 C2 는 유해 확정(가설 대장 H15). C2 축 폐기. 원문 = analysis/2026-08-16-c2-mcc-ab-verdict.md | - |
| DRN-260926-141 | reject | P49.1-AIR DELIVER 768 본 런(yeon, ep150 완주): 학습 로그 test-best 58.30 은 비legal, fair-eval(val.py) test@768 55.66 으로 '57.68 SOTA' 주장 철회(2026-08-14). γ 가 0.10→0.019 로 감쇠해 주입 기제가 아니라 RGB full-FT 가 이득 원천일 가능성. 폐기 | - |
| DRN-260926-142 | reject | P49-AIR DELIVER 768 본 런(yeon): ep30 게이트① 위반(γ 정체 0.0006, val 63.83@50 열세)으로 ep54 중단(가설 대장 H13). 폐기 | - |
| DRN-260926-143 | reject | DELIVER RGB+Depth 2모달(P46 C3-only λ0.05, ep66 val-best) 1024 공정 평가: val 65.79 / test 54.14 로 SOTA RGB-D 대비 −3.81 / −3.21, 우리 4모달 대비 −2.85 → '2모달로 충분' 가설 기각. 원문 = analysis/2026-08-10-rgbd-2modal-fair-eval.md | - |
| DRN-260926-144 | reject | P48(쿼리 경로 인스턴스 감독: 클래스 단위 타깃을 연결 성분 인스턴스 단위로 전환) MUSES 4모달 제안: 게이트 미달로 폐기(2026-08-06). 이후 R2(연결 성분 손실)도 같은 축에서 실패 | - |
| DRN-260926-145 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(event 제거) P39.1-rank, 2026-08-10 미기록 상태로 발견된 완주 런. val 82.00@ep136. DRN-260926-53·131 과 같은 런 | - |
| DRN-260926-146 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): RGB+Depth 2모달 완주 체크포인트의 1024 재평가 작업(학습 0). 결과 val 65.79 / test 54.14 는 DRN-260926-143 에 기록. registry '착수' 표기는 낡은 것 | - |
| DRN-260926-147 | reject | P46 C3-only + P50-EXT Phase2 init 채택 게이트 파인튠(시드821, hpca100): ep30 legal test 53.10 < 게이트 55.25 → 기각(user 결정 2026-09-07). DRN-260926-32 와 같은 항목 | - |
| DRN-260926-148 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A(센서별 LoRA r16, 시드821 짝, yeon) 완주. 3갈래 비교의 A 팔이며 결론은 DRN-260926-03(축별로 다름, 후속 E1-shared)에 기록. registry '착수' 표기는 낡은 것 | - |
| DRN-260926-150 | reject | P52 RxDINO(적응형 C3·UniBal 단일 config) DELIVER 본 런 시드20260901(yeon): legal test 53.87 / val 66.17 / 24클래스 53.61. 5시드 기준선 평균 54.39 대비 미달 → P52 는 DELIVER 에서 이득 없음. 원문 = cards/2026-W37-verdicts.md ③ | - |
| DRN-260926-151 | reject | P52 RxDINO DELIVER 본 런 시드20260902(yeon): legal test 53.07 / val 66.15 / 24클래스 53.16. 2시드 평균 53.47 로 기준선 54.39 대비 −0.92 → P52 DELIVER 기각. 원문 = cards/2026-W37-verdicts.md ③ | - |
| DRN-260926-152 | reject | P52 RxDINO MCubeS 본 런 시드20260901(yeon, 200에폭 완주): val-best 58.18@174(final 57.96), C3-off 3시드 정본 58.07±0.49 대역 안 = 동률. 이득 없음 → MCubeS 헤드라인 교체 없음 | - |
| DRN-260926-154 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): N7 VICReg-off 격리(P46 C3-only λ0.1, 시드821 짝, yeon 200에폭): val-best 66.56@ep32(정본 본 런 67.79@70 대비 −1.23 트레이너 val). test-best 56.31 은 인용 금지, val-best 시점 legal test 미측정. VICReg 의 기여는 val 축에서만 확인됨 | - |
| DRN-260926-155 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 일일 카드 B0 기준선(P46 C3-only λ0.1 레시피, 40에폭 스크린 시드821, bengio): legal test 53.78 / val 64.97(체크포인트 epoch40_65.4_top1). 카드 통과선 = test ≥ 54.78. 원문 = registry 행, cards/2026-W37-verdicts.md | - |
| DRN-260926-156 | adopt | 일일 카드 E1(중간층 4탭 읽기, MODEL.TAPS per_modal [6,12,18,24]) 40에폭 스크린 시드821(bengio): legal test 54.85(B0 +1.07) 로 첫 통과 카드. 이후 확정 3쌍 채택(DRN-260926-01). 체크포인트 epoch35_67.25_top1(md5 e69d7deb) | - |
| DRN-260926-157 | reject | E0 특징 정보 프로브(원 4탭 vs 어댑터 후 vs 융합, 선형 프로브, 시드821 ckpt epoch90): test 27.8 / 35.6 / 45.1 로 게이트(i) 미달 → '어댑터가 정보를 버린다' 가설 기각. DRN-260926-25 와 같은 항목. 원문 = analysis/2026-09-08-daily-cards-E0-feature-probe.md | - |
| DRN-260926-158 | reject | E9 검증셋 사전 로짓 보정 τ∈{0, 0.5, 1.0}(학습 0): test 53.57 / 53.45 / 51.89 로 τ 가 커질수록 단조 하락, 게이트 +0.5 미달 → 폐기. τ=0 은 하네스 재동결용 대표 재채점으로 활용(636e490) | - |
| DRN-260926-159 | inconclusive | 일일 카드 E2(전 선형층 LoRA, qkv_full 등으로 확장) 40에폭 스크린 시드821(hpca100): legal val 68.38 / test 54.50, B0 대비 val +3.41 / test +0.72(회색지대 +0.5~+1.0). 이득이 RailTrack·TrafficLight 두 클래스에 몰리고 Wall·Water·Static 이 되받음. 시드 판별 대신 E12(E1+E2 결합) 가 test 54.39 로 가산성 기각되어 E2 축 종결(§5-9). 단독 판정은 회색지대로 미확정 | - |
| DRN-260926-160 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 일일 카드 E7 = MUSES PhysAug-off 기준선(P39.1-rank 시드2, 40에폭 스크린, hpca100): 학습기 val 80.29@ep40, 공식 native val 80.08. MUSES 40에폭 카드의 짝 기준선 | - |
| DRN-260926-161 | reject | 일일 카드 E7c = PhysAug-on 대조군(E7 과 PHYSAUG 만 다름): 학습기 val 80.18@ep40, 공식 native val 79.94 로 E7(80.08) 대비 −0.13, 작은 객체 클래스에서 손해 → PhysAug-on 폐기. 이후 MUSES 공정선은 PhysAug-off 로 통일 | - |
| DRN-260926-162 | adopt | 일일 카드 E3(센서별 클래스 prototype 손실, P46.C3_PROTO.SRC permodal) 40에폭 스크린 시드821(bengio): legal test 55.18(B0 +1.40, 넷 중 최고, val 보다 test 에서 더 버는 유일한 카드) → 스크린 통과. 후속 E13(E1+E3 결합) 확정 3쌍 채택(DRN-260926-01)으로 이어짐 | - |
| DRN-260926-163 | reject | 일일 카드 E4(혼동 쌍 margin, auto_val_k5·m 0.5·λ 0.05) 40에폭 스크린 시드821(bengio): legal val 65.86 / test 53.75(B0 대비 val +0.89 / test −0.03). RailTrack 만 오르고 24클래스로 넘어오지 않음 → E4 계열 종결(2026-09-11, §5-12). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-164 | reject | 일일 카드 E4b(혼동 쌍 명시: RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain) 40에폭 스크린 시드821(bengio): 35에폭 기준 legal test 55.14 로 한때 통과였으나 이득이 RailTrack 에만 있고 24클래스 Δ 는 게이트 미달, MARGIN 0.5 는 후기 Adam 불안정으로 죽음 → E4 계열 종결(§5-12). 후속 E4c 도 기각(DRN-260926-65) | - |
| DRN-260926-165 | reject | 일일 카드 E12 = E1(4탭) + E2(전 선형층 LoRA r32) 결합 40에폭 스크린 시드821(hpca100): legal test 54.39 로 E1 단독(54.85)보다 낮아 가산 가설 기각, E2 축 종결(§5-9). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-166 | adopt | E13(4탭 + 센서별 클래스 prototype) 확정 200에폭 시드1(jarvis): val-best epoch100 67.84(이후 100에폭 무갱신). 확정 3쌍 채택(DRN-260926-01)의 첫 시드, legal v2 재채점 완료. 체크포인트 NAS 보존(E13_s1_epoch100, md5 1517c12a) | - |
| DRN-260926-167 | adopt | E13 확정 200에폭 시드2(jarvis): val-best epoch160 68.0. 확정 3쌍 채택(DRN-260926-01)의 둘째 시드 | - |
| DRN-260926-168 | adopt | E13 확정 200에폭 시드3(jarvis): val-best epoch190 68.45. 확정 3쌍 채택(DRN-260926-01)의 셋째 시드, 판정 대장 22행 | experiments/judgment-ledger.md 행 22 |
| DRN-260926-169 | adopt | E1(중간층 4탭 읽기) 확정 200에폭 시드1(jarvis): val-best epoch140(ep138 부터 급등 68.83→68.90). DELIVER 헤드라인 3시드(test 56.24±0.42 / val 69.51±0.15)의 첫 시드이자 P56-A 의 동결 교사(E1_s1_epoch140, md5 c648f925). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-170 | adopt | E1 확정 200에폭 시드2(jarvis, 09-15 완주): legal v2 재채점 완료, 헤드라인 3시드의 둘째(test 최고 시드 56.73). 체크포인트 NAS 보존(E1_s2_epoch134, md5 79a3ff39) | - |
| DRN-260926-171 | adopt | E1 확정 200에폭 시드3(jarvis, 09-16 완주): legal v2 재채점 완료, 헤드라인 3시드의 셋째(val 최고 시드 69.68). 체크포인트 NAS 보존(E1_s3_epoch70, md5 d8e3e8ba) | - |
| DRN-260926-172 | adopt | E13 시드2 판별 40에폭 스크린(hpca100): 완주 + legal 재채점 완료, E13 통과(§5-13 '24클래스 안정성 네 번째 확증') → 확정 200에폭 3시드 기동의 근거 | - |
| DRN-260926-173 | adopt | E13 시드3 판별 40에폭 스크린(hpca100, 09-12 완주): 세 시드 평균 +0.91(최상값 +1.29 는 대표값에서 제외) → E13 확정 런의 근거(§5-16). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-174 | reject | E14(탭 깊이 어블레이션: E13 레시피 + TAPS.LAYERS [6,12,18,24]→[4,8,12,24]) 40에폭 스크린 시드821(hpca100): legal test 54.95 / val 66.43 으로 미달 → 얕은 탭 방향을 닫음(§5-18). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-175 | reject | E15(탭 투영 어블레이션: TAPS.MODE per_modal→mean, 투영 16개→4개) 40에폭 스크린 시드821(hpca100): 미달(§5-19). 단 mean 은 '투영 수'와 '센서별 유지' 두 요인을 묶어 해석에 한계(§5-15). per_modal 유지 확정 | - |
| DRN-260926-176 | adopt | E1M = E1(4탭)의 MUSES 이식(E7 PhysAug-off 기준선 대비 TAPS 만 on) 40에폭 스크린 시드3407(hpca100): E7 대비 3짝 +0.34 / +0.80 / +0.41, 평균 +0.52 로 스크린 통과(09-15). 다만 200에폭 풀 런에서는 이득 소실(DRN-260926-02) → 40에폭 통과 기록으로만 채택 | - |
| DRN-260926-177 | adopt | E13M = E13 의 MUSES 이식(E1M + 센서별 prototype) 40에폭 스크린 시드3407(hpca100): 공식 재채점으로 게이트 ①② 통과(§5-17), 야간 악조건 손실(fog/night −1.14, rain/night −1.84 vs E1M) 관찰. 200에폭 풀 런에서는 이득 소실(DRN-260926-02) → 40에폭 통과 기록으로만 채택 | - |
| DRN-260926-178 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13M 시드2 40에폭 스크린(hpca100): 공식 native val 81.132. 시드1 의 야간 손실이 재현돼(두 시드 같은 방향, 시드2 가 더 나쁨) 센서별 prototype 손실이 MUSES 야간을 일관되게 깎는다는 관찰 확정 | - |
| DRN-260926-179 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1M 시드2 40에폭 스크린(hpca100, 09-13 기동): E13M 의 야간 손실이 prototype 탓인지 가르는 대조군. E1M 3짝 평균 +0.52 판정(DRN-260926-176)에 사용됨. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-180 | reject | E13Mc = E13 의 MCubeS 이식 시드3407(hpca100, 200에폭 완주): final 57.82(짝 C3-off 대비 −0.11), val-best 58.73 은 val=test 라 판정에 미사용. 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-181 | reject | E13Mc(E13 의 MCubeS 이식) 시드20260827(hpca100, 200에폭): final 56.55(짝 C3-off 대비 −1.12). 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-182 | reject | E13Mc 시드20260828(hpca100, 200에폭): final 57.99(짝 C3-off 대비 −0.63). 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-183 | reject | E1Mc(E1 4탭의 MCubeS 이식) 시드3407(hpca100, 200에폭): final 58.68, 4탭 단독 3짝 평균 +0.10 으로 게이트(+0.5) 미달 → N-MC 종결(DRN-260926-10). MCubeS 헤드라인은 C3-off 58.07±0.49 유지 | - |
| DRN-260926-186 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B(완전 공유 LoRA r16, 1.57M) 시드821(jarvis, 200에폭 완주). clean·결측은 A 와 같고 depth 절반 열화에서 손해가 절반 이하(−6.82 vs A −15.77) → DRN-260926-03 재판정의 근거. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-187 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C(공유 r8 + 센서별 잔차 r8, 3.93M) 시드821(yeon, 200에폭 완주 + legal 재채점): depth 열화 −11.94 로 A 와 B 사이, 둘의 장점을 합치지 못함 → DRN-260926-03 의 근거 | - |
| DRN-260926-188 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 확정 런 매칭 분모 P46 C3-only 200에폭 시드20260902(jarvis, OOM 사망 후 재기동 완주, 학습기 val 67.20@ep140). E1·E13 확정 3쌍 판정(DRN-260926-01)의 시드2 짝. 판정 대장 25행 | experiments/judgment-ledger.md 행 25 |
| DRN-260926-189 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 확정 런 매칭 분모 P46 C3-only 200에폭 시드20260903(jarvis, 09-17 완주). E1·E13 확정 3쌍 판정의 시드3 짝. 판정 대장 25행 | experiments/judgment-ledger.md 행 25 |
| DRN-260926-190 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7(MUSES PhysAug-off 기준선) 시드20260902 40에폭 스크린(hpca100, 공식 재채점 lecun 09-15). E1M·E13M 3짝 판정의 시드2 짝 | - |
| DRN-260926-191 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7 시드20260903 40에폭 스크린(hpca100, 공식 재채점 09-15). E1M·E13M 3짝 판정의 시드3 짝 | - |
| DRN-260926-192 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1M 시드20260903 40에폭 스크린(hpca100): ep35 평가 직후 디스크 가득(Errno 28)으로 사망, 판정은 ep35 val-best 80.86 으로(생각정리 09-15, E4b 전례). 3짝 평균 +0.52 판정(DRN-260926-176)에 포함 | - |
| DRN-260926-193 | reject | E1M(E1 4탭의 MUSES 이식) 200에폭 풀 런 시드3407(hpca100, Errno 28 후 ep38 부터 이어 완주): 공식 val 81.70 으로 E7 풀 런(81.88)을 넘지 못함 → 40에폭 이득이 풀 런에서 소실(DRN-260926-02). test 제출 후보에서 제외 | - |
| DRN-260926-194 | reject | E13M(E13 의 MUSES 이식) 200에폭 풀 런 시드3407(hpca100): 공식 val 81.95 로 5런 중 최고이나 시드2 81.68 과 평균 81.81 ≈ E1M 81.82, E7 기준선(81.88)이 그 사이 → 이득 없음(DRN-260926-02). 제출본 교체 근거 없음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-195 | reject | E13M 200에폭 풀 런 시드20260902(hpca100): 공식 val 81.68. 2시드 평균 81.81 로 기준선과 동급 → 이득 없음(DRN-260926-02) | - |
| DRN-260926-196 | reject | E1M 200에폭 풀 런 시드20260902(hpca100): 공식 val 81.93. 2시드 평균 81.82 로 기준선(81.88)과 동급 → 이득 없음(DRN-260926-02) | - |
| DRN-260926-197 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7 기준선 200에폭 풀 런 시드3407(jarvis, 09-18 완주): 공식 val 81.88. E1M·E13M 풀 런과 같은 길이의 짝이며 MUSES 풀 런 판정(DRN-260926-02)의 분모 | - |
| DRN-260926-198 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13M 시드20260903 40에폭 스크린(hpca100, 09-16 완주·공식 재채점). E13M 3짝 판정의 셋째 짝. 40에폭 통과는 풀 런에서 소실(DRN-260926-02) | - |
| DRN-260926-199 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드3407 = MCubeS 같은 코드 기준선(TAPS off·C3 off, hpca100 200에폭). E1Mc·E13Mc 재판정(N-MC)의 분모 | - |
| DRN-260926-200 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드20260827(hpca100 200에폭). N-MC 판정의 분모 | - |
| DRN-260926-201 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드20260828(hpca100 200에폭). N-MC 판정의 분모 | - |
| DRN-260926-202 | reject | E1Mc 시드20260827(hpca100, yeon 무효 시드의 재실행, 200에폭 완주): 4탭 단독 3짝 평균 +0.10 으로 게이트 미달 → N-MC 종결(DRN-260926-10) | - |
| DRN-260926-203 | reject | E1Mc 시드20260828(hpca100, 200에폭 완주): 4탭 단독 3짝 평균 +0.10 으로 게이트(+0.5) 미달 → N-MC 종결(DRN-260926-10) | - |
| DRN-260926-206 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A(센서별 r16) 시드902(bengio, 200에폭). 3갈래×3시드 9칸 판정(DRN-260926-03)의 A 팔 시드2. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-207 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B(완전 공유 r16) 시드902(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 B 팔 시드2: depth 열화 손해 −6.82±0.57 의 자료 | - |
| DRN-260926-208 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A 시드903(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 A 팔 시드3: depth 열화 손해 −15.77±2.73 의 자료 | - |
| DRN-260926-209 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B 시드903(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 B 팔 시드3 | - |
| DRN-260926-212 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1 확정 시드4 v2(yeon 새 체크아웃, 20260904): 완주·재채점 완료. DELIVER 넷째 짝이며 헤드라인은 3시드 유지(4시드 평균은 감시 세션 std 회신 후 갱신 — 판정 대장 39행). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-213 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): C3-only 200에폭 시드4(yeon 새 체크아웃, 20260904): E1·E13 시드4 짝의 같은 코드 분모. 완주·재채점 완료 | - |
| DRN-260926-214 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13 확정 시드4(yeon, 20260904): legal test 55.67 / val 69.40(ep110) → E13 4시드 test {56.95, 55.21, 57.29, 55.67}. 헤드라인은 3시드(56.48) 유지, 4시드 평균은 std 회신 후 갱신. 판정 대장 39행 | experiments/judgment-ledger.md 행 39 |
| DRN-260926-215 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C(공유 r8 + 잔차 r8) 시드902(yeon, 옛 코드 버전 각주). 9칸 판정(DRN-260926-03)의 C 팔 시드2: depth 열화 −11.94±2.79 의 자료. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-216 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C 시드903(jarvis, 09-18 완주). 9칸 판정(DRN-260926-03)의 C 팔 시드3 | - |
| DRN-260926-217 | reject | E2(전 선형층 LoRA) 시드2 짝 40에폭 스크린(hpca100, B0s2 짝): 회색지대 판별용이었으나 E12(E1+E2 결합) 가산성 기각으로 E2 축이 09-11 종결(§5-9) → 단독 채택 없음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-220 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E3 legal test 재채점(ckpt epoch25_65.97_top1, bengio): test 55.18(B0 +1.40, 카드 최고), val 은 bengio 양도로 미측정. E3 스크린 통과(DRN-260926-162)의 원자료 | - |
| DRN-260926-221 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1 legal 재채점(ckpt epoch35_67.25_top1, bengio): val 66.90 / test 54.85(B0 +1.07), 전이율 55%. E1 스크린 통과(DRN-260926-156)의 원자료 | - |
| DRN-260926-222 | reject | E4 legal 재채점(ckpt epoch15_65.92_top1, bengio): val 65.86 / test 53.75(B0 −0.03) → E4 폐기(DRN-260926-163)의 원자료 | - |
| DRN-260926-226 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0 legal 재채점(ckpt epoch40_65.4_top1, bengio): val 64.97 / test 53.78 → 40에폭 카드 통과선 test ≥ 54.78 확정. DRN-260926-155 의 원자료 | - |
| DRN-260926-228 | reject | E17(4탭 + 고해상도 세부 가지) 40에폭 스크린 시드821 hpca100 사본(09-16 기동): jarvis 사본과 같은 카드. Δtest +0.88 로 게이트 미달, 확정 런 취소(DRN-260926-04). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-231 | adopt | Q2(두 패스 = clean CE + 열화 CE + 동결 E1 교사 증류) 를 이후 카드의 바탕 레시피로 채택: clean 축은 3시드 24클래스 평균 Δ +0.41 로 사전 기준(+0.6) 미달이라 clean 개선 주장은 하지 않음(손해도 없음). 강건 축은 3시드 재현: 모달 결측 15조합 평균 48.06 vs E1 스크린 40.40(+7.7), depth 부분 결측 저하 비율 0.26, S&P 노이즈 d.2 저하 약 3.8 감소. RGB 부분 결측에는 이득 없음. 헤드라인 교체 없음. 판정 대장 2026-09-28·09-29 행 | experiments/judgment-ledger.md 행 71 |
| DRN-260926-232 | reject | Q3(품질 헤드 QAF 본 카드) 종료: 시드821 legal v2 에서 Q2(품질 헤드 없는 두 패스+교사 증류) 대비 24클래스 −1.31·얇은 객체 4클래스 −5.67, 강건 축도 모달 결측 15조합 평균 47.09 vs Q2 48.68 로 Q2 이하. 사전 규칙(강건에서 Q2 를 못 넘으면 종료) 적용. 판정 대장 2026-09-24·09-29 행 | experiments/judgment-ledger.md 행 51,53,59,73 |
| DRN-260926-235 | adopt | det(poongsan, 국책 목표 mAP50 0.85) P29-Det egofill 데이터(train 11,799, bengio 50에폭): best mAP50 0.8501@ep9 로 목표 달성(2026-07-05, 동일 스택에서 데이터만으로 0.4455 → 0.850). det 트랙 목표 달성 기록으로 채택 | - |
| DRN-260926-236 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P29-Det 모달 ablation(img/event/thermal, egofill 11,799, bengio) 완주. 결론은 registry 모달 ablation 행과 메모리 det-final-ann-modality-ablation(RGB 단독 ≥ 3모달 on mAP50)에 기록 | - |
| DRN-260926-237 | inconclusive | P29-Det 최종 annotation 풀 런(det_P29_final_full): 2026-07-08 이후 상태 미갱신·결과 기록 없음. det 목표는 egofill 런(DRN-260926-235)과 D1 계열로 달성됐으므로 재실행 없이 미확정 종결 | - |
| DRN-260926-238 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): P29-Det poongsan_v2(clean label) 재학습(jarvis): 라벨 정정 후 스택 기준선. 원문 = det/p29det-data-fix.md | - |
| DRN-260926-239 | inconclusive | P31.1-Det(캘리브레이션 신뢰도 + decisive 라우터 + FCOS) poongsan v3clip 비공식 분할: 완주했으나 v2 분할과 직접 비교 불가(재평가 과제 D2 미수행). det 목표는 P29-Det egofill·D1 계열로 달성돼 재평가 없이 미확정 종결 | - |
| DRN-260926-240 | reject | P30-Det(라우터 + 쿼리 디코더) poongsan_v2: 소물체 붕괴로 mAP50 0.228 vs P29-Det 0.446 → dead. 비교 리포트 /mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/ | - |
| DRN-260926-241 | adopt | 측정 항목(채택 = 외부 기준점 기록으로 확정): YOLO11m RGB 단독(외부 헤드, hinton) poongsan label-v3 mAP50 0.864. 우리 det 계열의 외부 비교점 | - |
| DRN-260926-242 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): ProbeA2 백본 스케일링(동결 DINOv3 S+/B/L/H+/7B + 공용 경량 헤드, MUSES RGB 단독) 5점 완료. 축 분리 게이트 §⑧ 정식 판정 완료(백본 크기 축과 융합 축 분리 근거). 캐시 55.9G 는 2026-09-30 삭제(재생성 가능) | - |
| DRN-260926-293 | adopt | 기록 항목(채택 = 계보 기록으로 확정): MemorySAM 초기 기준선(SAM·LoRA-SAM·P4~P7·재구성). SAM2 계보(종료 트랙 MULTIAQUA)의 출발점. 상세 = models/arch-evolution.md '공통 기반' 절 | - |
| DRN-260926-257 | reject | P8(ConfidenceHeadV2 + Sigmoid UAMM, MULTIAQUA): M-score 77.5~78.5(증강별) 로 P9(81.47)에 대체됨. 종료 트랙 | - |
| DRN-260926-258 | adopt | P9(CrossModalFusionHead + Max-Norm UAMM, MULTIAQUA): M-score 82.10(재제출 #16710, val 93.29 / test 70.91) = 챌린지 공동 1위 제출본. SAM2 계보의 최선으로 채택(종료 트랙, DRN-260926-87 과 같은 런) | - |
| DRN-260926-261 | reject | P12(입력 조건부 Soft MoE LoRA, MULTIAQUA): M-score 80.80(P9 대비 −0.67) → 기각. 동적 융합 계열 실패 패턴의 시작 | - |
| DRN-260926-262 | reject | P13(Energy Score 융합 + expert collapse 수정, MULTIAQUA): M-score 81.21(P9 대비 −0.26) → 기각 | - |
| DRN-260926-263 | reject | P14(모달별 독립 보조 디코더, MULTIAQUA): M-score 74.27(P9 대비 −7.20) → 기각. 원인 = 동결 백본의 보조 마스크 품질 부족(ISSUE-008) | - |
| DRN-260926-265 | reject | P16(캘리브레이션 공간 엔트로피 융합, P15 설계 구현판, MULTIAQUA): M-score 68.42(P9 대비 −13.05) → 기각. 픽셀 단위 융합이 보조 마스크 오차를 증폭 | - |
| DRN-260926-266 | reject | P17(다중 스케일 FPN 보조 디코더 + 공간 엔트로피 융합, MULTIAQUA): M-score 73.23(P9 대비 −8.24) → 기각. 'P14~P17 동적 융합의 실패' 종합 분석(log.md)으로 축 종결 | - |
| DRN-260926-267 | inconclusive | P18(학습 가능한 ResNet-18 보조 백본 + 구성 가능한 융합, MULTIAQUA): 설계 절만 있고 log·registry 에 완주 수치가 없다. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-268 | reject | P19(학습된 공간 교차 모달 융합 헤드, MULTIAQUA): hardaug5 ep36 M-score 69.63 로 P9 대비 크게 미달 → 기각 | - |
| DRN-260926-269 | inconclusive | P20(공유 MLP 게이트 + 고랭크 MoE, 실험 J-A, MULTIAQUA): log 에 '학습 대기'로만 남아 완주 수치 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-270 | reject | P21(DeBA-FP 변형 가능 병목 어댑터, 실험 K, MULTIAQUA): M-score 81.77(#16792) 로 P9·P22(82.10) 미달 → 최종 제출본 아님(DRN-260926-89) | - |
| DRN-260926-271 | adopt | P22(다중 스케일 DeBA-FP, 실험 L, MULTIAQUA): M-score 82.10(#16932, val 93.42 / test 70.77) = P9 와 공동 1위 제출본으로 채택(종료 트랙, DRN-260926-88) | - |
| DRN-260926-273 | inconclusive | P24(P9 + 품질 인식 메모리 게이팅, 모달별 디코더 증류, 실험 N, MULTIAQUA): ep36 시점 sigmoid 버그(ISSUE-013)로 교사 신호가 잘못돼 미제출, 수정 후 완주 기록 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-276 | inconclusive | P27(교차 모달 메모리 attention 의 가산 attention bias, RBMA 전구체, 2026-04-14): 기구로는 성립했으나 성능 판정 불가(arch-evolution ④ 판정). 이후 P28 RBMA 로 승계 | - |
| DRN-260926-256 | inconclusive | P28 RBMA(신뢰도 편향 메모리 attention, DELIVER B200): val 57.87 / test 50.61@ep12 에서 ep16 사망(조기 종료) → 불완전 실패. 신호 정의는 계보 표준이 됐으나 성능 판정에 필요한 완주가 없음. 이후 P29·P31·P32 에서 attention bias 효과 ≈0 으로 반증됨 | - |
| DRN-260926-255 | reject | P29 SDC(자기 유도 조건 라우팅, DELIVER B200): val 63.20@ep100 / test 54.34@ep146 로 목표(66.51 / 56.71) 대비 −3.31 / −2.37 → 정체(미달). 비교 기준선으로만 남음(DRN-260926-95) | - |
| DRN-260926-291 | adopt | P29-Det(RBMA 백본 + FPN/FCOS 검출, poongsan): egofill 데이터로 mAP50 0.8501@ep9 달성(2026-07-04, 국책 목표 0.85). det 트랙의 목표 달성 계열로 채택(DRN-260926-235) | - |
| DRN-260926-253 | reject | P30-Det(P30 백본의 검출 확장: 신뢰도 라우터 융합 + 객체 쿼리 디코더 + FCOS 보조): 소물체 붕괴로 mAP50 0.228 vs P29-Det 0.446 → 실패(DRN-260926-240 과 같은 런) | - |
| DRN-260926-254 | reject | P30(클래스 토큰 디코더 + 신뢰도 앵커 학습 라우터, DELIVER): val 49.76 / test 44.10 으로 P29 대비 −13.4 / −10.2 → 명백한 실패(원인 귀속은 뒤집힘: 디코더 아닌 라우터 문제). arch-evolution ④ | - |
| DRN-260926-252 | inconclusive | P31(캘리브레이션 이중 신뢰도 RBMA + 다중 스케일 클래스 토큰 디코딩, DELIVER B200): 성능 정체, 기제 2건은 성공(arch-evolution ④). 수치 이득 없이 기제 자료만 남김. SAM2 계보 종료로 승계되지 않음 | - |
| DRN-260926-251 | reject | P32 CoRB(입증 편향 메모리 attention, DELIVER): val 64.12(계보 최고) / test 55.00(P28 −0.27), attention bias 는 유의한 순손해(p=4.5e-22) → 실패이자 계보에서 가장 결정적인 반증(attention bias 축 재시도 금지) | - |
| DRN-260926-277 | inconclusive | P33 CG-MoD(역량 게이트 hard 융합 + 모달 드롭아웃): P33.1 조기 중단(이득 0)·P33.2 ep52 무효과 뒤 P34 로 피벗, 완주 수치·판정 기록 없음 → 미완결(기록 소실). P33.3 은 구현 없는 placeholder | - |
| DRN-260926-278 | adopt | P34 ReliaDINO(동결 DINOv3-L + 센서별 LoRA, 계보 전환점, 2026-07-13): legal val 68.19 / test 56.62 로 계보 최대의 성공. 단 성공 원천은 백본 + LoRA 이지 제안 모듈(RBMA attention bias ≈0)이 아님. 이후 모든 세대의 바탕으로 채택 | - |
| DRN-260926-279 | adopt | P35 공정 레시피 동결(P34 − ATTN_BIAS − CONSISTENCY − PhysAug): 의도대로 공정선 확립, 성능은 −1.12. PhysAug 를 켠 수치는 헤드라인·게이트에 쓰지 않는다는 규칙(user 2026-07-20)의 근거로 채택 | - |
| DRN-260926-280 | adopt | P36 클래스별 신뢰도 앵커 라우터(= P35 + router): 유일하게 살아남은 모듈이나 증분은 잡음 대역(test +0.10). 현행 레시피의 ROUTER 로 유지(채택)하되 '새 모듈 + SOTA' 논문은 성립하지 않는다는 판정 기록 | - |
| DRN-260926-281 | reject | P37a CEFR-Head(클래스 기대 특징 라우팅)·P37b ClassToken-lite: 둘 다 실패(방식은 다름 — CEFR 은 라우팅 분화 실패 0/19, ClassToken 은 성능 미달). CEFR 라우팅은 재시도 금지 축 | - |
| DRN-260926-247 | adopt | P38 MaskQueryLite(Mask2Former-lite 쿼리 헤드): DELIVER 에서 모듈 자체는 실패(D1 평균 53.66, 추론 기여 no-op)이나 부산물 두 개 성공 — MUSES 공식 test 79.025(7월 기록)와 쿼리 헤드 구조가 현행 레시피(M2F)에 남음. 부분 채택 | - |
| DRN-260926-248 | reject | P39 DPC(이중 경로 경쟁): 기제는 성공, 성능 전환은 실패. 부수 결과: gate/calibration 이 얇은 클래스에 유해(off 시 +35.9/+26.0)로 재판정 → GATE·CALIBRATION 영구 off | - |
| DRN-260926-249 | adopt | P39.1 Rank 수리(gated_mlp trunk + VICReg): 기제 실증 성공 + MUSES 내부 최고(공식 test 79.788) → 현행 기준선으로 채택. 경고: 시드 분산 0.92 가 대부분의 델타보다 커 0.1~0.3 차이 판정은 재검토 대상(시드 평균 병기 규칙의 출발) | - |
| DRN-260926-250 | inconclusive | P40 RCA-Fusion(신뢰도 조건부 감쇠): 미실행(보류). P39 분석에서 gate/calib 이 유해로 재판정돼 착수 근거가 사라짐. 판정 불가로 종결 | - |
| DRN-260926-282 | reject | P41 FCR(융합 스펙트럼 붕괴 / 융합 클래스 정렬 규제): 결정적 기각(airtight falsification), 계보에서 가장 깨끗한 음성 결과 | - |
| DRN-260926-283 | reject | P42 lidar 강제(조건부 균형 img 마스킹): 전 FRAC 에서 미달, 단조 열화 → 실패 | - |
| DRN-260926-245 | reject | P43 PanopticDual(독립 주손실 mask-classification 헤드, MUSES): val 82.51 → 공식 test 79.351 로 P39.1(79.788) 미달, PQ 는 정답 부재로 판정 불가 → 승계되지 않음(DRN-260926-52·111) | - |
| DRN-260926-246 | reject | P44 BMR(균형 멀티모달 신뢰도) + P45 FogStyle: BMR 은 완전 실패 + 가설의 test 반증, FogStyle 은 미실행. 세대 종결 | - |
| DRN-260926-244 | adopt | P46 CTR(클래스 전이 복구: C1 희소 클래스 재샘플 + C2 masked consistency + C3 클래스 prototype): C1 은 순유해, C2 는 유해(Δ −1.67), C3 만 유효해 C3-only(λ0.1)가 단일 런 최고 test 56.99 → C3 prototype 손실을 현행 레시피의 일부로 채택(부분 채택). SOTA·재현성 검증은 미달 | - |
| DRN-260926-243 | adopt | P47-2 UniBal(모달별 독립 보조 헤드 + 단일 모달 CE, MUSES 4모달): val-best 82.06, 하네스 가드 재평가 81.72 → P52 게이트 G2 기준(81.42)으로 채택. 공식 test 미제출. 기법 단위 노벨티는 아님(선행 UMT·OGM-GE) | - |
| DRN-260926-285 | reject | P47-1 LiDAR 투영 밀도화(구 D-1, MUSES 4모달): 공식 test 78.790 으로 base 79.571 대비 −0.78, val 만 오르고 test 역행 → 음성 결과 확정(효과 없음) | - |
| DRN-260926-286 | reject | P48 쿼리 경로 인스턴스 감독(제안 단계, MUSES 4모달): 게이트 미달로 폐기(2026-08-06, DRN-260926-144) | - |
| DRN-260926-78 | reject | P49 AIR 비대칭 주입(제안 2026-08-10): DELIVER 에서 ep30 게이트 위반(P49) 및 fair-eval 로 '57.68 SOTA' 철회(P49.1), MUSES 팔은 완주 기록 없음 → 계열 종결(2026-08-16, 양 벤치 패배, 가설 대장 H14) | - |
| DRN-260926-79 | adopt | P50 MAP 모달 정렬 사전학습(제안 2026-08-17): Phase1(프로브, 200k) init 파인튠 게이트 통과(+0.74, H22 확인) → P52 init 으로 채택. Phase2 확장(EXT)은 기각(DRN-260926-32). 단 P52 본 런이 DELIVER 에서 기준선 미달이라 최종 레시피에는 남지 않음 | - |
| DRN-260926-80 | reject | P51 CMLC 교차 모달 LoRA 결합(제안 2026-08-21): 페어1 에서 가설 대장 H19 반증으로 2026-08-26 종결 | - |
| DRN-260926-81 | inconclusive | condexpert 어댑터 프로브 제안(2026-08-08): registry·monitor-log 에 실행 기록이 없어 실행 여부와 결과를 확인하지 못함. 판정 불가로 종결. 조건 전문가 어댑터 아이디어는 P56-C(환경 조건 LoRA 전문가 혼합)로 이어짐 | - |
| DRN-260926-82 | inconclusive | H10 재판정 실험 요청(2026-08-08, 런 이름 jarvis_muses_h10_instsup_mini): 실행 기록·결과가 없어 판정 불가로 종결. 인스턴스 감독 축은 P48·R2 에서 실패로 확정됨 | - |
| DRN-260926-83 | adopt | 공간 모달 오라클 프로브 제안(2026-08-18, H16): 2026-09-28 tools/oracle_headroom_probe.py 로 실행됨 — 창마다 최선 모달 부분집합을 고르는 오라클의 순 여유는 +1.0~2.2 mIoU(조건별 1.1~1.8), 64px 창의 약 70% 는 전 모달이 최선. 이 결과를 채택해 곱셈 게이트를 주 후보에서 제외(판정 대장 2026-09-28·29 행) | - |
| DRN-260926-84 | adopt | P36 노벨티 비판 검토(2026-07-16): 'attention bias 계열은 4세대·2백본에서 결정을 바꾸지 못했다'는 체계적 반증을 논문 ablation 재료로 삼고, '새 모듈 제안 + SOTA' 노선을 접는다는 결론을 채택. 이후 학습 설계 노선(Q2·P56)의 출발점 | - |
| DRN-260926-294 | reject | 실험 I P9 Gamma TTA(MULTIAQUA, softmax 확률 평균): 실패 — 기준선 대비 이득 없음. 종료 트랙 | - |
| DRN-260926-295 | reject | 실험 II·III I2I 변환(test 야간 RGB 를 img2img-turbo 로 주간화 뒤 추론, MULTIAQUA): 실패. 종료 트랙 | - |
| DRN-260926-296 | reject | 실험 IV P9 CV 휴리스틱 보정(worst 56장 밝기·대비 보정 뒤 재추론, MULTIAQUA): test mIoU −4.76, M-score −2.39(79.08) → 실패(잘 맞던 이미지까지 보정됨). 종료 트랙 | - |
| DRN-260926-298 | adopt | 실험 VI P9 PhysAug(물리 기반 증강, MULTIAQUA): hardaug8+physaug 가 P9 최선 M-score 82.10 의 레시피가 됨 → 종료 트랙에서 채택. 단 논문 트랙(DELIVER·MUSES)에서는 PhysAug 를 헤드라인에 쓰지 않는다(P35 규칙) | - |
| DRN-260926-299 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P9 증강 ablation 계열(hardaug2~8·night2·noCRM·tta, MULTIAQUA): hardaug8 이 최선(M 81.98, physaug 결합 시 82.10), Night Aug 는 포화. 종료 트랙 | - |
| DRN-260926-300 | inconclusive | Tiled Inference(원본 해상도 슬라이딩 윈도우 추론, MULTIAQUA, 2026-03-17): log 에 '대기 중 — 제출 후 비교 필요'로만 남아 결과 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-301 | adopt | 측정 항목(채택 = 진단 기록으로 확정): P25 공간 품질 게이팅의 야간 예측 품질(Pred Q) 분석(2026-06-14). SAM2 계보 품질 게이팅 실패 원인 자료 | - |
| DRN-260926-302 | adopt | 측정 항목(채택 = 기록으로 확정): P26 DELIVER AMP on + gradient checkpoint off 단일 GPU 메모리 프로브(2026-04-10). 메모리 예산 자료 | - |
| DRN-260926-303 | adopt | 측정 항목(채택 = 분석 기록으로 확정): 아키텍처 심층 분석 A·B(UAMM 실효성, P9 vs MMSamBase, 2026-03-24). 'SAM2 메모리 attention 이 이미 암묵적 교차 모달 적응을 한다'는 P9 우위 해석의 근거 | - |

## withdrawn
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-261020-11 | P2 | Ablation L3 · A2 − 4탭(→ 마지막 블록 탭 1개) | MODEL.TAPS.LAYERS: [24] (고해상 어댑터 탭 수 1 로 맞춤), 그 외 A2 동일 | - | ablation, leave-one-out, paper |
| DRN-261020-15 | P2 | Ablation L6 · A2 에서 교사 증류만 제거(KD_W 0, 열화 두 패스는 유지) | TRAIN.QAF.KD_W 0 (DEGRADE_P·MISSING_FRAC 은 A2 그대로) | - | ablation, leave-one-out, paper |
| DRN-261017-01 | P3 | A2 이식 · MUSES — E1M(4탭) screen40 + 고해상 경로, 시드 s2·s3 | MODEL.HIRES_ADAPTER on(그 외 E1M screen40 동일) | - | muses, A2, transfer |
| DRN-261028-02 | P3 | 고해상 신호 구동 토큰 적응 LoRA 구현: LORA_ROUTER.SIGNAL=hires_stats(P56-C 라우터 입력을 센서별 고해상 스템 통계로 교체) + 스모크 | encoder.py/p56c_router.py: SIGNAL hires_stats 분기(고해상 스템 센서별 출력 → 토큰 통계 3종 → stride-16 풀링 → MLP, bias 초기화 α≈0.5) · config 키 LORA_ROUTER.SIGNAL · 스모크(off 계약·grad·합성 블러 패치에서 α 저하) | - | routed-lora, hires-signal, build |
| DRN-261028-03 | P3 | 고해상 신호 구동 토큰 적응 LoRA · DELIVER 40ep 1시드 s821 — 기준 모델(A2-shared 또는 A2) + 공유↔센서별 토큰 혼합(신호=고해상 통계), Q2 학습 불변 | MODEL.LORA_MODE state_routed + LORA_ROUTER.SIGNAL hires_stats (그 외 기준 모델 config 동일; 기준 = L1 강건성 통과 시 A2-shared, 아니면 A2) | - | routed-lora, hires-signal, screen40 |
| DRN-260926-39 | P39 | #1 P40 RCA-Fusion | #1 P40 RCA-Fusion | - |  |
| DRN-260926-40 | P40 | #2 P39 radar-fix 재실험 | #2 P39 radar-fix 재실험 | - |  |
| DRN-260926-41 | P41 | #5 P47-MUB D-1 | #5 P47-MUB D-1 | - |  |
| DRN-260926-42 | P42 | #10 P49-AIR | #10 P49-AIR | - |  |
| DRN-260926-43 | P43 | A100 ③ P49 @1024 대조 | A100 ③ P49 @1024 대조 | - |  |
| DRN-260926-44 | P44 | ProbeA2-7B 추가 측정 | ProbeA2-7B 추가 측정 | - |  |
| DRN-260926-97 | P233 | [registry] P28 RBMA seg (B200 RUN-1) | P28 (RBMA) | - | registry, deliver, seg |
| DRN-260926-114 | P250 | [registry] jarvis_deliver_rgbdel_P46_ctr | P46-CTR (RCS+MCC+PROTO, P39.1 base) | - | registry, deliver, seg |
| DRN-260926-117 | P253 | [registry] hpca100_deliver_rgbdel_P46_ctr_c2c3 | P46-CTR C2+C3 변형 (C1_RCS off, C2_MCC on, C3_PROTO on·cross_view=true — C2 순기여 격리, A100 40GB) | - | registry, deliver, seg |
| DRN-260926-122 | P258 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam015 | P46-CTR C3-only λ 스윕 보강(λ3=0.15, C1 off/C2 off/C3 on·cross_view off, jarvis λ0.05/0.1(완주)/0.2 사이 보간) | - | registry, deliver, seg |
| DRN-260926-184 | P320 | [registry] yeon_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | E1Mc 시드 20260827 — E1(4탭)의 MCubeS 이식 | - | registry, mcubes, seg |
| DRN-260926-185 | P321 | [registry] yeon_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828 | E1Mc 시드 20260828 — E1(4탭)의 MCubeS 이식 | - | registry, mcubes, seg |
| DRN-260926-204 | P340 | [registry] lecun_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260827 | B0Mc 시드 20260827 (lecun) — MCubeS 같은 코드 기준선, 0827 비교 쌍의 기준선 쪽 | - | registry, mcubes, seg |
| DRN-260926-205 | P341 | [registry] lecun_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | E1Mc 시드 20260827 (lecun, TAPS 적용) — 0827 비교 쌍의 E1 쪽 | - | registry, mcubes, seg |
| DRN-260926-210 | P346 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E1_confirm200 | E1 확정 시드4 🔴 무효 — TAPS 미적용 | - | registry, deliver, seg |
| DRN-260926-211 | P347 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260904 | C3-only 200ep 시드4 (E1 시드4 매칭 분모) | - | registry, deliver, seg |
| DRN-260926-223 | P359 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260902_screen40_E1s2 | E1 시드2 페어(yeon판) — yeon GPU7 4시간 유휴 해소 | - | registry, deliver, seg |
| DRN-260926-227 | P363 | [registry] daily_cards_early_eval_bengio_gpu0 | E1 ep30·E3 ep25·E4 ep15 ckpt 조기 legal test 재채점(방향 확인, 판정 아님) | - | registry, deliver, 평가(학습 0) |
| DRN-260926-229 | P365 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_E17_confirm200 | E17 확정 런 시드1(200ep 목표) = E1(중간층 4탭 읽기: 블록 6/12/18/24 특징을 얕은 헤드로 보조 감독) + 고해상도 세부 가지(DETAIL_BRANCH, 얇은 객체 픽셀 디테일 복원용 보조 브랜치) — 위 40ep 스크린 통과 후 확정 착수 | - | registry, deliver, seg |
| DRN-260926-230 | P366 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E17_confirm200_s2 | E17 확정 런 시드2(200ep 목표), SEED 20260902 — 시드1과 동일 구성(E1 4탭 읽기 + 고해상도 세부 가지 DETAIL_BRANCH) | - | registry, deliver, seg |
| DRN-260926-259 | P610 | [세대] P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨) | P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨) | - | generation |
| DRN-260926-260 | P611 | [세대] P11: P10 + MI Routing Loss (취소됨) | P11: P10 + MI Routing Loss (취소됨) | - | generation |
| DRN-260926-85 | P670 | [대기 트리거] P37b-DELIVER 완주 시 → P34 per-class 비교 분석 | P37b vs P34 클래스별 Δ 표 | - | deliver, analysis |
| DRN-260926-86 | P671 | [대기 트리거] yeon P37b-det 완주 → P38-det 자동 기동 | P38-det(M2F query head를 detector로) | - | det |
| DRN-260926-297 | P703 | 실험 V · P9 FDA Augmentation | FDA 증강 | - | multiaqua |

## superseded
| id | 우선 | 제목 | 바꾼 변수 | runs | 태그 |
| --- | --- | --- | --- | --- | --- |
| DRN-260926-26 | P26 | Router-oracle-probe · 오라클 지도 라우터 프로브 (S0-feature-probe로 흡수) | 고정 특징 위 소형 라우터를 H16 오라클 정답 부분집합으로 지도학습 | - | deliver, router |
| DRN-260926-28 | P28 | S1-4tap · 4탭 읽기(블록 6/12/18/24)+SimpleFPN, 3페어 매칭 | 4탭 읽기(블록 6/12/18/24)+SimpleFPN | - | deliver |
