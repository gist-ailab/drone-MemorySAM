# 일요일(2026-11-08)까지 실험 계획 — Access 원고 동결(11-10) 직전 배치

작성: method(MMSAM | 생각정리) 2026-11-04 저녁. 집행 = manager(MMSAM | monitoring), 측정 = analysis, 원고 = specialist-paper.
정본은 `lab-plan`(상태·판정)이고 이 문서는 배치 순서와 근거를 한 곳에 모은 것이다. 바뀌면 이 문서를 덮어쓴다.

## 0. 11-04 저녁 기준 사실

| 항목 | 상태 | 출처 |
|---|---|---|
| MUSES 4모달 A2(DRN-261019-02) s2·s3 768 | **둘 다 완주**(학습기 val-best 80.56 / 80.18, yeon) | yeon `outputs/ReliaDINO/*muses*A2_s{2,3}_768` 직접 확인 |
| MCubeS C2D2(DRN-261017-02) hpca100 2시드 | **상태 불명**: hpca100 GPU 0/4 사용, 임대 종료 경고(10-30) | lab-status 11-04 09:21 |
| MCubeS E1Mc-B 교사(DRN-261102-01) | amphibious-demo 학습 중(torchrun 수정 후 재기동) | task log 11-04 |
| L8 router 제거 s821 / s902 | amphibious-demo 완주 여부 미확인 / jarvis ep30·40, ETA 11-04 밤(OOM 1회 후 ep25 재개) | lab-status |
| L2 단순 융합 s902 / L4 M2F 제거 s902 | bengio ep1 / yeon ep1 (manager 11-04 자동배치) | lab-status |
| L5(열화 패스 제거)·L6(교사 증류 제거) 2시드 | clean 측정 완료(yeon ckpt 있음), **EMM·RMM 미측정**, 판정 전 | tasks |
| L7(물리 열화 제거 → 결측 0채움만) s821·s902 | jarvis 학습 완료, **legal 평가 미실행** | DRN-261029-01 |
| 구조 의존성(DGFusion·CAFuser 두 패스+KD, DRN-261029-02·03) | **미착수**(specialist-baseline 미개방) | inbox |
| NM 벤치 정의 A2 3시드(DRN-261102-04) | hpca100 측정 중이었음 → hpca100 상태와 같이 확인 | task log |
| 비용표(DRN-261027-02) | 채택 완료 | verdict 11-04 |
| 공개 가중치 클래스별 IoU(CMNeXt·CAFuser·DGFusion) | 완료, paper 에 전달 | inbox 20261104-185601 |
| 벤치 교차 확인(DRN-261022-01) (a)(b) | 완료 → method 판정 adopt(§4) | analysis |
| 실패 키(DRN-261020-01) | 완료 → method 판정 부분 adopt(§4) | analysis |

user 결정(11-04, "mmsam meeting report" 세션): 쿼리 헤드는 논문에서 제외(Table 7 열·57.89 행 삭제) → **L4 ablation 은 논문에 불필요**. "C2D2 가 clean 을 올린다"는 주장은 하지 않음(3시드 E1 56.24 → Q2 56.59) → 두 패스 학습의 기여는 **강건성**으로 보인다.

## 1. 원고가 요구하는 것 → 실험 (우선순위 순)

| 우선 | 원고 칸 | 필요한 실험·측정 | 학습? | 서버 | 담당 | 목표일 |
|---|---|---|---|---|---|---|
| P1 | Table 2 MUSES Ours | s2·s3 **공식 val 1024/1536**(tools/eval_muses_official.py) → 해상도 선택 → 2시드 평균 | 0 | yeon GPU0·1 | analysis | 11-05 |
| P1 | MUSES test 제출본(DRN-261029-09) | 선택 해상도로 test 추론 → zip 포맷 검증, 제출은 user | 0 | yeon | analysis(zip)·manager(ckpt 백업 NAS+drone-demo md5) | 11-06 |
| P1 | Fig. MUSES adverse 짝 | s2 공식 val 250장 전체 예측 덤프(tools/muses_dump_preds.py) → paper 가 선별 | 0 | yeon | analysis | 11-06 |
| P1 | Table 3 MCubeS Ours | **hpca100 상태 확인** → 살아 있으면 완주 대기, 죽었으면 yeon 768 변형 2시드 즉시 재기동(ckpt 있으면 resume) | 200ep | hpca100 또는 yeon 4장 | manager | 확인 11-05 · 완주 11-09~10(위험) |
| P1 | Table 6b L5·L6·L7 행 | L7 s821·s902 legal v2 test(jarvis) · L5·L6·L7 **EMM test**(mm_eval_v2, 같은 프로토콜) | 0 | jarvis GPU3·7 / yeon GPU0·1 | analysis | 11-06 |
| P1 | Table 6c(3시드 강건성 ablation) | 이미 있음(Q2 vs A2 val) · NM 열은 DRN-261102-04 결과 확인 | 0 | hpca100 확인 | analysis | 11-05 |
| P1 | Table 7 L2 행 2시드 | bengio L2 s902 완주(11-05) → legal v2 | 40ep | bengio | manager→analysis | 11-06 |
| P1 | L8 router 행 | jarvis s902·amphibious s821 완주 → legal v2 + router 추론 끄기(DRN-261029-06) | 40ep | jarvis | analysis | 11-06 |
| P2 | Fig.(고해상 경로 특징 수준) DRN-261104-02 | 공급원 분리 패널 6장 + H-V1 집중도 + H-V2 **on/off 와 recombine 둘 다**(off 는 mIoU −10.6 로 파괴적, DRN-261020-01) | 0 | jarvis 1장 | analysis | 11-07 |
| P2 | Table 5 조건별 정확도 본문 | analysis 요청 진행 중(paper inbox) | 0 | — | analysis | 11-06 |
| P2 | §6 TL precision/recall 확인 todo | A2 3시드 TL P/R 재확인(기존 s821 값 0.80→0.85, 0.46→0.41) | 0 | jarvis | analysis | 11-06 |
| P2 | supp 표 test/val 불일치 재측정 | 해당 행 val 로 재측정 | 0 | yeon | analysis | 11-07 |
| P3 | 구조 의존성(DRN-261029-02·03) | DGFusion·CAFuser 두 패스+KD 구현 → 재학습 1런씩 | 재학습 ×2 | levine 3090×4(환경 확인 필요) | specialist-baseline(**미개방, user 결정**) | 11-05 착수 시 11-09 |
| 취소 | L4 M2F 제거 s902(yeon) | 쿼리 헤드 제외 결정으로 불필요 → **중단 권고**(user 승인 필요) | — | yeon 4장 회수 | manager | 11-04 |

## 2. 주장별 실험 설계

### 2-1. "고해상 경로가 도움이 된다"
- 이미 있는 근거: A2 vs Q2 같은 시드 3쌍(24클래스 +1.50, 얇은 4 +3.04), 클래스별 이득 그림(fig_thin), 해상도 반응(DRN-261016-03).
- 추가(특징 수준, DRN-261104-02): FPN s4 세 공급원 분리 그림 + 얇은 클래스 화소 집중도(H-V1 ≥1.5배) + 추론 시 경로 제거(H-V2). 주의: `hires_infer_mode='off'` 는 학습 때 본 적 없는 결선이라 전체 −10.6 이 나온다(DRN-261020-01). 그림에는 off 와 recombine(dense 만 off) 둘 다 두고, 본문은 "경로를 빼면 얇은 클래스가 가장 크게 무너진다"는 **상대** 비교만 쓴다.
- 부작용도 같이 보인다: TrafficLight 는 재현율 +, 정밀도 −(실패 사례 절). 숨기지 않는다.

### 2-2. "두 패스 학습(C2D2)이 도움이 된다" — clean 이 아니라 강건성
- 세 행 분해(모두 고해상 경로 유지, test clean + EMM): L5 열화 패스 제거(= clean + KD 만) · L6 교사 증류 제거(= 두 패스만) · L7 물리 열화 제거(= 결측 0채움만, RobustSeg basic distillation 등가). 사전 예측: EMM 은 L5 ≪ L7 < L6 ≤ A2, clean 은 L6 ≈ A2(L6 s821 58.77 이미 관측).
- 구조 의존성: 같은 두 패스+KD 를 DGFusion·CAFuser 에 얹었을 때 EMM 이 열화 재학습(45.52·46.50)보다 오르는가. 오르면 "학습 방법은 구조에 독립적으로 강건성을 준다", 안 오르면 "동결 VFM+LoRA 와 결합할 때만". 어느 쪽이든 쓸 수 있는 문장이고, 없으면 리뷰어 "증강 강화"를 못 막는다. **일요일까지 결과는 불가**(착수 자체가 안 됨). user 결정: 세션 개방 + levine 사용 여부.

### 2-3. RA-L(2차 논문) 노벨티 가안 검증 — 배치·진행 유무
| 가안 | 항목 | 상태 | 이번 주 배치 |
|---|---|---|---|
| S2 센서별 마스크 헤드(고해상 특징을 센서별 마스크 로짓 전용) | 구현 DRN-261020-06 · 스크린 -03/-04/-05 | 구현 queued, 미착수 | 교차 확인(DRN-261022-01) 통과 → **구현은 코덱스 워커로 11-05~06 착수 가능**(method 검수). 스크린 1시드(s821)는 levine 또는 yeon 여유 시 11-07 기동, 결과는 다음 주 |
| 내용 기반 토큰 MoE-LoRA | 구현 -04 · 스크린 -05 | idea | 이번 주 배치 안 함(Access 동결 후) |
| 공유 LoRA(A2-shared) 재료 | L1 강건성 측정 완료 | 완료(단일 센서 폴백 손실) | 추가 실험 없음 |
| 실패 키 H1~H3 | DRN-261020-01 | 판정(§4) | 후속: Pole·Pedestrian 의 off/recombine 변화(DRN-261104-02 에 포함) |

## 3. 서버별 배치 순서(manager 용)

- **yeon**: ① GPU0·1 = MUSES s2·s3 공식 val(analysis) ② L4 s902 중단 승인 시 GPU2~5 = MCubeS C2D2 768 2시드(hpca100 사망 시) 또는 L5·L6 EMM 평가 ③ GPU7 = supp 재측정. GPU6 은 CUDA 고장(메모리 `yeon-gpu6`).
- **jarvis**: L8 s902 완주 후 GPU2·4·5·6 = L7 legal·EMM → 고해상 특징 시각화(DRN-261104-02) → router 끄기(DRN-261029-06). 정성 은행 덤프 잔여.
- **bengio**: L2 s902 완주(11-05) → legal v2.
- **amphibious-demo**: L8 s821 → E1Mc-B 교사 연쇄(그대로).
- **hpca100**: 상태 확인이 최우선(임대 종료 경고). 살아 있으면 MCubeS C2D2·NM 측정 회수 + ckpt 백업.
- **levine 3090×4(빈 GPU 3·4·5·6)**: 환경(venv·DELIVER 캐시) 확인 후 구조 의존성 또는 S2 스크린 후보.
- 각 학습 완주 시 `tools/qualbank_dump.py` 전체 split 덤프(user 11-02 규칙).

## 4. 11-04 method 판정(lab-plan verdict 로 기록됨)
- DRN-261022-01 벤치 교차 확인: **adopt** — (a) 객체 클래스 오라클 DELIVER +4.22·MUSES +2.18 로 H-X1 충족. (b) 선형 프로브 H-X2 는 fused 도 AUROC 1.0 이라 전제가 깨져 판별력 없음. PRB-DRN-261020-01 조건 충족 → S2 진행 가능(RA-L 재료).
- DRN-261020-01 실패 키: **adopt(부분)** — H1 TL FP 는 고해상 경로가 만든다(off FP −95%·−99%), 단 TP 도 같이 사라져 "TL 예측 경로 자체". H2 미검증. H3(MUSES 야간 = 열화 패스 과소 표집) 기각.

## 5. user 결정 필요
1. yeon 의 L4 s902 학습 중단(쿼리 헤드 논문 제외로 불필요) → GPU 4장 회수.
2. MCubeS C2D2 가 hpca100 에서 죽었을 때: yeon 에서 즉시 재기동(완주 11-09~10, 동결일과 겹침) vs Table 3 를 "w/o C2D2 모델(60.27)" 만으로 내고 C2D2 는 보완 제출.
3. 구조 의존성 실험(DGFusion·CAFuser 두 패스+KD): specialist-baseline 세션 개방 + levine 사용. 일요일 내 결과 불가, 동결 전(11-09) 가능성만 있음.
4. S2 구현을 이번 주 코덱스 워커로 착수할지(Access 와 무관, RA-L 재료).
