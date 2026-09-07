---
created: 2026-09-07
author: fable (background 세션, 사용자 요청 "착안 이유 + 평가 방법 총정리, SOTA 병목 진단")
sources: models/arch-evolution.md(표1·표2·§0.5) · research/hypothesis-ledger.md(H1~H22) · research/synthesis-tried-map.md · experiments/analysis/* · issues/issues-and-fixes.md · status/current.md · model_competitiveness_report.md(2026-08-08)
---

# 착안·평가 회고 — 45세대가 무엇을 풀려 했고, 무엇으로 판정했으며, 왜 SOTA 앞에서 멈춰 있는가

> **역할**: P8~P52 전 세대의 "직전 문제 → 착안 → 결과 → 판정 → 검증 수단"과, 그 판정을 내린 **평가 프로토콜의 변천**을 한 문서로 모은 회고. 수치는 전부 legal(val-best 또는 final-iter)만 쓰며 test-best는 배제한다.
> 세부 수치의 정본은 여전히 [models/arch-evolution.md](../models/arch-evolution.md) 표1·표2, [hypothesis-ledger.md](hypothesis-ledger.md), [experiments/log.md](../experiments/log.md)다. 이 문서는 그것들을 **한 눈에 잇는 지도**이고, §4의 병목 진단이 이 문서 고유의 판단이다.

---

## 0. 한 줄 결론

**SOTA가 막힌 원인은 모델 아이디어가 고갈돼서가 아니라, (1) 쫓아온 효과 크기가 측정 노이즈보다 작았고, (2) 45세대 중 30세대 이상이 같은 가설("모달 신뢰도에 따른 적응 가중")의 변주였으며, (3) 격차의 실체(RGB 표현력·정보량)를 겨냥한 축은 이제 막 한 번 성공(P50 +0.74)했을 뿐이기 때문이다.** 측정 인프라(legal 규약·native GT·드라이버 통일·하네스 가드)가 2026-07-15부터 08-31까지 단계적으로 갖춰지면서 "SOTA 한 걸음"으로 보였던 DELIVER 격차는 정본 재측정 후 약 3점으로 드러났다.

---

## 1. 계보 지도 — 다섯 시대

| 시대 | 기간 | 버전 | 공통 가설 | 결과 한 줄 |
|---|---|---|---|---|
| Ⅰ 학습형 융합/게이팅 | 2026-02~04 | P8~P26 (SAM2, MULTIAQUA) | 학습된 게이트·MoE·품질 예측기가 모달 가중을 맞춘다 (H1) | 12세대 전원 P9(사실상 상수 가중)을 못 넘음. 게이트는 상수로 수렴, 신호를 공간화하면 오차만 증폭 |
| Ⅱ RBMA attn-bias | 2026-04~07 | P27~P33 (SAM2, DELIVER) | 무학습 신뢰도를 attention logit에 additive bias로 (H2) | P32에서 순손해 p=4.5e-22. 신호 AUROC를 올려도(P31 calibration) 성능은 불변 |
| Ⅲ 백본 전환 + 모듈 얹기 | 2026-07-13~07-21 | P34~P39.1 (DINOv3) | 백본을 바꾸고 그 위에 라우터·쿼리 헤드를 얹는다 | 백본 교체 단일 변수 +11.6(H6). 얹은 모듈은 전부 no-op 또는 유해(H3). P39.1(gated-MLP 트렁크+VICReg)이 기준선 |
| Ⅳ 약모달 강제·rank·쿼리 | 2026-07-22~08-05 | P40~P48 | lidar를 억지로 쓰게 하거나 fusion rank를 올리면 산다 | 마스킹 3변형·gradient 균형·η² 규제 전멸. "dense∥query 이중 중복 시스템에 얹은 모듈은 흡수된다" 발견 |
| Ⅴ 소거·프로브·학습전용 손실·정보 축 | 2026-08-05~09 | P46-C3, CEA oracle, ProbeA2, spatial oracle, P49, P50, P51, N1~N9, P52 | 남은 축을 상계·통제 실험으로 하나씩 닫고, 살아남는 축만 P52에 싣는다 | 양성 = C3 prototype(클래스축, dose-response 2/2 적중)·P50 정렬 사전학습(+0.74)·MCubeS 1위. 반증 = 조건 적응(oracle +0.2)·백본 스케일(7B +0.18)·공간 선택·비대칭 주입·LoRA 결합·트렁크 우위 |

가설 원장 기준 판정 집계(H1~H22, 파생 포함 25항목): **확인 7 · 반증 17 · 미결 1**. 확인된 7개는 전부 "표현·정보·학습 신호" 축(H5 상호작용 실재, H6 백본, H7 해상도, H8 C3, H20 dose-response, H22 사전학습)이고, 반증된 17개는 대부분 "추론 경로 안의 배분" 축이다.

---

## 2. 버전별 착안 경위 (문제 → 착안 → 결과 → 판정 → 검증 수단)

수치 규약: MULTIAQUA는 M-score, DELIVER는 val/test mIoU, MUSES는 val / 공식 Codabench test. 모달 수·PhysAug 여부가 다르면 표기.

### 2.1 시대 Ⅰ — SAM2 + MULTIAQUA (P8~P26)

| 버전 | 직전 문제 | 착안한 방법 | 결과 | 판정·기록된 원인 | 검증 수단 |
|---|---|---|---|---|---|
| P8 | 멀티모달 기준선 부재 | ConfidenceHeadV2 + sigmoid UAMM | M 78.45 | sigmoid 포화 → 가중 전부 ~1.0, AMF uniform 퇴화 | M-score, gate 값 관찰 |
| **P9** | sigmoid 포화 | CrossModalFusionHead + max-norm UAMM | **M 81.98 → 82.10(재제출)** | 장기 최선. 단 가중치는 사실상 상수(thermal 1.0/lidar .96/img .74), 이득은 파라미터 증가분. UAMM 스칼라곱은 Pre-Norm+Residual에 상쇄 | M-score, per-token routing 분석(entropy_ratio) |
| P10 | 게이트에 감독 없음 | ModalAuxHead + oracle KL | 79.27 | oracle이 주간에 과적합, 멀쩡한 gate에 감독을 얹어 순해악 | M-score(test 하락) |
| P11 | 게이트가 uniform으로 보임 | MI routing loss | 77.09 | "uniform"은 공간평균 측정 artifact. 이미 분화된 gate를 제약 | per-token 분석으로 artifact 확인 |
| P12 | 입력 무관 라우팅 | input-conditioned Soft-MoE LoRA | 80.80 | cond_proj zero-init 무기여, expert collapse 악화 | expert usage 분석 |
| P13 | 확신 신호 부재 | energy-score fusion + init fix | 81.21 | energy=확신≠정답, LiDAR "confidently wrong"(ISSUE-009); init은 resume이 무효화 | ISSUE 진단, night-val |
| P14 | 모달별 근거 부재 | per-modal aux decoder ×3 | 74.27 | frozen backbone 위 aux mask 부정확(ISSUE-008), Sky 붕괴 | per-class IoU |
| P15/P16 | 스칼라 신뢰도의 한계 | calibrated **spatial** entropy fusion | 71.05 / 68.42(최악) | 부정확 신호를 per-pixel로 증폭. thermal 저-entropy 지배 0.923 → Sky 3.17 | per-class IoU, 가중 히스토그램 |
| P17 | aux 해상도 | multi-scale FPN aux | 73.23 | Sky 회복(3→33)하나 aux 품질 천장 그대로 | per-class IoU |
| P18 | aux 품질 | trainable ResNet-18 aux backbone | 기록 없음 | — | — |
| P19 | 신호 대신 학습 | spatial head 직접 학습 | 69.63 | train-night 과적합, LiDAR 지배 0.992 | night-val vs test |
| P20 | MoE 용량 | shared gate + higher rank | 기록 없음 | — | — |
| P21/P22 | 구조 refinement 미실시 | DeBA-FP(deformable bottleneck adapter) 단일/멀티스케일 | 81.77 / **82.10(P9 동률)** | 구조 refinement는 유효(Dynamic +15). fusion 상수수렴은 미해결 | M-score |
| P23 | adapter를 백본에 | MoE DeBA-BB | OOM | — | — |
| P24 | 품질 신호 학습 | SQG distill(per-modal decoder teacher) | 미제출 | teacher sigmoid ep40 포화(ISSUE-013) | teacher 출력 모니터 |
| P25 | 스칼라→공간 품질맵 통합 | unified spatial quality fusion | 27ep 중단 | predictor 용량 부족(std 0.05 vs 0.40). lidar/thermal 단독 mIoU 18/15%라 RGB-first가 합리적 | 단독 모달 mIoU |
| P26 | SQG 붕괴 | per-modal SQG v5 설계 | 설계만 | SQG 계열 붕괴 진단으로 종료 | — |

### 2.2 시대 Ⅱ — RBMA on SAM2, DELIVER (P27~P33)

| 버전 | 직전 문제 | 착안한 방법 | 결과(val/test) | 판정·원인 | 검증 수단 |
|---|---|---|---|---|---|
| P27 | 학습형 가중 전멸 | 무학습 신뢰도를 memory-attn pre-softmax **additive bias**로 (배관) | 기록 없음 | 기구만 성립 | — |
| P28 | 신뢰도 신호 정의 | bias = training-free self-entropy | 57.87 / 50.61 (ep16 사망) | geometry 모달 anti-calibrated(event AUROC .30, lidar .22) | per-modal AUROC |
| P29 | 조건(주야) 미반영 | SDC: 이미지에서 무감독 파생한 조건 latent → FiLM Soft-MoE gate | 63.20 / 54.34 | gate 상수수렴, event/lidar drop Δ 0.02/0.01. **class-transfer(Wall 62→2)는 라우팅으로 불가** 실증 | drop-modality, per-class·per-condition |
| P30 | thin-class 사망 | class-token decoder가 conv head **대체** + reliability-anchored router | 49.76 / 44.10 **붕괴** | CTD 대체 단독 귀속. router는 hook 미호출로 200ep 미실행(ISSUE-022) | pred_agreement로 죽은 모듈 감지 |
| P31 | 신뢰도 anti-calibration, P30 붕괴 | per-modal temperature + calibration loss, CTD aux-only 강등, Hiera 마지막 3블록 unfreeze(다른 변경과 묶음) | 63.20 / 54.75 | 성능 정체. calibration(lidar AUROC .38→.97)·router(off −10.7~13.8)는 실효, bias는 Δ≈0 | 모듈 on/off, AUROC |
| **P32** | 자기확신 신뢰도의 confound | bias = cross-modal corroboration + unique-info veto | 64.12 / 55.00 | 🔴 attn-bias **순손해 ΔmIoU −0.013, p=4.5e-22**. AUROC↑(event .54, lidar .81)인데 drop Δ 여전히 0 → "신호 품질↑ ≠ 라우팅 이득" | Phase-0 무학습 AUROC 게이트, 1897장 4축 독립검증(Wilcoxon) |
| P33 | soft bias로 competence≈0 모달을 못 살림 | competence-gated hard fusion + 비대칭 modality dropout | 기록 없음(launch만, B200 마감) | 미완결. "무조건 dropout no-op"이 1차 로그 없이 후속 인용됨 | — |
| SAM3-RBMA | 백본 무관성 | RBMA를 SAM3 plain ViT로 이식 | val ~24 plateau | single-scale 한계 → ablation 용 | — |

### 2.3 시대 Ⅲ — DINOv3 전환 (P34~P39.1)

| 버전 | 직전 문제 | 착안한 방법 | 결과 | 판정·원인 | 검증 수단 |
|---|---|---|---|---|---|
| **P34** | SAM2 피쳐 rank-1 붕괴, 모달 비정렬(CKA ~0.1) | 백본 교체 DINOv3-L frozen + per-modal LoRA + SimpleFPN (+RBMA bias·consistency·veto 유지) | DELIVER 68.19 / 56.62 (PhysAug ON = 공정선 밖) · MUSES 80.86 / **78.979** | 계보 최대 성공. 단 원천은 백본: 제안 모듈 toggle Δ≈0.00 | **ProbeA1** 백본 통제 프로브(+11.6), module_ablation |
| P35 | 증강 조건 불공정 | 아키 변경 0, −bias −consistency −PhysAug | 67.61 / 55.52 | 공정선 확립. physaug 실제 크기 −1.12 = 제안 모듈 전체보다 큼 | 1-변수 config diff |
| P36 | P31 router가 유일 실효 | P35 + per-class reliability-anchored router | **67.74 / 55.62** (공정 legal 내부최고, 1024² 학습) | 유일 생존 모듈이나 +0.13/+0.10은 노이즈 대역. router_off +38~42는 co-adaptation 의존 | P35 대비 A/B, router off 토글 |
| P37a/b | 클래스축 라우팅 명시화 | CEFR per-class 라우팅 헤드 / ClassToken-lite | MUSES 81.16 / 미제출 · seg 기록 없음 | per-class 분화 0/19, cefr_off no-op. P37b는 비미분 mask_proj가 영구 random init(ISSUE-024) | 라우팅 엔트로피·winner, 토글 |
| P38 | PQ 산출 불가, thin-class | Mask2Former-lite query head, β zero-init 잔차 | MUSES **82.22 / 79.025** · DELIVER 65.19(ISSUE-026 오염) | 모듈은 추론 no-op(β 0.133), Wall −6. 부산물: deep-sup이 router 의존 +38→+2로 해소, PQ 경로 확보, fog train 클래스 0→100 | m2f_off/router_off, thin-class 게이트 |
| P39 | 실패-키 5개 문서화 | DPC: rank 확장·modal-token·anchored query·path dropout 경쟁·router 직접 CE | MUSES 81.52 / 78.881 · DELIVER 65.68(오염) | 5세대 만의 첫 non-no-op이나 성능 실패. **val↔test 순위 역전 2건**, fog_night −12, gate/calib "유해" 재판정 | ep 조기 토글 즉검, 3-ckpt 비교, 사전등록 thin-class 게이트(0/3) |
| **P39.1** | lidar eff-rank 4.7 붕괴 | gated_mlp trunk(γ init 0.1, zero-init 금지) + VICReg + gate/calib/veto off | MUSES seed2 **82.62 / 79.788**(5-seed 평균 82.03, 범위 0.92) · DELIVER 67.60/54.34 | **현행 기준선**. rank 20배 복구에 test +0.76. 이후 6세대 정체 | 사전등록 ep30 게이트(rank≥15, drop-lidar fog_night≥4), 5-seed, 공식 제출 |

### 2.4 시대 Ⅳ — 약모달 강제·rank·쿼리 (P40~P48)

| 버전 | 직전 문제 | 착안한 방법 | 결과 | 판정·원인 | 검증 수단 |
|---|---|---|---|---|---|
| P40 | 추론 재가중 반증 | 같은 신호를 학습 시 img 감쇠 조건화로 | 미기동 | 같은 목표의 P42·P44 전멸로 전제 흔들림 | — |
| **P41** | FUSED rank 6.8/256 병목 의심 | Phase-0(학습 0) η²·스펙트럼 판별 → `L_fcr = −λη²` | η² 0.35→**0.94**, mIoU 불변 | 🔴 가장 깨끗한 기각: 사전등록 falsification 정확히 실현. fusion rank는 레버가 아님 | **학습0 Phase-0 → 사전등록 → 조기 확정**(이후 표준) |
| P42 | img 과지배, lidar 미사용 | 학습 배치 FRAC에서 img 입력 0 마스킹 | FRAC 0.3/0.5/0.7 = 81.53/80.85/79.13 | 가릴수록 단조 열화. 게이트①(dMIoU-lidar↑) 측정 기록 없음 | FRAC 스윕 |
| P43 | β 잔차 no-op | 잔차 제거, 두 헤드 독립 주손실 + lateral | MUSES **82.51 / 79.351** (우리 2위) | 결선을 완벽히 고쳐도 −0.44 → **이중 중복(dense∥query 각각 99%)** 발견의 첫 신호 | 공식 제출, 표준분석 |
| P44 | 전역 마스킹 실패 | MMPareto gradient 통합 + peer 증류 + 국소 마스킹 (+P45 FogStyle off) | MUSES 80.71 / 78.429, **fog_night −13.2** | 완패 + "야간 lidar 편중 유리" 가설이 test에서 정반대 | 사전등록 게이트(dMIoU-lidar>1 미달), 공식 제출 |
| **P46** | val→test 하락 = per-class 전이 붕괴(Wall 62→2, 복구 상한 +7.9) | 학습 전용: C1 rare-class sampling / C2 masked-context consistency / **C3 per-class EMA prototype CE** | C3-only: base 대비 test **+1.35~1.74**, RailTrack 4→68. MUSES 이식 −0.765. 정본 5-seed **54.39±0.76**(best 55.29) | 혼합: RailTrack 게이트(4→≥40) 압도 통과, overall SOTA 미달. C1 순유해, C2 −1.67 유해(H15). "57.05 돌파"는 test-best라 철회 | 구성요소 귀속, λ 스윕(val-best/final-iter 병기), seed, 매칭 A/B |
| P47-1 | lidar 투영 희소 | DGFusion식 팽창+motion-comp(코드 0줄) | 82.58 (−0.04) / 78.790 | 노이즈 봉우리. "내부최고 82.62도 노이즈 봉우리일 수 있다" 경고 | 사전등록 val≥82.62 |
| P47-2 | 모달 추가할수록 악화 = laziness | 모달별 aux head + uni-modal CE (UniBal) | 81.93 (−0.69) | 실패 + **진단 반증**: RGB에 3.2× 가중 줘도 img acc 불변 | 2-arm, per-modal acc/ce 로그 |
| P48 | 쿼리가 dense 복제 | 인스턴스 단위 타깃으로 쿼리 감독 | 미실행 (things PQ 22.87) | 폐기 → 게이트 적용 시점 오류 지적 → user 취소(PQ 비경쟁축), H10 미결 동결 | 사전등록 PQ 게이트(시점 오류) |

### 2.5 시대 Ⅴ — 소거·프로브·정보 축 (2026-08-05~09-07)

| 실험 | 직전 문제 | 착안한 방법 | 결과 | 판정·원인 | 검증 수단 |
|---|---|---|---|---|---|
| CEA oracle | 조건×모달 상호작용은 실재(H5) — 추출 단계 조건 전문화 미점유 | 조건 서브셋별 LoRA+trunk 전문가 vs 공용 (상계) | fog_night **+0.21**, night +0.02 | 적응 가설 3단계(융합→추론→추출) 완결 폐쇄(H4). 주야갭 4.33 중 +0.02만 회수 = 격차는 배분이 아니라 **정보** | 사전등록 G-P1(≥+1.0)/G-P2, LR 쌍 |
| ProbeA2 | "남은 격차 = RGB 표현력" 검증 | frozen S+/B/L/H+/7B + 동일 경량 head | S+ 59.85 → L 68.67 → H+ 69.19 → 7B 69.37 | L→7B +0.70(8×): 표현력 축 **frozen 스케일업으로는 소진**(H12). L−S+ +8.82 = 대형 백본 전제 정직 공개 필요 | 사전등록 G-A2 상·하한, 축분리 |
| Spatial oracle (H16) | user 재개방: 공간×모달 라우팅 여지 | 15부분집합 픽셀별 GT-argmax 상계 → no-GT 라우터 3종 | 상계 **+8.5** 실재 / no-GT majority −1.9·consensus −2.0·confidence −12.5 | 여지가 anti-consensus·anti-confidence → 추론-시간 **선택 불가**. 전역 drop≈0 ≠ 국소 상보 부재 | 사전등록 게이트 2단 + 실현성 통제 |
| H17 xattn | user 가설: 명시적 cross-attn 트렁크 우위 | drop-in 교체 A/B | 54.94 < 56.99 (−2.05, 파라미터 15.9×) | 반증. 믹서는 비병목 | 동일 시드·레시피 A/B |
| P49/P49.1 AIR | MM-SA(현 SOTA) 해부: 비대칭 RGB 주경로 + 인코더 내부 주입 | DINOv3-L **부분 fine-tune**(RGB_FT, LoRA 폐지) + ConvNeXt-S 보조 + zero-init γ 주입 + query 제거 (A1~A5 동시) | P49 γ 정체(5번째 zero-init 사망) → P49.1(γ 0.1): DELIVER 56.19 < 56.99, MUSES 81.16 < 82.13 | H13·H14 반증: 강한 RGB 백본일수록 주입 이득 축소. ⚠️ A1(백본 FT) 단독 기여는 분해 측정 기록 없음 | ep30 게이트(γ 성장), 벤치 완주 게이트, 4모달 G-4M |
| **P50-MAP** | 융합 정교화 상한 ~1.2; VLM 교훈 "connector는 2차, 정렬 사전학습이 1차" | Places365 200k pseudo-모달로 어댑터만 정렬 사전학습 → 파인튠 (추론 그래프 무변경) | legal test **54.95 = +0.74** ≥ 게이트 +0.5 (H22 ✓) | 캠페인 "정보 축" 첫 양성. 이득은 조기 수렴 국면(val-best ep30). **Phase2(500k·6모달)는 ep30 53.10으로 기각(09-07)** — 원인 분석 기록 없음 | 사전학습 유/무 매칭 시드 A/B, N8 재선택 |
| P51-CMLC | 선택 축 종결 후 "인코딩 시간 결합" 미검 | LoRA 저랭크 코드의 대칭 결합(γ init 1.0) + RGB 마스킹 | Δ(on−off) **−0.82**, fog −2.03·night −0.58 | H19 반증, "기제 사용≠유익" 5번째. 융합-기제 4갈래(선택·attn·평균·결합) 전부 소거 | on/off 매칭 페어 |
| N2 mean-fusion | "우리 트렁크 우위" 주장 검증 | MLE-SAM식 산술 평균 baseline | 55.45 ≥ gated-MLP 54.2~55.4 | H21 반증: 트렁크 우위 철회, 기여 후보는 VICReg으로 축소(N7 격리 진행 중) | seed821 매칭 |
| N4/N4b MCubeS | 3번째 벤치·dose-response 예측 | 통일 레시피 C3-off / C3-on 예측 사전등록 | **58.07±0.49 (1위, +3.42)** / C3-on rubber +9.76·overall −0.10 (예측 2/2 적중) | H20 ✓: C3 효과는 class-transfer 붕괴 강도에 단조 → 논문 헤드라인 기둥 | 3-seed, 사전등록 예측 |
| N6/N8 재선택 | ISSUE-033 드라이버 통일 | legal-val 재선택 | DELIVER 5-seed 53.82→**54.39±0.76**, best seed816 55.29 | 선택 아티팩트 2/5런, base outlier 서사 해소 | 하네스 가드 |
| E-LoRA | H22가 "per-modal 어댑터 미정렬" 증거 | A per-modal r16 / B 공유 / C 공유+잔차 | arm A 착수(09-07), B·C 미착수 | 미판정 | 사전등록 C ≥ A−0.3 |
| **P52 RxDINO** | 오프라인 벤치별 C3 on/off = per-dataset 튜닝과 구분 불가(user 지적) | 진단의 온라인화: C3-adaptive(혼동 EMA→λ_c) + UniBal-adaptive(손실갭→λ_u) + P50 init, 3벤치 단일 config | 본런 착수 단계, 결과 없음 | 미판정 — 캠페인 최종 실험 | G1 DELIVER ≥54.65 · G2 MUSES ≥81.12 · G3 MCubeS ≥57.77 · G4 λ 궤적 창발 |

---

## 3. 평가 방법 총정리

### 3.1 벤치별 채점 프로토콜과 그 변천

| 벤치 | 지표·프로토콜 (현행 정본) | 변천·확정 계기 |
|---|---|---|
| MULTIAQUA | **M-score = (val+test)/2** (챌린지 공식). val=주간 145장 로컬, test=야간 챌린지 서버(`--macvi`) | ⚠️ CLAUDE.md의 "0.75×val+0.25×test"는 낡은 표기 — [experiments/log.md](../experiments/log.md) 상단이 검증 후 0.5/0.5로 정정. day-val이 야간 test proxy가 아님(ep131 vs ep188) → night-val ckpt 계열 신설(ISSUE-001) → night-val 증강 오염(ISSUE-007) 제거 |
| DELIVER | `val.py` **native 1042² GT · BS1 · val-best 또는 final-iter** · 학습 768²/평가 1024²(mismatch 논문 명시) · 하네스 가드 `--check` 통과 필수 | 07-15 test-best 철회(P34 57.60) → legal 규약 · 08-06 fair-eval(@1024 평가) · 08-14 트레이너 768-리사이즈 GT 낙관 발견(P49.1 57.68→55.66) · 08-15 @1024 **학습**은 test 유해(N=3) · **08-26 ISSUE-033** ERC(1024-리사이즈 GT, +2.56 낙관) vs val.py 이원화 → val.py 정본, 구 헤드라인 69.44/56.99 → 66.88/55.18 · 08-31 하네스 가드(8파일 SHA256 동결) |
| MUSES | `tools/eval_muses_official.py`(CAFuser 평가기와 동일, native 1080×1920) · Codabench 공식 test(GT 비공개, val-best 단일 선택) · **제출 게이트: 공식 val ≥ 82.62일 때만 1회** | 07-15 내부 letterbox 81.02 ≠ 공식 80.86 → native 평가기 · SOTA 오독 정정(79.72=DGFusion val, 82.39=GtA camera-only) · val 단독 이득의 test 미전이 3회(P46-C3·P49.1·P47-D1, 전이율 ~4%) → 제출 게이트 |
| MUSES PQ | `tools/eval_pq.py`(로짓 단계 역변환, 공식 AUPQ 스코어러와 일치) | 08-06 첫 측정 things 22.87 / All 35.55. 경쟁 논문 전부 mIoU만 → 비경쟁축, limitation 절 |
| MCubeS | 커뮤니티 표준 test split(102장, `list_folder/test.txt`) | 08-25 CMNeXt·MMSFormer 로더와 동일 split 검증 후 진입 |
| 공개표 정합 | MM SAM-adapter = native GT(우리와 동일) · CMNeXt/CAFuser/DGFusion = 1024-리사이즈 GT(낙관) → 이들 대비 우리 수치는 과소 | 08-14 3개 repo 1차 소스 확인, 1042→1024 델타≈0 실측 |

### 3.2 판정 설계의 진화 (무엇으로 "됐다/안 됐다"를 정했나)

| 단계 | 도입 시점 | 방법 | 도입 계기 |
|---|---|---|---|
| ① 모델 선택 proxy | 02~03월 | day-val → night-val ckpt → 증강 제거된 night-val | day-val↑인데 test −2.21; night-val 오염 시 test −19.5 |
| ② legal ckpt 규약 | 07-15 | val-best 또는 final-iter만, test-best 금지, 둘 병기 | test-best 철회 3회(07-15·08-03·08-04). 규칙마다 λ 순위가 뒤집힘 |
| ③ 모듈 토글 ablation | 07-12~21 | `tools/module_ablation.py`: \|Δ\|<0.5 & agreement>0.99 = no-op, +20↑ & agreement<0.8 = co-adaptation 의존, pred_agreement≈1 = 죽은 모듈 | ISSUE-022(router 200ep 미실행) 감지 필요 |
| ④ 사전등록 게이트 | 07-20~ | 기동 전 통과/기각 기준 + **적용 시점** 명기, ep30 조기 판정 | P39 thin-class 게이트, P48 게이트 시점 오류 재발 방지 |
| ⑤ 학습0 Phase-0 판별 | 07-22 | 기존 ckpt로 병목 여부를 먼저 재고 학습 | P41 FCR(η²↑ mIoU 불변으로 즉시 기각) |
| ⑥ 매칭 시드 A/B | 07-15~ | 동일 시드·레시피·측정기, 유일 변수 1개 | P34 vs P36이 PhysAug 미통제로 무효 → P35 vs P36 |
| ⑦ multi-seed mean±std | 08-20 | 3~5-seed 필수, best 병기, 비교 비대칭 명시 | **가짜-시드 사고**: `fix_seeds(3407)` 하드코딩이라 "편차 0.59"는 GPU 비결정성뿐, 진짜 시드 std 2.21 → 근SOTA 철회 |
| ⑧ oracle 상계 프로브 | 08-08~20 | GT를 주고 얻는 상한을 먼저 잰 뒤 실현 가능성 통제 | CEA(+0.2 → 폐쇄), spatial(+8.5 → no-GT 3종 음수) |
| ⑨ drop-modal ablation | 07-30~08-10 | 추론 시 zero-fill Δ **+** 학습부터 제거한 재학습 Δ | 추론 Δ≈0인 depth가 학습 제거 시 −2.85 → 두 가지를 구분 |
| ⑩ 통제 프로브 | 07-12 / 08-09 | frozen 백본 + 동일 경량 head로 백본만 교체(ProbeA1/A2) | 백본 기여 분리(+11.6), 스케일 소진(7B +0.18) |
| ⑪ dose-response 예측 | 08-25~27 | 기동 전 방향·크기 예측 등록 → 실측 대조 | N4b 2/2 적중 = 사후설명이 아닌 예측력(H20) |
| ⑫ 하네스 가드 | 08-31 | 채점기 8파일 SHA256 동결, FAIL 상태 수치 인용 금지 | ISSUE-033 드라이버 혼용 |

### 3.3 분석 도구 스위트 (새 모델에 코드 새로 짜지 말 것)

표준 분석항목 1~4([tools/README_seg_analysis.md](../../tools/README_seg_analysis.md)): ① 어댑터 적응도(`modal_adaptation.py`·`adapter_health.py`) ② 모달별 피쳐 통계(`feature_stats.py`: norm·dead-ch·eff-rank·CKA, `viz_features.py`, `module_diagnostics.py`) ③ 모듈 전후(`module_ablation.py`, `eval_reliability_auroc.py`) ④ 클래스×도메인(`eval_per_domain.py`→`analyze_per_domain.py`→`compare_models.py`: STRUCTURAL/DESIGN-GAP/DOMAIN-GAP/SOLVED 자동 분류). 원스톱 `seg_analysis_pipeline.py`. 평가기: `val.py`·`eval_muses_official.py`·`eval_pq.py`·`eval_harness_guard.py`·`oracle_spatial_modality.py`·`probe_backbone_scaling.py`·`verify_submission.py`. `eval_reliadino_ckpt.py`는 낙관 채점이라 **토글 진단 전용으로 강등**.

### 3.4 평가 사고 27건의 유형별 요약

| 유형 | 건수 | 대표 사례 → 조치 |
|---|---|---|
| 체크포인트 선택 오염 | 5 | test-best 인용 3회 철회 · 중간 ckpt "예비 도달" 철회 · val-best 이동 시 test −2.76 |
| 채점기·해상도 | 6 | 트레이너 768-리사이즈 GT 낙관 · ERC vs val.py 이원화(ISSUE-033) · MUSES letterbox ≠ native · 공개표 프로토콜 혼재 · 구 per-domain 로그 혼용 · BS 의존 오진 |
| 시드·재현성 | 3 | 가짜 시드 · 모듈 중복 생성으로 seed 파손 · A100 TF32로 peer 재현 FAIL |
| 게이트 스펙 | 3 | P48 적용 시점 오류 · oracle 사전확률·널 부등호 역방향 2회 |
| 비교 기준 오독 | 4 | SOTA 기준 낡음(68.79→69.60) · DGFusion val/test 혼동 · 존재하지 않는 모델 쌍(67.74/56.62) · PhysAug 미통제 비교 |
| 파이프라인 버그 | 6 | ISSUE-022 router 미실행 · ISSUE-025 radar 디코딩 · ISSUE-026 ColorAug 붕괴 · ISSUE-029 HF offline RANDOM INIT · ISSUE-032 no_grad 누락 · PQ 게이트 오보 |

---

## 4. 병목 진단 — 왜 SOTA 앞에서 멈춰 있나

### 4.1 효과 크기가 노이즈 아래에 있었다

| 항목 | 값 | 출처 |
|---|---|---|
| 캠페인이 쫓은 모듈 증분 | +0.10(P36) ~ +0.76(P39.1 test) · −0.04(P47-1) · −0.44(P43) | 표1 |
| DELIVER 진짜-시드 std (08-20 판정 시) | 2.21 | seed-variance verdict |
| DELIVER 5-seed std (드라이버 통일·재선택 후) | 0.76 | N6 |
| MUSES 5-seed 범위 | 0.92 | P39.1 |
| val-best ckpt 이동 한 번의 test 변동 | −2.76 | monitor-log 08-07 |
| SOTA와의 격차 (DELIVER, 08-08 보고 → 정본) | −0.36 → **−2.96(mean) / −2.06(best)** | ISSUE-033 |

08-20 이전의 단일 런 판정은 대부분 노이즈 대역 안에서 내려졌다. 쫓아온 이득(0.1~0.8)이 시드 편차(0.8~2.2)보다 작았으므로, 그 기간의 "미달/돌파" 판정 상당수는 원리적으로 판별 불가능한 것이었다. 정직한 측정이 갖춰진 뒤(08-26) 격차는 "한 걸음"이 아니라 약 3점으로 확정됐다. 즉 **병목의 첫 번째 정체는 모델이 아니라 측정 분해능**이었고, 이것은 지금은 해결됐다(3-seed·하네스 가드).

### 4.2 가설 공간이 한 축에 편중됐다

45세대 중 P8~P39(30세대)가 "추론 경로 안에서 모달 가중을 적응적으로 정하기"의 변주였다(학습 게이트 → attn-bias → 추론 재가중 → 조건 라우팅 → 클래스 라우팅). 2026-07-08 시도 지도가 이미 "fusion 축 이탈 필요"(가정 ①)를 적시했고, 08-08 조건 oracle이 그 축의 상한이 +0.2임을 확정했다. 성능을 움직인 단일 변수는 전부 다른 축이다.

| 양성 단일 변수 | 크기 | 축 |
|---|---|---|
| SAM2 → DINOv3-L | +11.6 | 표현 |
| S+ → L 백본 | +8.82 | 표현 |
| 평가 해상도 768→1024 | +1.08~2.01 | 정보 |
| C3 prototype 손실 (DELIVER) | +1.35~1.74 | 학습 신호(클래스축) |
| PhysAug (공정선 밖) | +1.12 | 데이터 |
| P50 정렬 사전학습 | +0.74 | 정보·초기화 |
| P36 router | +0.10 | 배분 |

### 4.3 격차의 실체를 겨냥한 레버가 아직 단일 변수로 측정되지 않았다

- DELIVER SOTA(MM SAM-adapter)는 **RGB-D 2모달**, MUSES 1~4위는 **camera-only/2모달**이다. 우리 융합은 within-method로는 단조 이득이지만 상한 근처이고, 남은 격차는 RGB 표현·정보량 쪽이다(H5′·E-2 판정).
- frozen 백본 **스케일업**은 소진됐다(H12). 그러나 frozen 전제 자체를 깨는 **백본 부분 파인튠**은 2026-06-30 실패 분석과 07-02 P31 제안서가 "구조적 사망 클래스의 유일한 지렛대"로 지목한 이후, P31(SAM2, 3블록)과 P49(DINOv3, RGB_FT)에서 **다른 변경 4~5개와 묶여서만** 실행됐고 단독 기여는 한 번도 분리 측정되지 않았다. MM SAM-adapter 자체 표에서 frozen 55.35 vs fine-tuned 57.14의 차이(+1.8)는 현재 DELIVER best 격차(−2.06)와 같은 자릿수다.
- 07-08 시도 지도가 "제안됨·미실행"으로 1순위에 올린 functional-entropy 정규화(2505.06635)는 두 달이 지난 지금도 config조차 없다.
- 정보 축은 P50 한 번 성공 후 스케일업(Phase2)이 실패했는데 **실패 원인 분석 기록이 없다.** 유일한 양성 축의 실패 원인을 모르는 채 P52로 넘어가면, P52가 미달할 때 무엇을 고칠지 알 수 없다.

### 4.4 val→test 전이가 약해 선택 자체가 노이즈원이다

MUSES val 이득의 test 전이율은 약 4%(val +1.20 → test +0.046), DELIVER는 val-best 에폭 하나가 test를 ±2.7 흔든다. val↔test 순위 역전이 두 벤치에서 독립적으로 재현됐다. 따라서 "val-best 선택 → test 보고"라는 구조 자체가 매 런마다 ±1~2점의 추첨을 포함하며, 이것이 4.1의 노이즈를 키운다. final-iter 고정 + 3-seed 평균이 현재의 완화책이고, 근본책(test 분포와 상관 있는 선택 proxy)은 아직 없다.

### 4.5 세션 구조의 관성

시도 지도(07-08) §2 가정 ③이 지적한 "새 P 구현 → 학습 → 미달 → 다음 P" 루프는 8월 소거 캠페인으로 상당히 교정됐다(학습0 프로브, 사전등록, 매칭 A/B). 그러나 버전 번호가 안 올라가는 작업(사전학습 실패 원인 분석, 단일 변수 unfreeze, 데이터 축)이 여전히 뒷순위다.

---

## 5. 이 회고가 시사하는 다음 순서 (판단 — 실행 지시 아님)

1. **P52는 예정대로 완주하되, G1~G4가 3-seed·하네스 가드 통과 수치로만 판정되게 한다.** 노이즈 아래 판정을 반복하지 않는 것이 이 캠페인이 얻은 가장 비싼 교훈이다.
2. **백본 부분 파인튠을 단일 변수로 1회 측정한다** (현행 P46-C3 레시피 + LLRD 부분 unfreeze, seed 매칭, 다른 변경 0). 자체 문서가 2개월 넘게 "유일한 지렛대"로 지목했으나 분리 측정이 없는 유일한 대형 레버다. 이득이 +1.5 이상이면 DELIVER 격차의 절반이 설명된다.
3. **P50-EXT 실패 원인을 분석한다** (학습 0, 저장된 어댑터로 pseudo-모달 품질·정렬 지표 비교). 정보 축은 유일한 양성 축이며, 실패 원인을 모르는 채 두면 P52의 init 선택도 근거를 잃는다.
4. **RGB-D 2모달 fair-eval**(학습 0, current.md 미결 #3)로 SOTA와 동일 구성 비교표를 확보한다.
5. 포지셔닝은 이미 확보된 것으로 간다: DELIVER "DGFusion 상회·MM-SA 미달" 정직 공개, MUSES 융합계보 1위 + adverse robustness 인과, MCubeS 1위, 소거 증명 체인, dose-response 예측력.

---

## 6. 정정 사항 (이 회고 작성 중 발견)

- CLAUDE.md "M-score = 0.75×val + 0.25×test"는 낡은 표기. 정본([experiments/log.md](../experiments/log.md) 상단)은 (val+test)/2. CLAUDE.md 갱신 필요.
- [research/00_MOC.md](00_MOC.md)의 hypothesis-ledger 설명 "H1~H11"은 현재 H22까지 확장됨.
