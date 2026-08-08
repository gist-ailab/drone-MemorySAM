---
split_from: 01_project_status.md
created: 2026-07-08
period: 2026-07-01 ~ 2026-12-31
---

> **역할**: 2026 하반기(2026-07-01~) 역시간순 진행 로그. **새 진행 엔트리는 이 파일 최상단(이 안내 블록 바로 아래)에 추가**한다.
> 현재 상태는 [current.md](current.md), 2026-06-30 이전 이력은 [history-2026H1.md](history-2026H1.md) 참조.

## 역시간순 진행 로그 (History — 2026H2)

## 2026-08-09 — ProbeA2 백본 스케일링 프로브 완료 (opus 세션)

plan.md #9 실행. 코드(`tools/probe_backbone_scaling.py`, develop `6b33a2b`/`2511b61`)는 labcode 위임 후
fresh-eyes 검수(실제 pretrained 로드·음성 가드 2종·selftest 독립 재현) + jarvis 실전 스모크(4/4 PASS) 통과 후 기동.

**결과**: frozen DINOv3 + 공용 경량 head, MUSES RGB, val mIoU(best) S+ 59.85 / B 62.82 / L 68.67 / **H+ 69.19**.
- G-A2-상한 Δ(H+−L)=+0.52 → 중간대역(사실상 포화 근접, S+→L +8.82 대비 급격한 수확체감)
- G-A2-하한 Δ(L−S+)=+8.82(>3.0) → "방법 기여는 대형 백본 전제"로 논문 스코프 정직 공개 필요, 용량 정합 방어 불가
- 조건별: H+가 야간·악천후(fog/rain/snow_night) +3.7~4.7 vs clear_day −3.06 — 헤드라인 상쇄됨. 오늘 밤 다른 발견(drop-lidar day 0.64 vs fog_night 7.19~7.39)과 같은 축.

원장 H12/H12′ 신설. 상세 = [experiments/analysis/2026-08-09-probea2-backbone-scaling.md](../experiments/analysis/2026-08-09-probea2-backbone-scaling.md).
미결: 7B 추가 측정(hpca100 A100 필요, 24GB OOM 위험) — 코디네이터 판단 대기.

## 2026-08-08 — current.md 스냅샷 전면 재작성 + .claude_logs 정리 (discussion 세션)

- **current.md를 진짜 스냅샷으로 복원**: 07-15~08-06 날짜 엔트리 22개가 스냅샷 블록에 적층돼 있던 것을 본 파일로 이관(아래 아카이브 절). 연구 정체성 문구를 반증된 RBMA attn-bias 중심에서 "학습 전용 손실 + per-modal LoRA 트렁크" 축으로 개정. DELIVER 최고 수치 3중 모순(68.19/67.74/69.44 동시 유통) 해소 — 현행 legal 최고 = P46 C3-only 본run @1024 평가 val 69.44/test 56.99.
- **MOC 전면 재등록**: 미등록 문서 40건(experiments 30·decisions 8·det 1·meta 1) 등록, 00_INDEX 갱신. 감사 및 적용 = 서브에이전트.
- **신규 제안 등록**: [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md) — 조건×클래스 어댑터(CEA) 방향, oracle 조건-전문가 상한 프로브 선행(게이트 G-P1/G-P2 사전 등록).
- 남은 화석 문서(이번에 배너만 부착): meta/taskboard.md(07-03 이후 방치), research/ral-paper-plan.md(07-15 이후 방치), plan.md GPU 예약표(07-18 시점).

### 🗄 구 current.md 스냅샷 블록 원문 (2026-07-15~2026-08-06 적층분, 2026-08-08 이관 — 아래 07-27~08-06 결과 엔트리들은 이 이관 전까지 본 history에 누락돼 있었음; 07-15~07-25 엔트리는 기존 엔트리와 중복될 수 있음)

**연구 정체성**: 기여는 **RBMA (Reliability-Biased Memory Attention)** — SAM2/SAM3 memory cross-attention **logit에 training-free reliability를 additive bias로 가산**. canonical 정리 = [research/novelty-and-related-work.md](../research/novelty-and-related-work.md).

**🎯 공식 목표 (2026-07-03 사용자 설정 — 모든 수치는 이 기준과 비교해 보고)**: ① **Seg = 논문 publish** — DELIVER(all-modal) **val ≥66.51 / test ≥56.71**, MUSES SOTA **val 79.72 / test 79.49**, MULTIAQUA도 실행 예정. ② **Det = 국가연구개발과제 R&D** — **mAP50 0.85** (v2 split 기준). 세션별 액션 할당 = [meta/taskboard.md](../meta/taskboard.md).

**챌린지 최선 (MULTIAQUA, 고정)**: **P9 ep131 & P22 ep120 공동 1위, M-score 82.10** (Val 93.3 / Test 70.9). P10~P27의 adaptive fusion은 모두 gate 상수수렴 병목으로 P9 미돌파 → 이 진단이 RBMA 동기.

**📝 2026-07-15 RA-L 논문 트랙 개시**: NAS 볼트 `_paper_submission/`에 ReliaDINO(=P34/P36 계보) RA-L 초안 v1 전 섹션 작성+컴파일 완료(9p, `ReliaDINO_RAL_latest.pdf`). 타 세션이 채울 실험 슬롯 8개 = [research/ral-paper-plan.md](../research/ral-paper-plan.md). ⚠️ legal 최선 = P34 val 68.19/test 56.62(test-SOTA −0.09, "57.60"은 test-best라 철회) → P34 재중심화 리라이트 예정.

**📝 2026-07-18 P38 MaskQueryLite hpca100 4×A100 본학습 중 (ETA 07-19)**: P36 공정 레시피(GATE·VETO·CALIB·ROUTER on / ATTN_BIAS·CONSISTENCY·PHYSAUG off / DGFUSION_AUG on) 동결 위에 Mask2Former-lite query head(100 query, 6-layer masked cross-attn, 공유 cls/mask-embed head, β-zero-init로 시작 시 P36 byte-identical) 추가한 1-변수 비교. mask-classification 구조로 전환해 `panoptic_inference()` 경로 확보 = MUSES **PQ** 산출이 처음으로 가능해짐(기존 per-pixel head는 구조적으로 PQ 불가 — DGFusion/CAFuser는 OneFormer 스택이라 PQ가 주표). 커밋 3bb2c41(develop 병합 tip 6d922bd). hpca100 GPU 0-3(A100×4)에서 07-18 launch(config `configs/hpca100-deliver_rgbdel_P38_m2f.yaml`, develop @c3d1184, EPOCHS 200, ~0.77s/it·497it/ep) — 기동 검증 통과(iter 342→420/497 전진, 4GPU 25GB/83-100%, 에러 0, M2F ENABLE 확인). 실데이터 2ep 스모크는 yeon GPU0에서 병행 진행 중(참고용). 상세 [models/arch-evolution.md](../models/arch-evolution.md) P38 / 실행 현황 [experiments/plan.md](../experiments/plan.md). 판정 게이트 = P36 fair(val 67.74/test 55.62) 대비 + thin-class(Wall/Water/RailTrack) IoU. 🔴 **bengio seg-P37a/b는 사망 확정**(GPU5 HW 고장, 재부팅 후 SSH 미복귀) — jarvis 재기동분이 계보 승계.

**📝 2026-07-20 P39 Dual-Path Compete 구현 완료 (학습 대기)**: P38(MaskQueryLite) 게이트 미달을 이어받아, P30~P38 계보의 반복 실패 패턴(zero-init 잔차 사장·router 유일 실적·FUSED rank 병목·클래스축/도메인축 위치 불일치·event 기여의 데이터셋 종속성)을 실패-키로 역변환해 설계에 내장(근거 [decisions/2026-07-20-p39-dual-path-compete-proposal.md](../decisions/2026-07-20-p39-dual-path-compete-proposal.md)). 구조 = **V1 트렁크 rank 확장**(모달별 선형 투영 가산 합류, zero-init 아님) + **V2 modal-token query attention**(융합 병목 우회, det 폴백 유지) + **V3 anchored+free query**(K 클래스 고정 + 자유 Hungarian) + **V4 balanced point sampling**(클래스당 쿼터 256) + **V5 per-class Λ 중재 + path dropout 경쟁(dense-only 25% / query-only 25% / 결합 50%) + router 직접 CE(0.4)** — β 잔차 결선은 폐기. 전 항목 토글 가능(`p39_query_off`/`p39_trunkexp_off` 등 + config off). **단일 아키텍처로 DELIVER·MUSES 모두 커버**(user 지정). 합성 스모크 **PASS**(5지점 grad, 토글 유효, det 폴백, P38 호환 등가성 확인) — 실데이터 스모크는 미실행(yeon 배치 예정, 본학습 선행조건). config 3벌: `configs/hpca100-deliver_rgbdel_P39_dpc.yaml`(200ep)/`configs/jarvis-muses_rgbel_P39_dpc.yaml`/`configs/yeon-deliver_rgbdel_P39_dpc_smoke.yaml`(2ep). **판정 게이트(사전 등록)** = DELIVER: P36 fair(val 67.74/test 55.62) + thin-class 복원(Wall≥13/Water≥9.5/RailTrack≥62) · MUSES: P38 val 82.22 이상. 배치 = 대기열 1순위([experiments/plan.md](../experiments/plan.md)) — hpca100(P38-DELIVER 종료·판정 후 그 슬롯) / jarvis(P38-MUSES 완주 후). 상세 [models/arch-evolution.md](../models/arch-evolution.md) P39. 커밋 c31dcd5(develop).

**📝 2026-07-20 P39 학습 진행 + 분석 판정 (analysis 세션)**: 두 벤치 모두 학습 중이며 **모듈 기제는 성공, 성능 전환은 실패** 상태.
- **모듈(조기 즉검, [analysis/2026-07-20-p39-earlycheck-toggles.md](../experiments/analysis/2026-07-20-p39-earlycheck-toggles.md))**: V1 rank확장 off-Δ **+0.76~2.89(전 조건·양 벤치 최대 기여)**, V5 query경쟁 MUSES 전 조건 +(최대 +1.09), router 의존 **+22~40 → +0.4~2.3으로 해소**(직접감독 성공), arb λ 0.69→1.0~2.3 성장. **5세대 만의 첫 non-no-op** — 실패-키 처방(키1 경쟁결합·키2 직접감독·키3 rank확장)이 기제 수준에서 전부 유효.
- **DELIVER 4모달(img/depth/event/lidar) 3시점([analysis/2026-07-20-p39-deliver-3ckpt-compare.md](../experiments/analysis/2026-07-20-p39-deliver-3ckpt-compare.md))**: val **65.68@ep64로 P38 피크(65.19) 첫 돌파**했으나 **test 5-cond 평균은 최저 50.98**(ep60 51.96 > ep38 51.65 > ep64) — **val↔test 순위 역전**. 손실의 대부분이 **RailTrack 단독 −20.4**(cloud 59.2→6.4, 전 조건 진행형). 원인 = query·router가 동일 클래스를 상충 점유(ep38 night: query_off RailTrack −25.4, router_off −24.4) + gate/calib이 thin-class에 **유해**(ep60 night off 시 +35.9/+26.0 — 3세대 no-op에서 유해로 판정 변경). thin-class 게이트 **세 시점 모두 0/3 미달**.
- **MUSES 3모달(img/lidar/event) 공식 test**: P39-DPC ep146 = **78.881**(P38-m2f 79.025 −0.144, P34-3modal 78.979 −0.098) → 미돌파. 단 **주야 격차 5.14→3.73(−1.41)**로 개선, clear_night +3.69인데 **fog_night 62.68로 −12.05 붕괴**(전 제출 최저)가 전체를 상쇄. fog_night 정밀 대조분석(P39 ep146 vs P38 ep156, 조합 CASE 지원 커밋 dee524f) 진행 중.
- **🔴 physaug는 공정성 문제로 사용 배제(user 판정 07-20)** — P39.1 변수에서 제외하고 게이트는 아키텍처만으로 넘는다. 헤드라인 비교표는 physaug-off 계열(P35/P36 fair·P38·P39)로만 구성.
- **P39.1 후보(physaug 제외)**: D-1 V5 Λ 배타/온도 선택(query·router 상충 해소) · D-3 gate/calib 완전 off(유해 판정) · D-2 앵커 query 클래스 균형 손실 가중 · MUSES는 전 모달 동시 열화 시 V2 modal-src fallback.
- 모듈·제안영역 시각 리포트(fig 8장+판정표) = NAS `analysis_logs/module_report_20260720/`, 포인터 [analysis/2026-07-20-module-visual-report.md](../experiments/analysis/2026-07-20-module-visual-report.md). 실패-키 canonical = [analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](../experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md).

**📝 2026-07-21 P39.1(rank 수리) + P40(RCA) 구현 완료 (학습 대기)**: P39-MUSES 표준분석(lidar effective-rank 4.7 붕괴, adapter가 압축 주체)과 fog_night 붕괴(62.68) 원인규명을 관련연구 딥리서치 3편(rank collapse / modality imbalance / fog 물리)과 교차검증해 제안·구현. 근거 = [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md). 커밋 **ac5c7fe**(develop).
- **P39.1** = P39-DPC 위에 V1 트렁크 결합을 `fused += tanh(γ)·MLP_m(f_m)`(LN→1×1→GELU→1×1, γ init 0.1 — 0이면 gradient 완전 차단이라 절충)로 교체(R-1) + per-modal 토큰 VICReg var+cov(R-2, lidar×1.0/기타×0.25, λ 0.1/0.01, 2048 서브샘플) + M-2(gate/calib/veto config off, fog_night 유해 실증 반영) + eval마다 per-modal effective-rank 로그(`p391/rank_*`) 추가.
- **P40** = P39.1 위에 RCA(Reliability-Conditioned Attenuation) — C-1 lidar 리턴 유효성 신호(입력 유도) 가드/분석 + C-2 자기추정 rel(img) 배치 하위 분위(30%) 샘플 img feature soft 감쇠(α 0.1~0.5, hard-zero 금지, p_max 0.5, warmup 20ep, 학습 전용) + C-3 감쇠 샘플 한정 lidar readout 보조 CE(w 0.5, gradient 출구).
- config 5벌: `jarvis-muses_rgbel_{P39_1_rank,P40_rca}.yaml` / `hpca100-deliver_rgbdel_{P39_1_rank,P40_rca}.yaml` / `yeon-deliver_rgbdel_P40_rca_smoke.yaml`(아키 동일 = 단일 모델 제약 유지).
- 합성 스모크 **PASS**(RCA pick 발생, C-1 가드 동작, 손실 유한, grad 흐름, eval 결정론, linear 모드 하위호환).
- **판정 게이트(사전 등록)**: P39.1 ep30 = lidar effective-rank ≥15 & fog_night drop-lidar ≥4.0(미달 시 R-3: r16+rsLoRA로 재기동) · P40 = MUSES test ≥79.025 & fog_night ≥74, DELIVER = P36 fair + thin-class.
- **실행 순서**: 분석 선행 2건(fog val per-scene 감사 + P39 ckpt trunk_exp-off rank 재측정 — **분석 세션 몫, 학습 0**) → P39.1 투입(첫 빈 슬롯) → rank 게이트 통과 후 P40 투입. 대기열 [experiments/plan.md](../experiments/plan.md) #1(P39.1)/#2(P40). 상세 [models/arch-evolution.md](../models/arch-evolution.md) P39.1/P40.

**📝 2026-07-24 P43~P45 CVPR SOTA 제안 등재 (딥리서치 6축, 학습 대기)**: 멀티에이전트 딥리서치(모달불균형·상호증류·fog·panoptic·condition-adaptive·SOTA지형 6기 병렬) 교차 종합으로 3안 등재 — [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md). **전략 판정(Codabench 실측)**: MUSES mIoU 1위 = 미발표 카메라단독 GtA 82.39, 2위 = frozen-SAM 2모달 81.07 → **mIoU는 융합에 죽은 SOTA 축**; **PQ 1위 DGFusion 61.03은 사정권 + frozen-VFM 참가자 0 = 유일한 현실적 SOTA 축**(우리는 현재 PQ 산출 불가 = 구조적 배제 상태). ① **P43 PanopticDual(헤드라인)** = dual-head 공동학습(per-pixel 유지 + M2F Hungarian 독립 주손실, PMT 2603.25398 레시피 + multi-depth lateral) → PQ 확보, 착륙 지대 58~61, ep30 게이트 = PQ_thing>0 & thin-class −1pt 이내. ② **P44 BMR** = MMPareto gradient 통합 + peer 상호증류 + MCRM 국소 마스킹(전부 loss/gradient 레벨, P42 후계) → fog +2~5pt, ep30 게이트 = dMIoU(lidar)>1. ③ **P45 FogStyle** = FIFO식 fused-feature 스타일 불변(P44 위 토글). 대기열 #10/#11 등재. 선행(학습0) = P38 ckpt 기존 m2f_head로 val PQ 하한 실측.

**📝 2026-07-25 P43/P44/P45 구현 완료 + develop 병합 (학습 대기, hpca100 첫 슬롯)**: P43 PanopticDual + P44 BMR + P45 FogStyle 구현 완료, develop 병합 35ddbe0(+config 3e3b54f). 제안 문서 = [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md)(§7 토론 반영 포함). P43 = `semseg/models/reliadino/panoptic_head.py`(MaskClsHead, 100 query, Hungarian CE/BCE/Dice + PointRend, semantic mask-cls 모드) + encoder.py multi-depth lateral(blocks 5/11/17) + model.py 배선(`_encode_all`/`_apply_p43_lateral`/`panoptic_inference`/`semantic_from_queries`), 독립 주손실 `L = L_pixel + λ(t)·L_mask`(λ 0.1→1.0 warmup 5ep), configs {jarvis-muses,hpca100-muses,hpca100-deliver,yeon-deliver-smoke}_P43_pdual. 합성 스모크(`tools/smoke_p43.py`) 전건 PASS(독립성 assert 포함, off 시 baseline byte-identical). PQ 실측은 MUSES panoptic GT 부재로 TODO. P44 = `semseg/models/reliadino/mmpareto.py`(B-1 gradient 통합) + `p44.py`(B-2 mutual KL/relational correspondence, B-3 coverage-pattern 국소 마스킹, V-1 presence 재정규화, P45 fogstyle) + fusion.py/model.py/train_reliadino.py 배선, configs {jarvis-muses,hpca100-deliver}_P44_bmr + yeon smoke. 스모크(`tools/smoke_p44.py`) 86 assert PASS. 구현 기록 = [models/p44-bmr-implementation.md](../models/p44-bmr-implementation.md). 병합 후 P43+P44+P45 동시-on 통합 검증(forward/backward 유한·eval 결정론·panoptic_inference 동작) PASS. 학습0 검증 2건(`tools/analyze_router_coverage.py`, develop 7b053e0)이 hpca100 GPU2/3에서 실행 중 — 결과는 별도 세션이 회수 예정. 배치: P43-MUSES가 hpca100 GPU2,3 첫 슬롯([experiments/plan.md](../experiments/plan.md) ⚡ 절 기재).

**📝 2026-07-27 MUSES 공식 test 신기록 — P39.1-seed2 79.788**: `muses_P39_1_seed2_3modal_ep208_submission.zip`(val-best 82.62@ep208) 공식 test mIoU **79.788** — 구 최고 P38-m2f 79.025 대비 **+0.763, 새 MUSES test-best**로 대체. SOTA(GtA 82.39) 격차 **−3.37 → −2.60**로 축소. val→test 낙차 −2.83(82.62→79.79). per-condition: clear 79.300/fog 78.705/rain 79.063/snow 79.042, day 80.246/night 76.818(격차 −3.43), fog_night **69.610(전 조합 최악)**, snow_day 71.155<snow_night 77.413(**역전 3회째**). 약클래스(full): motorcycle 58.07·rider 59.47·pole 62.07·fence 65.70. 상세 [experiments/log.md](../experiments/log.md) §2026-07-27, 제출 인덱스 `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/muses/MUSES_TEST_RESULTS_INDEX.md`.

**📝 2026-07-27 P43-PanopticDual MUSES 표준분석 완료**: adapter·lidar eff-rank(23.5~28.0, VICReg OFF에도 건강, P39-DPC rank붕괴는 트렁크 자초로 P43 회피)·LATERAL(Δ +0.3~+1.9, feat_cos~0.75 no-op 아님)·router(Δ +4.7~+11.3) 정상 작동, 융합병목(per-modal 25~35 → FUSED_pf 5.5~11.3 급압축)만 잔존. **4모달 실증 근거**: drop-lidar dMIoU day/clear 0.64 → night 2.26·snow_night 2.73·rain_night 4.99·**fog_night 7.19**(비RGB의 adverse-night 인과 기여 확인, P39.1-DELIVER −0.78과 정반대) vs drop-event≈0~음수(잉여/사망, CKA(event~lidar) 0.79~0.85) → 4번째 모달은 event 대체가 아니라 **radar 추가**가 유력(radar-fix 재실험 최우선). 상세 [experiments/analysis/2026-07-27-p43-pdual-muses-standard-analysis.md](../experiments/analysis/2026-07-27-p43-pdual-muses-standard-analysis.md).

**📝 2026-07-27 P39.1-seed2(우리 최고, val 82.62/test 79.788) MUSES 표준분석 완료**: VICReg(R-2)가 lidar eff-rank를 78.5~100.3까지 확장(P43 VICReg-off 23.5~28.0의 3~4배) 실증, trunk(R-1, p39_trunkexp_off) 전조건 +2.05~+6.78 순기여, router +0.5~+4.5 순기여, drop-lidar 야간·adverse(fog_night 7.39/snow_night 7.6/rain_night 7.57) 인과 기여 확인 — P39.1의 R-1·R-2 기제 모두 검증됨. 흠: arbiter query가 일부 야간 조건(rain −0.26/night −0.37/clear_night −0.29)에서 미세 유해. 4-modal(+radar) 착수 근거로 연결(drop-lidar 야간 기여 → radar가 fog에서 lidar 산란 보완 기대). 상세 [experiments/analysis/2026-07-27-seed2-p39_1-muses-standard-analysis.md](../experiments/analysis/2026-07-27-seed2-p39_1-muses-standard-analysis.md).

**📝 2026-07-28 P44-BMR MUSES — hpca100 외부 preempt 사망 → 재개 성공**: hpca100 외부 preempt 사망(07-28, best val 80.59@ep126 미완주) → **hpca100 preempt 후 재개 성공(07-28, ep150부터, HF offline fix)**. BMR 기제는 DELIVER P44-BMR(jarvis 진행중)로 계속 검증.

**📝 2026-07-28 P39.1-rank 5-seed variance 완결**: seed3 완주(81.89@ep204, Total 20:47:50)로 5-seed 전원 완주 — seed1 82.03/seed2 82.62/seed3 81.89/seed4 81.92/seed5 81.70(범위 81.70~82.62, 평균 82.03, 논문 variance 보고용).

**📝 2026-07-28 P44-BMR MUSES 표준분석 완료**: val 80.71@ep156 완주분 표준분석 결과 — **BMR이 비RGB(lidar) 사용을 P39.1/seed2 대비 늘리지 못함**(drop-lidar day −0.42, seed2의 day 4.24보다 낮음). val 이득도 없음(80.71 < seed2 82.62, −1.91). 유일한 특징은 lidar 사용의 야간 편중(fog_night 6.71 vs day −0.42) — test(adverse-night 비중 높음) 전이 여부는 남아있음. DELIVER에서도 P44-BMR(66.31)이 P39.1-rank(67.60) 대비 우위 없음(정체 지속). 상세 [experiments/analysis/2026-07-28-p44-bmr-muses-standard-analysis.md](../experiments/analysis/2026-07-28-p44-bmr-muses-standard-analysis.md).

**📝 2026-07-28 P44-BMR MUSES 공식 test — 78.429, BMR 방향 종료**: `muses_P44_bmr_3modal_ep156_submission.zip` 제출 결과 **공식 test 78.429** — seed2(79.788) 대비 **−1.36**, P38(79.025)·P34(78.979)보다도 낮음. SOTA(82.39) 격차 −3.96. 🔴 **fog_night 56.443 = seed2(69.61) 대비 −13.2pt 파국** — "야간편중 lidar 사용이 유리"라는 BMR 가설이 test에서 완전 반증(오히려 야간·fog에서 seed2보다 나쁨). snow_day(68.60)<snow_night(72.07) 역전 재현. **BMR 방향 종료(val·test 모두 P39.1 열세)** — 우리 test 최고는 seed2 79.788 그대로. 상세 [experiments/log.md](../experiments/log.md) §2026-07-28.

**📝 2026-07-28 P43-PanopticDual MUSES 공식 test — 79.351(우리 2위)**: `muses_P43_pdual_3modal_ep156_submission.zip`(val 82.51) 제출 결과 **공식 test 79.351** — seed2(79.788) 대비 −0.44로 **2위**, 단 P38(79.025)·P34(78.979)·P44(78.429)보다는 높음. SOTA(82.39) 격차 −3.04. day 80.81(seed2 80.25보다 우세)·night 75.19(seed2 76.82보다 열세)·fog_night 67.76(seed2 69.61보다 열세) — PQ 헤드 병행학습이라는 다른 기제인데도 P39.1 계열 수준의 성능, MUSES 최고는 여전히 **seed2 79.788**. 상세 [experiments/log.md](../experiments/log.md) §2026-07-28.

**📝 2026-07-30 P46-CTR DELIVER — RailTrack 게이트 통과(C1+C3 ep40, class-transfer 가설 확증)**: c1c3(C1_RCS+C3_PROTO, C2_MCC off) ep40 체크포인트(val 67.36)의 test@768 per-class eval에서 **RailTrack test 4.02(base)→59.10(+55.1)** — 사전등록 primary falsifiable 게이트(≥40) 압도적 통과, DGFusion(64.47)에 근접. Wall/Water/Bridge는 게이트 제외(DGFusion도 test IoU 0~4로 동반붕괴 확인됨, §9) 그대로 저조(10.84/10.96/0.02). Overall test는 52.47→54.92(+2.45)로 개선되었으나 secondary gate(56.62)·DGFusion(56.71) 미달 — **RailTrack 회복이 overall 돌파로 직결되진 않음**(다른 붕괴 클래스가 천장). val에서는 RailTrack 18.53으로 test보다 낮은 역전 현상 관찰(해석 보류). ep40은 중간 체크포인트(학습은 계속 진행 중, ep200 완주 후 재판정 예정) — C1 RCS의 단독 기여를 분리하는 C3-only ablation(jarvis GPU4-7)도 병행 중이며 ep40 도달 시 동일 gate eval 예정. 상세 [experiments/analysis/2026-07-30-p46-ctr-c1c3-railtrack-gate.md](../experiments/analysis/2026-07-30-p46-ctr-c1c3-railtrack-gate.md).

**📝 2026-08-06 P46-CTR C3-only 완주 — fair-eval 최종(SOTA 미돌파, DGFusion 상회 유지)**: C1 유해 판정 이후 C3-only 단독(본+seed2)이 완주, val-best ckpt만으로 fair-eval 확정 — **legal 최고 = 본(original) val-best@ep70, @1024 평가 val 69.44 / test 56.99**(RailTrack 67.69, base 4.02 대비 압도적 회복). **현행 DELIVER SOTA(MM SAM-adapter val 69.60/test 57.35) 대비 val -0.16/test -0.36 미돌파**, 단 **구 DGFusion 기준(66.51/56.71)은 val·test 동시 상회(+2.93/+0.28)로 no-tradeoff 우위** 유지. seed2 재현성 확인(같은 방향, test -0.59 단일런 편차). 이전 ep40 중간ckpt의 "test-SOTA 예비 도달" 판정은 **철회**. 상세 [experiments/analysis/2026-08-06-p46-c3only-fair-eval-final.md](../experiments/analysis/2026-08-06-p46-c3only-fair-eval-final.md).

**🏆→⚠️ 2026-08-03 P46 C3-only λ0.2 DELIVER 완주 — SOTA 미달(정정 2026-08-04)**: jarvis GPU4-7, 200/200 완주(Total 07:58:28). 원 보고 "test 57.05@ep108 = DELIVER test SOTA 돌파"는 **오보로 철회** — 57.05는 test-best 체크포인트 값으로 규약(val-best 또는 final-iter만 legal)상 무효. **legal 재계산: val-best test 54.60@ep118 / final-iter(ep200) test 55.69**, val 67.47@ep118. λ 스윕(0.05/0.1/0.15/0.2/0.3) 중 legal 최고는 λ0.05(55.62 val-best/55.69 final-iter) ≈ λ0.2와 평탄, λ0.3에서 악화(54.52/55.04) — **legal 최고 test ≈ 55.7, DGFusion SOTA 56.71 대비 −1.0(미달)**. 단 base(P39.1-rank) 대비 P46-3의 실제 이득은 견고: test **+1.35(val-best)/+1.74(final-iter)**, val **+0.97**. val은 λ0.05가 최고(68.57) — **val·test가 서로 다른 λ를 선호**. 미해결: RailTrack val<test 역전, DGFusion final-iter 프로토콜 차이. 상세 [experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md](../experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md).

**📝 2026-08-04 MUSES 4모달 seed2(P39.1-rank ep260) 공식 test 수신 — 79.571, radar 무익 재확정**: `muses_P39_1_rank_4modal_seed2_ep260_submission.zip`(val 82.35) Codabench 공식 test **79.571** — 3모달 seed2(79.788) 대비 **−0.217**. drop-radar ablation(dMIoU +0.13)·3-seed race 실패(전원 82.62 미돌파)에 이어 공식 test에서도 radar가 이득 없음이 재확인됨. 조건별 clear 78.926/fog 78.432/rain 78.671/snow 78.537(spread 0.49), day 80.435/night 76.442(gap 3.99), fog_night 64.238(전 조합 최악). 상세 [experiments/registry.md](../experiments/registry.md) `hpca100_muses_rgbelr_P39_1_rank_4modal_seed2` 행, 조건별/per-class 전체 `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/muses/MUSES_TEST_RESULTS_INDEX.md` 제출 8.

**🛠 2026-08-04 P47-2 UniBal(Uni-modal Balance, 구 D-2) 구현 완료 — 학습 대기**: 제안 [decisions/2026-08-03-p47-mub-muses-proposal.md](../decisions/2026-08-03-p47-mub-muses-proposal.md) §3 D-2의 코드화. Base = **P39.1-rank MUSES 4모달 seed2 동결**(val 82.35). 각 모달 encoder(frozen ViT+LoRA) 출력에 **모달별 독립** 경량 head(GroupNorm→1×1)를 달고 동일 GT CE를 주손실에 직접 합산(키1) — 진단은 modality laziness(리더보드 모달↑=순위↓, 우리 4모달 82.35 < 3모달 82.62; P46-C3 손해가 clear/day 집중 = RGB 본류 병목). 신규 `semseg/models/reliadino/p47.py` + `tools/smoke_p47.py` + `configs/hpca100-muses_rgbelr_P47_2_unibal_4modal.yaml`. **추론 불변**(eval |Δ|max=0)·**추가 forward 없음**(feats 재사용)·**DELIVER 무영향**(P47_2 키 부재 시 완전 동일, 해당 P46 C3-only λ0.2 run 경로 무변경 — 그 run의 legal 최고 test는 55.7, 정정 2026-08-04: 57.05는 test-best로 무효). 메모리 실측 +51.7 MiB/스텝(BS1·4모달·1024²·bf16). 스모크 `--ddp` 포함 전항목 PASS, **실데이터 미기동**. ⚠️ 구현 중 확인: base에 이미 per-modal aux CE(`FUSION.AUX_CE_WEIGHT`)가 있어 P47-2는 그것과 **head 목적 분리 + 모달별 가중 + OGM-GE 결선**으로 차별화했다 — 이 판단의 1차 확인점이 **ep30 즉검(`[P47-2] per-modal acc`의 모달별 분화)**이다. 선택 토글 OGM-GE(2203.15332)는 구현·검증했으나 기본 off. **D-1(투영 밀도화)은 미포함**(단독 변수). 다음 = fresh-eyes 검수 → develop 병합 → A100급 4장 확보 시 기동. 상세 [models/arch-evolution.md](../models/arch-evolution.md) §P47-2.

**🛠 2026-08-04 PQ 평가 경로 배선 완료 — 이제 우리 최고 계보(M2F)에서 PQ 산출 가능, 남은 블로커는 GT 다운로드뿐**: 지금까지 `model.panoptic_inference()`는 **`self.p43`으로만 라우팅**돼, P43 off / `MODEL.M2F.ENABLE: true`인 실전 P39.1-rank 계보는 **PQ를 낼 수 없었다**. 활성 헤드 분기(p43 → 기존 경로 무변경 / m2f → `MaskQueryLiteHead.panoptic_inference` / 둘 다 없으면 RuntimeError)를 넣고, MUSES **레터박스 역변환**(마스크 **로짓** 단계에서 stride4→1024²→pad crop→native 1080×1920, 그 다음 sigmoid/argmax — 이진화 후 리샘플 금지)과 **AUPQ 포맷 writer**(COCO-panoptic json + `rgb2id` PNG + Cityscapes **labelIds**)를 붙였다. 신규 `tools/eval_pq.py`·`tools/pq_format.py`·`tools/smoke_pq.py`(73 check PASS, 공식 AUPQ 스크립트와 수치 일치 실측). **재학습 불필요**(기존 ckpt query 사용), semantic 추론 `|Δ|max=0`, P43 경로 무변경(`tools/smoke_p43.py` 전건 PASS). 🔴 **남은 블로커는 코드가 아니라 데이터** — MUSES `gt_panoptic`/`gt_uncertainty` 미다운로드([experiments/plan.md](../experiments/plan.md) #4)라 실측 PQ는 GT 확보 후. test는 GT 비공개라 도구가 `--split test`를 거부한다. 상세 [status/history-2026H2.md](history-2026H2.md) 2026-08-04.

**⚠️ 2026-08-05 P46 C3-only λ0.2 seed2 DELIVER 완주 — 재현성 검증도 실패**: jarvis GPU1,2,3, 200/200 완주(04:43, Total 18:09:40). **val-best test 55.55@ep62**(val 67.74) / **final-iter(ep200) test 55.31**(val 65.71) — legal 두 프로토콜 모두 **내부최고 DELIVER test 56.62(P34/P36 fair)에 미달**(val-best −1.07, final-iter −1.31). best test 56.30@ep146은 **test-peeking이라 사용 불가**(참고 기록만). 원본 lam02 런(legal val-best 54.60/final-iter 55.69, 08-03 정정 참조)과 같은 미달 대역으로 재현 — **λ0.2 SOTA 돌파는 재현성 검증에서 최종 반증**. 추가 관찰: val이 ep62 67.74 → ep200 65.71로 **−2.03 하락**(138 epoch 추가 학습이 오히려 val 악화) → **λ0.2에 EPOCHS200은 과함**, 후속 λ 실험은 EPOCHS 재검토 필요. **DELIVER 현재 최고는 여전히 P34/P36 fair val 67.74/test 56.62 — 변동 없음.** 상세 [experiments/registry.md](../experiments/registry.md) `jarvis_deliver_rgbdel_P46_ctr_c3only_lam02_seed2` 행.

**⚡ 2026-07-08 최신 (아래 표는 07-02 시점, P30~P31 시대의 기록임)**

| 트랙 | 상태 | 수치 / 다음 액션 |
|------|------|------------------|
| **P32 (CoRB) seg** | 🏁 **학습 완료 + 4축 독립 검증 완료** | 최종 **Day-Val 64.12@ep98(계보 최고) / Test 55.00**(P31 54.85 +0.15, P28 55.27에 −0.27 미달; 목표 갭 val −2.39/test −1.71). 검증 결론: CoRB attn-bias는 **유의한 순손해**(ΔmIoU −0.013, p=4.5e-22) — 신호는 유효, pre-softmax 주입은 무효. 지배 원인 = per-class 전이 붕괴(복구 상한 +7.9pt). 상세 [experiments/analysis/p32-verification-p33v2.md](../experiments/analysis/p32-verification-p33v2.md) + 볼트 `research/vault/P32_CoRB/P32_정량검증_실패분석_20260708.md` |
| **P33-v2 (CG-MoD 개정)** | ✅ **설계 완료 (구현 대기)** | 원안(doc 26) 적대적 비판 + 딥리서치 3축 반영: M0 무학습 진단 3종 → M1 class-transfer 복구(RCS+text-anchor+MIC consistency, night+**sun**) → M2 dropout+distillation → M3 soft gate(corr_veto) → M4 CoRB 제거. 기대 test 56.5~58. Global escape: val<65.5 → 카드 A(DINOv3-RBMA) 전환. 볼트 `research/vault/P33_CGMoD/P33_v2_설계개정_20260708.md` |
| **Det (국책과제)** | 🎯 **목표 달성** | egofill 데이터(2.01×)만으로 **mAP50 0.8501**@ep9 (official v2 test). 남은 서사 = 저조도 robustness delta. 상세 doc 19 E2.5 |
| **옵시디언↔repo 동기화** | ✅ 규약 제정 | NAS 볼트 canonical, `scripts/sync_research_vault.sh`(NAS→repo pull), 실험폴더 패턴 `P<N>_<이름>/`. `research/vault/README.md` §🔄 |

**⚡ 2026-07-15 15:50 갱신 — seg 현재 트랙 (최신). B200 학습 전부 완주, 도는 프로세스 0.**

> 🔴 **보고 기준 정정(07-15)**: 트레이너는 `epochNN_<val>_topK`(val-best)와 `test_epochNN_<test>_topK`(**test-best**) 두 계열을 저장한다. **test-best 인용 = test셋 훔쳐보기라 논문 불가.** 소유자가 P35 config에 이미 명시: *"ckpt 선정: val-best만 보고(합법)"*. 07-12~14의 "P34 test-SOTA 57.60 돌파" 등 보고는 **전부 test-best 기반이라 철회**. 아래는 **legal(val-best) 실측**.

| 모델 | val-best | 그 에폭 test | vs val-SOTA 68.6 | vs test-SOTA 56.71 | 목표(66.51/56.71) |
|---|---|---|---|---|---|
| **P34 ReliaDINO (최선)** | **68.19** @ep120 | **56.62** | **−0.41** | **−0.09** | val ✅ / test ✗ |
| P35 paper | 67.61 @ep78 | 55.52 | −0.99 | −1.19 | val ✅ / test ✗ |
| P36 router | 67.74 @ep52 | 55.62 | −0.86 | −1.09 | val ✅ / test ✗ |

> 🔴 **어떤 모델도 test-SOTA 미돌파.** 최선 = **P34: val 68.19 / test 56.62 (test −0.09로 아깝게 미달)**.

**모델 구성(config diff 실측)**: `P35 = P34 − ATTN_BIAS(RBMA) − CONSISTENCY − PhysAug`(DGFusion 공정 레시피) · `P36 = P35 + Per-Class Reliability-Anchored Router`(P31 포트).
- **P34 vs P36 직접 비교는 부당**(P34만 PhysAug on). **정당한 짝 = P35 vs P36 → 라우터 val +0.13 / test +0.10 근소 우위**(노이즈 수준). 지난 "라우터 이식 실패" 판정 철회.
- **노벨티**: 새 메커니즘은 **P36 > P34**(router는 P36만 보유). P34가 더 가진 ATTN_BIAS·CONSISTENCY는 소유자 G0c ablation에서 **효과 ≈0**(baseline 68.20/56.64 vs strip-full 68.45/56.38; **gate/calib만 test +0.26 실기여**). → **간판 노벨티 RBMA attn-bias가 DINOv3 계보에선 무력**, P34 수치 우위는 대체로 PhysAug(증강) 덕. 논문 서사 재정비 필요.

| 트랙 | 상태 | 비고 |
|---|---|---|
| **P36_router** | 🏁 **완주**(ep200/200, 07-15 11:14 KST) | best ep52/58 이후 148/142ep 미갱신, val 끝까지 61.45 열화. per-class 붕괴: Bridge 0.06·Other 4.35·Ground 4.83·Wall 5.67·Dynamic 6.31·Water 10.10 (주력 Road/Sky/Cars/Bus/Truck은 90+ 정상). ckpt 백업 완료. |
| **MUSES × P34-ReliaDINO** | 🏁 **완주 + 공식 재평가 완료** | **공식 val mIoU 80.86**@ep276(내부 81.02 −0.16, thin class 집중). 프로토콜=CAFuser `MUSESSemSegEvaluator`(=stock detectron2, argmax 전 native 업샘플·GT 무리사이즈) 소스 확정. **DAY 83.56 / NIGHT 82.03(−1.53만) — 악조건 robustness 강함**(공통 11클래스 통제; naive는 조건별 클래스수 상이로 오해 유발). 🔴 **SOTA 주장 불가**: **79.72=DGFusion val(CAFuser 아님; CAFuser 78.71/CAA 79.04)**, **MUSES는 test로 랭킹**(DGFusion test 79.49)인데 우리는 test 없음 + 백본 10×(ViT-L 300M vs Swin-T 28M) + val-selected ckpt. **결론 = Codabench 14005 test 제출**(hinton 가능, 계정 필요); *방법* 주장하려면 Swin-T 동급 재학습. 회수 `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/MUSES_P34_20260715/`(ckpt 1.7G + official_eval/ raw confusion 포함). loader+config develop 병합(b4d69c1). |
| **🔴 B200 마감** | **2026-07-15 23:59 KST** (잔여 ~8h) | 학습 전부 완주·회수 완료. 백업: `B200_backup_20260715/`(8.7G) + `P34_final_20260713/` + `MUSES_P34_20260715/`(1.7G). 구세대 가중치 ~400GB는 의도적 미백업(로그·config만). |

**진행 중 트랙 (2026-07-02 시점 기록 — 위 표가 최신)**

| 트랙 | 상태 | 최신 수치 / 다음 액션 |
|------|------|----------------------|
| **seg: P34 ReliaDINO (B200 DELIVER)** | 🏁🏆 **완주**(07-13 15:34, DINOv3 ViT-L/16 frozen+RBMA) — 최종 **Val 68.19@ep120 / Test 57.60@ep140**. **Test-SOTA(DGFusion 56.71) +0.89 돌파**(경쟁 지표 승리, 계보 최초) / Val 목표 66.51 달성(val-SOTA 68.6엔 −0.41). **P34=확정 최선 seg.** best ckpt NAS 회수·검증 완료(/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/P34_final_20260713). 모니터 RUN-20. |
| **SAM2 RBMA seg (P29, B200)** | ⏹ **종료**(ep150, 2026-06-30 11:03; P30 띄우려 수동 중단) | 최종 best **Val 63.20@ep100 / Test 54.34@ep146**(ckpt 보존). val 70 미달·ep34부터 60~63 정체. 모니터 RUN-2 |
| **SAM2 RBMA seg (P28, B200)** | 🔴 사망(ep16, 2026-06-24) → **P29로 대체됨** | best Val 57.87@ep12 / Test 50.61@ep12. `last_checkpoint.pth` 보존. 모니터 RUN-1 |
| **Det 객체검출** | 🎯 **best=bengio det_P29_egofill mAP50 0.8501** / event ablation 완주·final_full 진행 | **egofill(RUN-11) 🏁완주**: best **mAP50 0.8501@ep9**(공식 v2 test) — **목표 0.85 달성**(lidar egofill+2×데이터). **det_P29_event(RUN-14) 🏁완주**(07-07): best mAP50 **0.8427**@ep14 → event≈egofill-lidar(−0.008, 모달 ablation 유의미). **det_P29_final_full(RUN-15) 🟢학습중**(07-08~): P29+egofill을 최종 annotation(_final_ann/instances_train_egofill.json)으로 재학습, EPOCHS50 ep0. det_P31_v3clip(RUN-10) 완료: mAP50 0.4724(v3clip=비공식). P30-Det 0.256. 모니터 RUN-10/11/14/15 |
| **SAM3 RBMA (포팅)** | **학습/디버깅 중** (DELIVER 25cls) | ✅6/21 decoder repurpose로 class-collapse 돌파: val 8.49→**16.27@ep22 (상승 중)**. 다음=ep40~60+ 상한 확인 |
| **P29 (SDC 조건 라우팅)** | **설계 완료 (구현 대기)** | Soft-MoE LoRA 라우팅 비특화 진단 → label-free image-derived 조건 latent+prototype→FiLM gate(헤드라인), RBMA 신뢰도를 라우팅으로 확장(P29-B). 상세 [models/arch-evolution.md](../models/arch-evolution.md) P29 / 노벨티 [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.7 |
| **P30 (class-token decoder + reliability-anchored router)** | **구현 완료 (학습 대기, P28 종료 후 GPU 2,3)** | P28 실패분석(rare-class collapse: Water/Bridge=0; event/LiDAR 미사용 Δ≈0) 직격 → ① class-token decoder(SAM3-RBMA class-collapse break 이식, m_feat에 class query cross-attn) ② reliability-anchored 학습 modality router(상수수렴 방지, per-class). 두 모듈 CPU smoke PASS, 모델 wiring compile-only. config `b200-deliver_rgbdel_P30_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P30 / 노벨티 [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.8 |
| **P31 (Calibrated Dual-Reliability RBMA + MS-HR class-token decoder)** | **구현+P31.1 수정 완료 (B200 자동 launch 대기)** — 2026-07-03, develop 반영 | doc 20 P31-Seg core 우선순위 ①② 구현: [Seg-A] per-modal temperature + correctness-contrastive **calibration loss**(event/LiDAR AUROC .30/.22 수리) / [Seg-C] `ClassTokenDecoderMS`(simple-FPN {4,8,16,32} + 학습형 ConvTranspose HR pixel-embed + training-only aux CE @H/4) / [레버①] Hiera 마지막 3 block unfreeze(LR×0.1) / [레버②] **router 'decisive' reg**(uniform 라우팅 해소) / [Seg-B] consistency bias·rel-AMF는 기본 OFF(AUROC>0.5 조건부). **P31.1 (비판 리뷰 `/mnt/HDD2/src/logs/P31_review_20260702/` 검증 반영)**: P30-seg 실측 붕괴(Val 49.76@ep136/Test 44.10@ep146 = P29 대비 −13.4/−10.2) 확인 → **CTD aux-only 강등**(최종 출력=SAM decoder 복원, CTD는 training-only aux CE) + **per-modal reliability AUROC/router-w 학습 중 로깅**(tb/wandb `p31/*`) + **SDC OFF**. B200 watcher가 P30(ep~194/200) 종료 시 GPU 4-7에 자동 launch. config `b200-deliver_rgbdel_P31_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P31 |
| **P32 (CoRB — Corroboration-Biased Memory Attention)** | 🟡 **학습 plateau + 분석 완료**(2026-07-06, branch `worktree-p32-corrb` 미병합) | RBMA 신뢰도 신호를 self-entropy → **무학습 cross-modal corroboration(corr_veto)** 으로 교체(`LoRA_Sam_P32._compute_bias_source`, λ만 학습). **Phase 0 게이트 PASS**(corroboration이 event/LiDAR AUROC .30/.22→.54/.81 반전, [experiments/analysis/p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md)). **학습 결과 미달**: Test **53.45@ep40** / Val **61.65@ep30** (P28 55.27/63.40·P31 54.75/63.20 대비 −1.8/−1.8, ep26~30 plateau). **분석 판정(핵심)**: "신호는 맞고 라우팅 실패" — corroboration AUROC는 좋으나 **drop-modality Δ[img6.2,depth15.6,event~0,lidar~0] = event/LiDAR 여전히 죽음(Mode C)**. soft attention-bias로는 feature/decoder가 약한 모달(competence≈0) 부활 불가, 오히려 P28 self-entropy(약모달 down-weight)보다 소폭 악화. 구조적 사망 class(Bridge/Water/Wall/Other IoU~0)=frozen-backbone ceiling. 산출물 `/mnt/HDD2/src/logs/P32_eval_20260706/`. **처방**: ① **P32-C(PruneMem: hard pruning+modality dropout)** 로 event/LiDAR 강제 사용(다음 단계) ② calibration 복원+corroboration 결합 ③ 구조적 사망=backbone unfreeze/CTD. config `b200-deliver_rgbdel_P32_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P32 / [experiments/analysis/p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md) |

**📦 2026-07-28 리포 통합 + 재현 경로**: worktree 브랜치를 **develop 하나로 정리**했다(worktree 15→3, 로컬 브랜치 22→5). 삭제 브랜치의 원본 커밋은 전부 **`archive/<브랜치>` 태그 11개**로 보존 — 옛 브랜치를 찾으면 `git tag -l 'archive/*'`를 보라. **`26-drone-certificate`는 통합 대상이 아니며 유지**. 사용 중인 worktree 2개(`p34-det`·`p30-det`)는 다른 세션이 점유 중이라 보존했다. 폐기 브랜치에만 있던 고유 자산(**MUSES 공식 native-해상도 재채점기 `tools/eval_muses_official.py`** 등 도구 7종·config 5종·연구문서 6종)은 develop 택소노미로 회수했다. **정량 재현 경로 신설** = [`REPRODUCE.md`](../../REPRODUCE.md) + `bash scripts/reproduce_eval.sh <deliver|muses|muses-official|multiaqua|det>`. 상세·검증 내역 = [history-2026H2.md](history-2026H2.md) 2026-07-28 엔트리. ⚠️ 아래 P32 행의 "branch `worktree-p32-corrb` 미병합" 표기는 **옛 기록** — 해당 브랜치는 develop에 포함돼 정리됐다.

**열린 블로커**
- SAM3 ViT single-scale 한계 → SAM2 P28(val~55) 대비 격차 규명 필요.
- SAM3 최소 클래스(Pedestrian/Pole/sign/Dynamic/Water) 여전히 0 → `decoder_high_res`(FPN skip) 후속 실험 후보.
- P28 multiaqua B200 config 경로 검증.

**다음 마일스톤**: ① SAM3-RBMA 수렴 곡선 확보 → ② SAM2 P28 B200 학습 → ③ RBMA ablation(SoftMoE LoRA / SQG / AMF 제거 robustness).

---

### 2026-08-05 — P46 C3-only λ0.2 seed2 DELIVER 완주: 재현성 검증 실패, EPOCHS200 과다 발견

- **실험**: `jarvis-deliver_rgbdel_P46_ctr_c3only_lam02_seed2`(DELIVER 4모달 img/depth/event/lidar), jarvis GPU1,2,3. **04:43 완주, 200/200, Total 18:09:40**.
- **결과(legal)**: val-best **ep62** → val 67.74 / test **55.55**. final-iter **ep200** → val 65.71 / test **55.31**. best test 56.30@ep146은 **test-peeking이라 사용 불가**(참고 기록만).
- **판정**: **실패**. 두 legal 프로토콜 모두 내부최고 DELIVER test 56.62(P34/P36 fair) 미달(val-best −1.07, final-iter −1.31) — 원본 lam02 런(legal val-best 54.60/final-iter 55.69, 08-03 정정 — `experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md`)과 같은 미달 대역으로 재현되어 **λ0.2 SOTA 돌파는 재현성 검증에서 최종 반증됨**.
- **부수 발견**: val이 ep62 67.74 → ep200 65.71로 **−2.03 하락** — 138 epoch 추가 학습이 오히려 val을 악화시킴 → **λ0.2에 EPOCHS 200은 과함**, 후속 λ 실험 EPOCHS 재검토 필요.
- **DELIVER 최고는 여전히 P34/P36 fair val 67.74/test 56.62, 변동 없음.**

### 2026-08-04 — PQ 평가 경로 배선 (M2F 라우팅 + tools/eval_pq.py, 학습 무관)

- **범위**: 코드 + 스모크까지. **학습 미기동**(지시), push 없음(리뷰 후 develop 병합). 재학습 불필요 — 기존 ckpt의 query를 그대로 쓴다.
- **고친 문제**: 우리 최고 계보(P39.1-rank)는 `MODEL.M2F.ENABLE: true`로 `MaskQueryLiteHead`를 쓰는데, `model.panoptic_inference()`가 **`self.p43`으로만 라우팅**돼 있었다. 실전 config는 P43 off / M2F on이므로 **현 코드로는 우리 모델의 PQ를 낼 수 없었다**(P43 계열 ckpt로만 가능).
- **신규/변경**: `model.py`(활성 헤드 분기 — p43 → 기존 경로 그대로, m2f → 신규 `_m2f_forward_out`+헤드 호출, 둘 다 없으면 RuntimeError; `_resolve_thing_ids`; Cityscapes thing trainId 상수) · `m2f_head.py`(`panoptic_inference`에 `size`/`crop`/`crop_size` 추가) · `tools/pq_format.py`(신규 — AUPQ 포맷 writer + 표준 PQ 스코어러) · `tools/eval_pq.py`(신규 러너) · `tools/smoke_pq.py`(신규, 73 check PASS).
- **AUPQ 입력 형식(코드에서 확인, 추측 아님)**: COCO-panoptic json 2개 + `rgb2id` RGB PNG 폴더. category_id는 **Cityscapes labelIds**(스크립트 상수 `STUFF=(7,8,11,12,13,17,19,20,21,22,23)` / `THINGS=(24,...,33)` = trainIds 0~10 / 11~18 — **thing_ids 규약의 1차 근거**). AUPQ 전용 추가 입력 = GT `gt_uncertainty` 폴더 + pred `classConfidence`/`instanceConfidence`(pred 폴더 이름이 반드시 `labelIds`여야 문자열 치환이 성립). **confidence를 255로 포화시키면 n² 임계 셀이 전부 동일해져 AUPQ ≡ PQ** — 스모크가 공식 스크립트와 우리 스코어러의 일치를 실측(96.9 == 96.9)한다.
- **letterbox 정합**: MUSES val은 1080×1920 → 1920² 레터박스(ignore=255) → 1024² 리사이즈인데 **panoptic GT는 native 1080×1920**이다. `--geometry native`(기본)는 `tools/eval_muses_official.letterbox_valid_box`(왕복 증명 있음)로 **마스크 로짓 단계에서** stride4→1024²→패드 crop→1080×1920 순서로 역변환한 뒤 sigmoid/argmax/0.5 임계를 적용한다(이진화 후 리샘플하면 얇은 세그먼트가 부서지고 픽셀 소유 query가 바뀐다). GT와 해상도가 다르면 스코어러가 거부한다.
- **🔴 남은 블로커(코드 아님, 데이터)**: MUSES **panoptic GT 미보유**(`gt_panoptic`/`gt_uncertainty` 미다운로드 — plan.md #4 그대로). 배선·포맷·기하는 끝났고 실제 PQ 수치는 GT 확보 후 산출된다. GT json이 동봉돼 있으면 그걸 쓰고, 없으면 `--build-gt-json`이 Cityscapes `category*1000+instance` 규약으로 유도하되 **파생 category가 표에 없으면 즉시 중단**(추측 금지). test는 GT 비공개라 `--split test` 자체를 거부.
- **DELIVER**: 25클래스에는 공식 panoptic 규약이 없어 `thing_ids` 기본값을 **주지 않고** 명시 요구(`--thing-ids`), GT 없으면 예측만 쓰고 채점 안 함.

### 2026-08-04 — P47-2 UniBal(Uni-modal Balance, 구 D-2) 구현 완료 (학습 대기)

- **범위**: 코드 + config + 합성 스모크까지. **학습 미기동**(지시), push 없음(리뷰 후 develop 병합). 제안 정본 = [decisions/2026-08-03-p47-mub-muses-proposal.md](../decisions/2026-08-03-p47-mub-muses-proposal.md) §3 D-2. 네이밍 규칙 변경 반영해 코드·config는 `P47_2`/`p47_2`(주석에 "구 D-2" 병기).
- **신규/변경**: `semseg/models/reliadino/p47.py`(신규 — `UniModalHead`/`UniModalBalance`/`OGMGE`/`resolve_modals`) · `model.py`(P47_2 파라미터 8개 + `self.p47_2` 생성 + forward에서 `aux['p47_2_uni']` + `build_reliadino` 파싱) · `train_reliadino.py`(손실 합산·per-modal 로깅·OGM-GE step 결선) · `reliadino/__init__.py`(export) · `tools/smoke_p47.py`(신규) · `configs/hpca100-muses_rgbelr_P47_2_unibal_4modal.yaml`(신규).
- **🔴 구현 중 확인한 사실(설계 문서에 없던 것)**: base P39.1에 **이미 per-modal aux CE가 존재**한다(`FUSION.AUX_CE_WEIGHT` 0.5 × `fusion.aux_decoders`의 모달평균 CE). 그래서 D-2를 문자 그대로 "per-modal aux head + CE"로만 만들면 사실상 기존 항의 재구현이 된다. P47-2는 세 지점에서 분리했다: ① head를 reliability/router/calibration 신호원과 **분리**(기존 aux head는 `rel_cal`·`rbma_cal_loss`·P44 mutual-KL의 입력이라 "정확도"와 "보정"의 타협점으로 최적화됨) ② **모달별 가중**(`MODALS` × `LAMBDA_U` — 기존은 모달평균 고정이라 "RGB에만 더 걸어라"를 표현 불가. §1 진단이 지목하는 게 정확히 RGB 본류다) ③ OGM-GE 결선에 필요한 per-modal 성능치 노출. 이 판단이 틀렸다면 P47-2는 λ 재조정과 동치가 되므로, **ep30 즉검에서 per-modal acc가 모달별로 갈라지는지**가 1차 확인점이다.
- **계약 검증(스모크 전항목 PASS, `tools/smoke_p47.py` + `--ddp`)**: 키1 = uni-aux **단독 backward** 시 per-modal LoRA `b_q/b_v` **모달 슬라이스별** grad — `MODALS: all`이면 4모달 전부 >0, `['img']`면 img만 >0이고 나머지 **정확히 0**(모달별 독립의 실증). 추론 등가성 `|Δ|max = 0`(신규 state_dict 키도 `p47_2.*`뿐). 부수효과 0(다른 aux 손실 값 불변 + `encoder.forward` 호출 4회로 동일 = **추가 forward 없음** → ISSUE-028의 2-forward 문제 구조적 무관). DDP는 `find_unused_parameters=True`가 warmup 구간 미사용 head를 처리(스모크가 `WARMUP_EP=5`로 실제 재현).
- **OGM-GE(기본 off)의 DDP 함정**: gradient는 backward 시점에 이미 all-reduce된다 → rank마다 자기 배치로 잰 점수로 다른 k를 곱하면 **rank 간 파라미터가 갈라진다**. k 계산 전에 점수를 all_reduce(mean) 하도록 결선했고(전 rank 대칭·크기 M·step당 1회), 스모크가 rank별로 **다른** 점수를 주입해 k와 gradient가 모두 일치하는지 확인한다. P44 MMPareto와 동시 사용은 `RuntimeError`(둘 다 step 직전 `p.grad` 재작성).
- **메모리 실측(autograd saved-tensor 계측, dim 1024·1024²·4모달·BS1·bf16)**: **+51.7 MiB/스텝** + params 336 KiB(+AdamW 1.0 MiB) = A100 40GB의 0.13%. 초안은 head 앞 정규화에 `encoder.LayerNorm2d`를 썼는데, 그 구현이 파이썬으로 편 elementwise 체인이라 **모달당 full-size 중간텐서 3장**을 그래프에 남겨 증분이 2배였다 → `nn.GroupNorm`(융합 op, 리포 AuxDecoder/FPNSegHead 관례)으로 교체해 4.6배 절감.
- **DELIVER 보호**: DELIVER config에 `MODEL.P47_2` 키가 없어 `ENABLE=False` → `self.p47_2 is None`. 모듈 생성을 `__init__` **최말단**에 둬서 off일 때 init RNG 스트림도 안 건드린다(2026-07-21 ClassTokenLiteHead 중복 생성으로 seed 재현이 깨졌던 사고의 교훈). 현 DELIVER 최고 test 55.7(legal, **정정 2026-08-04: 57.05는 test-best로 무효**) 경로 무변경.
- **⚠️ D-1과 분리**: config에 `DATASET.PROJ_DIR`을 **넣지 않았다**(= SDK 기본 `projected_to_rgb`). 단독 변수 실험이며 합본(D-1+D-2)은 두 단독 결과 후 별도 config.
- **다음**: fresh-eyes 검수(conventions §"코드 검수 파이프라인" 1단계) → develop 병합 → A100급 4장 확보 시 기동 → ep30 즉검(per-modal acc 분화 + base 대비 −1.0 kill).

### 2026-07-29 — P46-CTR jarvis OOM 진단: 누수가 아니라 **warmup 계단** (ISSUE-028), 메모리 회계 5건 수정

- **접수된 증상**: `jarvis-deliver_rgbdel_P46_ctr.yaml`(all-on, BS1) 4090×4에서 ep1~5 정상(15.2GB, ep4 val 59.66) → **ep6 iter0 4-rank 동시 OOM**(23.47GiB). "에폭이 갈수록 서서히 증가하는 누수"로 접수.
- **판정 = 누수 아님**. C2_MCC/C3_PROTO `WARMUP_EP:5` + `for epoch in range(...)`(0-index) + 로그 `epoch+1` → **로그상 ep6 = epoch 5**가 보조 student branch·EMA teacher forward·주 forward prototype 손실이 **최초로 켜지는** epoch이다. ep1~5는 P39.1-base만 돌았으므로 15.2GB에는 **P46 비용이 0**이고, "ep1~5 대비 증가"라는 비교 자체가 성립하지 않는다. iter0에서 4-rank가 동시에 죽는 것도 스텝 누적이 아닌 구조적 peak 증가의 서명(EVAL_INTERVAL:2 → eval은 ep4에서 끝, ep5는 학습만).
- **계측**: CPU tiny 모델 gc live-tensor bytes로 12스텝 추이 — 수정 전 86.3MiB / 수정 후 69.5MiB, **둘 다 완전히 평평(단조증가 0%)**. P46에 단조 누수는 없고 아래는 전부 상수 오버헤드였다.
- **수정 5건(의미·게이팅·warmup 로직 무변경, 메모리 회계만)**: ① 보조 branch `_baux`의 backward **미도달** 서브그래프(m2f/vicreg/aux_ce/router — backward가 해제해 주지 않는다) 즉시 `del` ② 루프 지역변수(`logits`/`aux`/`total`/`_blogits`/`_tlogits`)를 iteration 끝에서 명시 해제 ③ EMA teacher의 eval 분석 탭 `_last_*`(~41MiB, 아무도 안 읽음) 호출마다 해제 — student 탭은 보존 ④ `PrototypeBank._sample` 인덱스-먼저/캐스팅-나중(gather): fp32 전체 사본 3장(108MiB) → 4.0MiB, 호출 2회로 **-208MiB**, 수치 bit-exact ⑤ eval 직후 `empty_cache()`.
- **근본 비용은 남는다**: peak에 student 그래프 2개가 동시에 산다. backward 분리는 **불가** — `find_unused_parameters=True`가 마지막 forward로 unused를 정하는 DDP 계약 때문에 보조 그래프에서만 grad를 받는 파라미터가 생겨 reducer가 정지한다(07-16 NCCL 데드락 부류). `GRADIENT_CHECKPOINT`는 ISSUE-027로 봉인. → 24GB 4090에서 BS1 all-on은 여전히 빠듯하며, 기동 전 `P46_MEM_LOG=1`로 ep5→ep6 계단 실측이 필요하다.
- **회귀 방지**: `tools/smoke_p46.py`에 G(teacher 캐시 해제)·H(PrototypeBank 등가성 `max|diff|=0`)·**I-a 스텝 간 참조 해제**(weakref, tolerance 없음 — **step1의 peak 지점**에서 판정해야 잡힌다. 스텝 종료 후에 재면 수정 전에도 다 죽어 있어 무력)·I-b 메모리 단조성 추가. I-a는 수정 전 5/5 생존 → 수정 후 0/5로 검출력 실측 확인. `--ddp` 포함 전항목 PASS.
- **교훈**: warmup이 걸린 모듈은 **warmup 이후 epoch을 최소 1회 통과해야** 자원 검증이 끝난 것이다. config 주석의 "BS1, OOM-safe per smoke test"가 그 오기록이었고 이번 사고의 실체다.
- 브랜치 `p46-oom-fix`(develop 기준). 사용자 리뷰 후 develop 병합 예정 — push 안 함.

### 2026-07-28 — 브랜치·worktree를 develop 하나로 통합 + 정량 재현 경로(REPRODUCE.md) 신설

- **통합 범위**: worktree 15개 → 3개, 로컬 브랜치 22개 → 5개. 삭제분은 전부 `archive/<브랜치>` 태그 11개로 보존(`git tag -l 'archive/*'`). **`26-drone-certificate`는 사용자 지시로 통합 대상에서 제외**하고 그대로 유지.
- **사용 중이라 보존한 worktree 2개**: `p34-det`(jarvis P39rf det 학습 모니터링 프로세스 15개 상주), `p30-det`(3개). 다른 세션이 실제로 점유 중이라 제거하지 않았다.
- **회수한 고유 자산**(폐기 브랜치에만 있던 것): `tools/eval_muses_official.py`(MUSES 공식 native 1080×1920 재채점기 — develop에 대응물이 없던 유일 평가기), `tools/{viz_features_full,probe_det_features,compare_muses_projections,muses_motioncomp_analysis,muses_pixelmean_check,project_muses_dgfusion}.py`, `configs/hpca100-muses_rgbelr_P34_reliadino.yaml`, `configs/det/det_P29_{egofill_bengio,indoor_jarvis_v2,indoor_jarvis_v3}.yaml`, `configs/det/{det_P34_final_full_local.yaml,README_det_training.md}`, 연구 문서 6종(→ decisions/ det/ experiments/analysis/ research/ 로 이관·MOC 등록). det_eval PNG 24장(19MB)은 미디어 규약대로 git에 넣지 않고 태그로만 보존.
- **이식성 수정**: det config 7개의 `MODEL.COCO_CKPT` 절대경로(`/SSDb/...`) → repo-상대 `weights/rf-detr-large-2026.pth`. hpca100 MUSES config 2개의 `TEST.FILE`이 사망한 b200 경로(`/NHNHOME/...`)를 가리키던 것 → `DATASET.ROOT`와 일치시킴.
- **재현 경로 신설**: `REPRODUCE.md` + `scripts/reproduce_eval.sh <deliver|muses|muses-official|multiaqua|det>`. 기존 평가 진입점만 호출하고 데이터 경로를 덮어쓴 임시 config를 생성하는 방식(원본 `configs/` 무변경). 기대 수치는 `.claude_logs` 실측값만 기재하고 미확인분은 TODO로 명시.
- **검증**: 회수 도구 7종 `py_compile`+`--help` 7/7 통과, config 5종 `yaml.safe_load` 통과 및 모델 식별자 실존 확인, 이관 문서·MOC 상대링크 0 broken, `reproduce_eval.sh` `bash -n`·usage·DRY_RUN(4/5 rc=0; multiaqua는 P9 ckpt가 정규 웨이트 루트에 부재해 의도된 에러 종료).
- **재현 과정에서 잡은 결함 2건**: ① `val_det.py --score_thresh` 기본 0.3이 `train_det.py`(임계값 없음)와 달라 기록된 AP50 0.9321이 재현되지 않음 → 스크립트가 `0.0`을 명시. ② 빈 GPU 자동선택이 `nvidia-smi` 오류 문자열을 그대로 GPU 인덱스로 넘겨 **GPU0(타인 학습)에 얹힐 수 있던 결함** → 숫자 목록이 아니면 중단하도록 방어 추가(현재 이 박스가 NVML mismatch 상태라 실제 발동 확인).
- **CLAUDE.md §1.7의 "P37 병합 대기" 경고 해소**: 9c5e2cc가 develop 조상임을 확인(CEFRHead·classtoken·P37 configs 모두 develop 보유).
- 코딩은 labcode(연구실 계정) 워커 2개에 파일 집합을 분리해 병렬 위임했고, 합격 판정·수치 검증은 이 세션에서 직접 수행.

### 2026-07-25 — P43 PanopticDual + P44 BMR + P45 FogStyle 구현 완료, develop 병합

- P43/P44/P45 구현 완료, develop 병합 35ddbe0(+config 3e3b54f). 제안 문서 = [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md)(§7 토론 반영 포함).
- P43: `semseg/models/reliadino/panoptic_head.py`(MaskClsHead) + encoder.py multi-depth lateral(5/11/17) + model.py 배선, 독립 주손실 `L = L_pixel + λ(t)·L_mask`. 합성 스모크(`tools/smoke_p43.py`) 전건 PASS. PQ 실측은 MUSES panoptic GT 부재로 TODO.
- P44: `mmpareto.py`(gradient 통합) + `p44.py`(mutual KL/coverage 마스킹/presence 재정규화/P45 fogstyle). 스모크(`tools/smoke_p44.py`) 86 assert PASS. 구현 기록 = [models/p44-bmr-implementation.md](../models/p44-bmr-implementation.md).
- 병합 후 P43+P44+P45 동시-on 통합 검증 PASS. 학습0 검증 2건(`tools/analyze_router_coverage.py`)이 hpca100 GPU2/3에서 실행 중(별도 세션 회수 예정). 배치 = P43-MUSES가 hpca100 GPU2,3 첫 슬롯.

---

### 2026-07-24 — P43~P45 CVPR SOTA 제안 등재 (딥리서치 6축 교차)

- 멀티에이전트 딥리서치 6기 병렬(모달불균형/상호증류/fog/panoptic/condition-adaptive/SOTA지형) → [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md).
- 핵심 판정: **MUSES mIoU는 융합에 죽은 SOTA 축**(1위 미발표 카메라단독 GtA 82.39, Codabench 실측), **PQ가 유일한 현실적 SOTA 축**(1위 DGFusion 61.03, frozen-VFM 참가자 0, 우리는 PQ 산출 불가 상태).
- 3안: P43 PanopticDual(dual-head 공동학습, PQ 58~61 착륙 지대) / P44 BMR(MMPareto+peer 증류+MCRM, P42 후계) / P45 FogStyle(FIFO 이식 토글). 대기열 #10/#11. 게이트 전부 사전등록(ep30 조기 kill 포함).

### 2026-07-21 — P37~현재 코드 전수조사 완료: 확정 21건(critical 2·major 6·minor 13), 12건 develop 수정 반영

멀티에이전트 32기로 P37~현재 코드를 전수조사(발견→반증검증 2단계). **확정 21건**(반증 3건 별도), **12건은 이미 develop에 수정 커밋 반영**.

**critical 2건(수정됨)**:
- **ISSUE-026**: ColorAugSSD brightness가 uint8(0-255) 입력을 [0,1] 클램프 → 발화 샘플(p=0.5)의 RGB가 백색 상수로 붕괴(사실상 RGB-dropout 0.5). **영향은 07-16 커밋 이후 `DGFUSION_AUG:true` DELIVER 학습 전부** — jarvis P37a-DELIVER, bengio P37a/b(사망런), **hpca100 P38-DELIVER 200ep 완주분(=P38 게이트 미달 판정에 쓰인 그 런)**, **hpca100 P39-DPC resume(현재도 오염 상태로 학습 중)**, yeon 스모크들. **MUSES 전 계보는 무영향**(`DGFUSION_AUG` 키 자체가 없음). ⚠️ 재해석: P36 fair 게이트(67.74/55.62)는 07-16 이전 학습이라 정상 RGB — **P37+/P38/P39 DELIVER와 P36의 비교는 불공정했음** → **P38-DELIVER "게이트 미달 −1.63" 판정, P39-DELIVER "−1.63 thin-class 퇴행" 판정 모두 보류**. P39.1부터 픽스 적용 첫 클린 DELIVER 런.
- **ISSUE-027**: `GRADIENT_CHECKPOINT=true` 시 timm non-reentrant 재계산이 stale `active_modality`(backward 시점엔 마지막 모달로 고정)로 LoRA를 재실행 → 비최종 모달 gradient가 잘못된 파라미터 경로로 오염(무경고). 팀이 bengio에서 실증해 config 주석("절대 true 금지")은 있었으나 코드 가드가 없었고 체크인 configs 9종에 `true` 잔존. 수정 = encoder 강제 off 가드 + configs 9종 `false`. 실피해는 bengio 사망런·yeon 스모크 등에 한정.

**major 6건 요지**: det seam 2곳 tanh(γ) 게이트 생략(P39.1 잠복, 헬퍼로 통일) / val.py CEFR ckpt가 λ2=0으로 평가되던 버그(**P37a-CEFR 분석·판정 재검증 필요 가능성**) / module_ablation cross-generation no-op 토글 오판 가드 / fog audit 키메라 측정 중단 가드 / panoptic overlap 표준 M2F 정정 (나머지 1건 포함, 상세는 각 커밋 참조).

**minor 13건 중 수정분 요지**: epoch 경계 grad 유출, vicreg fp32 강제, tb 로깅 누락 3종, classtoken 중복 생성 가드, eval_per_domain exit code 정정 등.

**미수정 기록**: `last_checkpoint`가 eval 후 미갱신(resume 시 top-k 불일치 가능, minor) — 후속 처리 대상.

**조치**: `experiments/plan.md` 실행 중 표에 hpca100 P39-DPC resume 오염 표기 + 사고 기록 1줄, 대기열 #1(P39.1)에 클린런 표기. `issues/issues-and-fixes.md`에 ISSUE-026/027 등재(인덱스+본문), ISSUE-024에 전수조사 언급 1줄 추가. 상세는 [issues/issues-and-fixes.md](../issues/issues-and-fixes.md) ISSUE-026/ISSUE-027, [experiments/plan.md](../experiments/plan.md).

---

### 2026-07-21 — ISSUE-025: MUSES radar 디코딩 3중 버그 발견 + develop 수정 완료

P39 4모달(rgbelr) radar 기여 재검토 과정에서 radar 디코더 경로 실측 검증 중 발견(jarvis radar 75파일 실측). **3중 버그**: ① `_open_radar`가 자체 구현 없이 `_open_lidar`로 폴스루 ② 데이터셋 디스패치(`__getitem__`)가 radar를 `_open_radar`가 아니라 `_open_lidar`로 직접 라우팅해 `_open_radar` 자체가 죽은 코드 ③ 결과적으로 radar range가 `LIDAR_RANGE_MAX=100m`에 클립(실측 유효픽셀 2.76% 포화 — radar 센서 실제 캡 = 정확히 150.0m 확인) + height 채널(radar는 전 픽셀 0)이 정규화 후 0.25 상수 평면으로 오염.

**수정**(develop, merge 80d65a0 계보): `RADAR_RANGE_MAX=150.0` 클립 상수 도입, ch3(height)를 radar에서는 occupancy 마스크로 대체, 디스패치에서 radar를 `_open_radar`로 정상 라우팅. lecun 세션의 동시기 미검증 픽스(`lecun-wip-20260721`)와 방향은 같으나 독립 실측 검증 후 재작성.

**영향 범위**: **4모달(radar 포함) 실험만 오염** — P34 4모달 test 78.256("radar 유해 −0.72" 판정은 broken decoder 기준이라 보류), diag_D zeroradar 계열, P39 4모달(jarvis 진행 중, "+0.86"은 broken-radar 하한 취급, 완주는 그대로 진행해 기준선 보존). **3모달 전 계보는 무영향**(P34-3모달 78.979 / P37a / P38-m2f 79.025 / P39-3모달 78.881 등 — radar 미사용, lidar/event/camera 디코더는 원래 정상). 제출 수치 중 오염은 P34-4모달 1건뿐.

**후속**: 대기열에 "P39-4모달 radar-fix 재실험"을 P39.1/P40 다음 순위로 등록([experiments/plan.md](../experiments/plan.md) #3) — 픽스 후 radar 기여 재측정으로 P34 판정 확정/철회. 상세 = [issues/issues-and-fixes.md](../issues/issues-and-fixes.md) ISSUE-025.

---

### 2026-07-21 — MUSES fog per-scene 감사 완료 → 파국장면 가설 기각, P39.1 투입 GO

P39-DPC ep146 vs P38-m2f ep156 fog(n=58)/night(n=100) per-image mIoU 분포 감사(`tools/p39_fog_scene_audit.py`, jarvis GPU6). worst5도 조밀(fog worst ~51+, skew≈0) — 소수 파국 장면이 평균을 끌어내리는 패턴 없음, **가설 기각**. fog 약점은 장면 품질이 아니라 **희소 클래스의 조건부 전멸**(traffic light/rider/train 0@fog)로 판정, 헤드룸 추정 하향 조정. **P39.1 투입 판단: GO**(rank 수리 근거인 공식 test fog_night −12.05는 per-scene 문제가 아니므로 유효). 상세 = [experiments/analysis/2026-07-21-p39-fog-scene-audit.md](../experiments/analysis/2026-07-21-p39-fog-scene-audit.md), 산출물 NAS `analysis_logs/P39_fog_scene_audit_20260721/`. plan.md #1 선행조건 갱신 = fog 감사 완료·GO, trunk_exp-off 재측정은 무효 판정으로 취소(ep30 rank 게이트가 대체).

---

### 2026-07-21 — P39.1(rank 수리) + P40(RCA) 구현 완료 (학습 대기)

**배경**: P39-MUSES 표준분석(2026-07-21)이 **lidar effective-rank 4.7 붕괴**(adapter가 압축 주체, feat_cos 0.115)와 **fog_night 62.68 붕괴**(P39 제출 최저, 전 조건 최저)를 지목. 관련연구 딥리서치 3편(rank collapse / modality imbalance / fog 물리)으로 원인을 교차검증해 제안·구현. 제안 문서 = [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md)(등재 완료).

**딥리서치 대응 요지**: ① lidar rank 붕괴는 **선형 cascaded 경로(V1 선형 투영 + LoRA BA)의 암묵적 저rank 편향**(deep matrix factorization/DirectCLR, LoRA intruder dimensions 문헌)과 정합 — 단순 r 상향은 rsLoRA 없이는 무효. ② 카메라 편중·fog에서 lidar 대체 실패는 **modality laziness/imbalance** 문헌 전반과 일치하되, frozen backbone이라 gradient-modulation류는 지렛대가 없고 **무조건 드롭아웃은 역효과가 실증**(우리 P33 no-op와 정합) — 조건부 강모달 감쇠 + 약모달 보조손실만 유효. ③ fog에서 가장 죽는 센서는 lidar 자신(물리 상한 존재) — "lidar로 fog를 전부 메운다"는 목표는 비현실적, P38 수준(fog_night 74) 복원이 현실 목표. ④ 조건부 모달리티 드롭아웃 자체는 선행(OPM/SGMA)이 있으나, **자기추정 per-sample 신호 + dense prediction + frozen-VFM 제약** 조합은 미점유.

**P39.1 (즉시 수리, 주 변수 1개)**: P39-DPC 위에서 V2(modal-token attention)·V3(앵커)·V4(쿼터)·router 직접감독·deep-sup은 동결. **R-1(주 변수)**: V1 트렁크 결합 `fused += P_m(f_m)`(선형)을 `fused += tanh(γ_m)·MLP_m(f_m)`(LN→1×1→GELU→1×1 + 모달별 스칼라 γ, **init 0.1**)로 교체 — γ=0(완전 zero-init)이면 tanh(0)=0이라 MLP가 첫 스텝부터 gradient를 못 받아(키1 "zero-init 잔차 사장" 재판) 학습이 시작되지 않음을 스모크로 확인, 0.1로 절충. **R-2**: per-modal 토큰 VICReg var+cov(lidar 가중×1.0/기타×0.25, λ_var 0.1/λ_cov 0.01, 2048 서브샘플, fp32) — lidar rank 붕괴 직접 복원용. **M-2**: gate/calib/veto config off(fog_night ablation에서 유해로 재판정된 것 반영). eval마다 per-modal effective-rank(RankMe) 로그(`p391/rank_*`) 추가. **R-3(조건부 2차, 미구현)**: ep30 게이트 미달 시 r 8→16 + rsLoRA + AdaLoRA 직교항으로 재기동.

**P40 (RCA-Fusion, 신모델·논문 주장 모듈)**: P39.1 위에 조건부 감쇠 추가 — **C-1** lidar 리턴 유효성(입력 유도 내부 신호) → 가드/분석. **C-2** 학습 중 자기추정 rel(img)가 배치 하위 분위(30%)인 샘플의 img feature를 soft 감쇠(α 0.1~0.5, hard-zero 금지, curriculum warmup 20ep). **C-3** 감쇠 샘플 한정 lidar readout 보조 CE(w=0.5, gradient 출구). **C-4** 사전 검증: fog_night rel AUROC(img) ≥0.75 확인(학습 전 게이트, 미달 시 C-1 통계 신호로 대체). 서사: 신뢰도 기계를 5세대(P28→P39) 실패한 "추론-시 재가중"에서 "학습-시 조건화"로 이동 — 외부 신호 0 유지.

**판정 게이트(사전 등록)**: P39.1 ep30 = **lidar effective-rank ≥15** & **fog_night drop-lidar ≥4.0**(미달 시 R-3 재기동, R-1/R-2 모두 무효면 V2를 원인으로 전환). P40 = MUSES **test ≥79.025**(P38-m2f 현 최고) & **fog_night ≥74**(P38 복원) · DELIVER = P36 fair(val 67.74/test 55.62) + thin-class 유지.

**검증**: 합성 스모크 **PASS** — RCA pick 발생, C-1 가드(lidar 부재 샘플 제외) 동작, vicreg/readout 손실 유한, γ/MLP grad 흐름, eval 결정론, linear 모드(구 V1) 하위호환.

**config 5벌**: `configs/jarvis-muses_rgbel_P39_1_rank.yaml` / `configs/hpca100-deliver_rgbdel_P39_1_rank.yaml` / `configs/jarvis-muses_rgbel_P40_rca.yaml` / `configs/hpca100-deliver_rgbdel_P40_rca.yaml` / `configs/yeon-deliver_rgbdel_P40_rca_smoke.yaml`(아키 동일 = 단일 모델 제약). 커밋 **ac5c7fe**(develop). 대기열 갱신 = [experiments/plan.md](../experiments/plan.md) #1(P39.1)/#2(P40) — 선행 = 분석 세션 몫인 fog per-scene 감사 + trunk_exp-off rank 재측정 2건(학습 0). 상세 아키텍처 = [models/arch-evolution.md](../models/arch-evolution.md) P39.1/P40.

---

### 2026-07-20 — P39 Dual-Path Compete 구현 완료 (학습 대기)

**배경**: P38(MaskQueryLite)이 게이트(P36 fair 대비 + thin-class) 미달로 판정됨에 따라, P30~P38 계보에서 반복된 5개 실패 패턴(키1 zero-init 잔차 4연속 사장 / 키2 router 유일 실적+co-adaptation / 키3 FUSED rank 7/256 병목 / 키4 문제 위치가 클래스축·도메인축으로 상이 / 키5 event 기여가 데이터셋 속성)을 규칙으로 역변환해 P39를 설계. 제안 문서 = [decisions/2026-07-20-p39-dual-path-compete-proposal.md](../decisions/2026-07-20-p39-dual-path-compete-proposal.md)(등재 완료). **user 지정 제약**: 단일 아키텍처로 DELIVER·MUSES를 모두 커버 — 데이터셋 적응은 학습된 모듈로만.

**구조 (P38 대비 변경 5개, 전부 토글)**: **V1** trunk rank expansion(`fused' = fused + Σ_m P_m(f_m)`, small-random init) — 게이트 뒤 소실된 모달 부분공간을 주 경로에 복원. **V2** modal-token query attention — m2f query가 fused map 대신 per-modal 토큰 합집합에 직접 cross-attend해 융합 병목(rank 7)을 우회, det 폴백 유지. **V3** anchored+free query — 100개 중 K개는 클래스 고정 할당(Hungarian 없음, P37b 방식+직접 감독)으로 thin-class Hungarian 기아 제거, 나머지는 자유 Hungarian. **V4** balanced point sampling — mask BCE/dice 샘플링에 클래스당 최소 쿼터 256pt(thin 마스크 소멸 방지). **V5** compete-and-arbitrate — zero-init β 잔차 결선을 폐기하고 **path dropout 경쟁**(dense-only CE 25% / query-only CE 25% / 결합 CE 50%)으로 학습, 추론은 **per-class 학습 중재** `final_k = dense_k + softplus(Λ_k)·query_k` + **router 직접 CE(w=0.4)**로 router를 자립 기여로 전환.

**판정 게이트(사전 등록)**: DELIVER = P36 fair(val 67.74/test 55.62) + thin-class 복원(Wall≥13/Water≥9.5/RailTrack≥62, P36 수준) · MUSES = **P38 val 82.22 이상**(신규 내부 최고). 모듈 판정 = `module_ablation.py` 토글 즉검(`p39_query_off`/`p39_trunkexp_off`/`p39_anchored_off`/`router_off`, 완주 후 발견 금지, |Δ|>0.5 & agreement<0.99 no-op 기준). **ep30 조기판정** 규칙 적용(2026-07-16 EPOCHS 사고 규칙 준용).

**검증**: 합성 스모크 **PASS** — 5지점 grad 흐름, 토글 5종 전부 유효, det(query-only 등) 폴백 확인, β/Λ 초기화 경로에서 P38 호환 등가성 확인. **실데이터 스모크 미실행**(yeon 배치가 본학습 선행조건).

**config 3벌**: `configs/hpca100-deliver_rgbdel_P39_dpc.yaml`(200ep) · `configs/jarvis-muses_rgbel_P39_dpc.yaml` · `configs/yeon-deliver_rgbdel_P39_dpc_smoke.yaml`(2ep). 커밋 **c31dcd5**(develop). 대기열 등재 = [experiments/plan.md](../experiments/plan.md) 대기열 #1(hpca100은 P38-DELIVER 종료·판정 후 그 슬롯, jarvis는 P38-MUSES 완주 후 이어달리기). 상세 아키텍처 = [models/arch-evolution.md](../models/arch-evolution.md) P39.

---

### 2026-07-18 — P38 MaskQueryLite hpca100 본학습 launch + bengio P37a/b 사망 확정 + hpca100 WIP 보존

**seg-P38 본학습 launch**: hpca100 GPU 0-3(A100×4)에서 develop @c3d1184 기준 본학습 기동. EPOCHS 200, config `configs/hpca100-deliver_rgbdel_P38_m2f.yaml`, launch 스크립트 `launch_p38_m2f.sh`, log `logs/hpca100-deliver_rgbdel_P38_m2f/run_20260718_033931.log`(서버시간 03:39:31 ≈ KST 12:39경). ~0.77s/it·497it/ep → **ETA ≈24-26h, 07-19 완주 예상**. 기동 검증 통과(iteration 실제 전진 342→420/497, rank0 포함 4GPU util 83-100%·메모리 25GB, 에러 0, M2F ENABLE 확인, params 355.4M/trainable 52.3M). 판정 게이트 = P36 fair(val 67.74/test 55.62) 대비 + thin-class(Wall/Water/RailTrack) IoU. `~/SSDb/jemo_maeng/dset/DELIVER`(13118MB)가 hpca100에 스테이징 완료(yeon→릴레이, .DONE 검증)돼 이후 세션은 재스테이징 불필요.

**실데이터 2ep 스모크 병행**: yeon GPU0에서 진행 중(log `/SSDb/jemo_maeng/src/p37_test/logs/p38smoke_20260718_121834.log`, 10.2GB@bs1GPU, ETA ~2h) — 본학습 launch 전 선행조건이던 실데이터 미검증을 메우는 용도, 완주 시 GPU0 반납하며 수치는 참고용.

**bengio seg-P37a/b 사망 확정**: GPU5 HW 고장으로 인한 노드 CUDA 전체 장애가 재부팅 후에도 SSH 미복귀로 확정 — bengio분 seg-P37a/b는 ep1~2에서 종료. jarvis에서 재기동된 P37a→P37b 체인이 계보를 승계(남 세션 소관, 수치는 갱신하지 않음).

**hpca100 로컬 WIP 보존**: seg-P38 launch 전 hpca100 체크아웃에 남아있던 타 세션의 미커밋 MUSES 작업을 wip 커밋(3e7fd68)해 브랜치 `hpca100-wip-20260718`로 보존, GitHub에 push(릴레이). hpca100 체크아웃은 develop @c3d1184로 전환 — 원 세션이 회수 가능.

---

### 2026-07-17 — P38 MaskQueryLite 구현 완료 (학습 대기)

**배경**: P36 공정 레시피(GATE·VETO·CALIB·ROUTER on / ATTN_BIAS·CONSISTENCY off / PHYSAUG off / DGFUSION_AUG on)를 동결한 채 **Mask2Former-lite query head** 하나만 추가해, head confound를 제거한 1-변수 비교를 구성. 동기 3가지: ① DGFusion/CAFuser는 OneFormer(mask-classification) 스택이라 MUSES 주표가 PQ인데 우리 per-pixel head는 구조적으로 PQ 산출 불가였음 → mask-cls 전환으로 해소 ② mask-cls는 문헌상 thin/희소 클래스(Wall/Water/RailTrack)에서 +1~3 mIoU 우세 ③ head를 통제 변수로 고정해 남는 성능차 = 신뢰도 라우팅 융합의 몫으로 귀속 가능.

**구현**: 100 learned query, 6-layer masked cross-attn(gated fused stride-16 map 위, P37b `_TokenDecoderLayer` 재사용) + 공유 cls(K+1)/mask-embed head + deep supervision. attn mask = 이전 layer의 공유-head mask 예측을 stride4→16 리사이즈(Mask2Former 관행). 손실 = Hungarian(scipy) 매칭 + CE(no-obj weight 0.1) + point-sampled BCE/dice(가중 2/5/5, 12544 pts), 모델 내부에서 계산해 `aux['m2f_loss']`로 노출(trainer LOSS_W 0.5). 최종 출력 = `conv_head + β·sem_query + router_alpha·routed`(β zero-init) → 시작 시 P36과 **byte-identical**(합성 스모크 |on−off|max=0.0로 검증). 파라미터 ~5.2M. `panoptic_inference()`(표준 M2F 후처리) 포함 — MUSES PQ 산출용.

**파일**: `semseg/models/reliadino/m2f_head.py`(신규), `model.py`/`train_reliadino.py`(배선), `configs/bengio-deliver_rgbdel_P38_m2f.yaml`(200ep, 768², bs2, 8-GPU 상정), `configs/yeon-deliver_rgbdel_P38_m2f_smoke.yaml`(2ep). 커밋 **3bb2c41**(develop 병합 tip 6d922bd).

**검증**: 로컬 합성 스모크 **PASS**(fwd/bwd 유한, query/cls grad 흐름 확인, β-zero 등가성 exact, panoptic 경로 동작). **실데이터 스모크는 미실행** — yeon 8-GPU 전부 다른 실험(det_P37 등)에 점유되어 있어 유휴 GPU 확보 전까지 보류. 대기열 등재 = [experiments/plan.md](../experiments/plan.md) 대기열 #7. 상세 아키텍처 = [models/arch-evolution.md](../models/arch-evolution.md) P38.

**발견 이슈(신규 등재, ISSUE-024)**: P37b `classtoken.py`의 `mask_proj`(attn-mask 예측기)가 threshold 비교(비미분)로만 쓰여 gradient를 전혀 받지 않음 → 영구 random init, masked attention이 사실상 random 마스킹으로 동작(NaN guard + layer1 unmasked 덕에 치명적이진 않음). P38 `m2f_head.py`는 공유-head 예측 리사이즈 방식으로 처음부터 올바르게 구현. P37b가 kill-gate 생존 시 동일 방식으로 후속 수정 필요 — 상세 [issues/issues-and-fixes.md](../issues/issues-and-fixes.md) ISSUE-024.

**미해결**: bengio 여전히 SSH 불통(GPU5 HW 고장 추정) — seg-P37a/b 생존 미확인 유지, P38 launch 우선순위는 P37 지속 다음.

---

### 2026-07-15 — RA-L 논문 트랙 개시 (세션 "MMSAM | *Paper")
- NAS 볼트 `_paper_submission/` 생성, RA-L 템플릿(ieeeconf.cls/IEEEtran.bst) 적용, 멀티에이전트 워크플로우(10 agents)로 abstract~conclusion 전 섹션 + references.bib + TikZ 그림 3종 + figure_plan 작성, pdflatex 컴파일 통과 (초안 v1, 9p).
- PDF 열람 = `_paper_submission/ReliaDINO_RAL_latest.pdf` (스텝마다 갱신). 타 세션 실험 슬롯 8개 = [research/ral-paper-plan.md](../research/ral-paper-plan.md).
- ⚠️ 초안은 P36 headline으로 작성됐으나 07-15 최종 판정(legal 최선 = P34 val 68.19/test 56.62, test-SOTA −0.09 미달·57.60 철회)에 따라 P34 재중심화 + 주장 완화 리라이트 예정(사용자 피드백 대기).

### P29·P31·P32·P34 표준분석 종합 완료 (동일 프로토콜 4모델) — 2026-07-12

lecun GPU0-3에서 표준 분석항목 1-4 풀 파이프라인 실행 완료. **종합 = [experiments/analysis/2026-07-12-p29-p34-standard-analysis.md](../experiments/analysis/2026-07-12-p29-p34-standard-analysis.md)**, 산출물 = NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/` (⚠️ HDD2 ISSUE-023 MFT 고갈 **재발** — 쓰기 불가, NAS 대체). 헤드라인: ① **P34(ep40 스냅샷) 전 도메인 1위**(mean 53.96, +1.75 vs P29) + Water 0→12 부활 = ISSUE-008(frozen backbone ceiling) 실증 ② SAM2 계열 피쳐 **rank-1 붕괴**(depth 1.1, fused 1.3) + 모달 비정렬(CKA~0.1) vs DINOv3 rank 10-20 + 정렬(0.85) ③ additive-bias 3세대 연속 no-op(P32 CoRB·P31 RBMA-eval·P34 λ1/λ2) vs **P31 router +10.7~13.8 유일 대형 기여** ④ P31 calibration loss만 geometry AUROC 수리(lidar .38→.97). 구 P29 per-domain 로그(mean 59.06)는 프로토콜 상이 확정 — 폐기. 발견 버그 수정: val.py num_classes(DELIVER=25), pipeline --label, feature_stats 채널 수 상이. P34 분석 지원(빌더 분기·eval 스태시·mm_lora 패턴·fusion 토글 5종) develop 반영.

### 표준 분석항목 1–4 도구 스위트 완성 (모델 분석 전담 체계) — 2026-07-12

**배경**: 사용자가 Seg SOTA 도달을 위한 **모델 분석 전담** 지시 + 표준 항목 4개 지정 (① VFM adapter 모달 적응도 ② 모달별 피쳐 수치+시각화(전체 테스트셋) ③ fusion/모듈 전후 비교 ④ 모델별 클래스×도메인 격차→극복 지점). **P31/32/33/34+ 공통 재사용**이 요구사항.

**산출물** (`feat/seg-analysis`→develop): worktree-p30-det의 4차원 파이프라인(seg_analysis_pipeline/adapter_health)을 develop에 포팅 + 신규 도구 4종으로 갭 마감 —
- `tools/modal_adaptation.py` (D3B, 항목①): adapter 출력이 additive인 점을 이용한 **adapter on/off A-B** — per-modal Δfeat/Δcos/Δacc로 non-RGB 적응도 직접 측정
- `tools/feature_stats.py` (D2N, 항목②): full-testset per-modal 수치 통계 — norm/dead-channel/effective-rank/cross-modal CKA + fused 통계 + PCA png
- `tools/module_ablation.py` (D5, 항목③): 모듈 toggle 전후 per-class ΔmIoU + fused-feat cos/shift — **no-op 모듈 감지기**(ISSUE-022류) 겸용
- `tools/compare_models.py` (항목④): N모델 D1 산출물 통합 → STRUCTURAL/DESIGN-GAP/DOMAIN-GAP/SOLVED 자동 분류 digest. **실데이터 검증 완료** (P29 ep100 vs ep146: STRUCTURAL=Ground/Other, Water 16pt 격차 검출)
**canonical 문서 = `tools/README_seg_analysis.md` 상단 매핑표** (experiments/00_MOC.md 등재). 원칙: 새 모델 분석 코드를 새로 짜지 말고 이 매핑으로 실행, 부족하면 도구를 확장. 산출물은 `/mnt/HDD2/src/logs/<model>_eval_<date>/` 누적.

### 차세대 아키텍처 브레인스토밍 + deep-research 완료 — 2026-07-08

**산출물**: `research/vault/material/brainstorm_next_arch_20260708.md` (proposal, 미승인). 내부 문서 전수 + 신규 deep-research 2트랙(VFM/fusion) 기반 후보 카드 5개(A DINOv3-RBMA / B SAM2-RBMA v2 / C SAM3-RBMA 2.0 / D C-RADIOv4 / E Det-deformable) → **추천 top-2 = A(본명)+B(안전판, 병행)**. 선행 검증 실험 6건 제안(최우선 = B-1: 학습 없는 consistency-신호 AUROC 스왑). doc 12 §5·doc 10 상단에 포인터 기록. 다음 액션 = 사용자 승인 + B-1 실행.

---
### P32 (CoRB) 구현·학습·분석 — corroboration 신호는 OK, 라우팅 실패 — 2026-07-06

**배경**: P32 로드맵(doc 23 seg-arch-proposals-P32, `worktree-p32-corrb` 브랜치, branch `worktree-p32-corrb`) §7 step 1-2. RBMA self-entropy(event/LiDAR anti-calibrated AUROC .30/.22)를 무학습 cross-modal corroboration으로 교체.
**구현**: `tools/eval_reliability_auroc.py`(Phase 0 진단) + `LoRA_Sam_P32(LoRA_Sam_P31)._compute_bias_source` override(corr_veto = leave-one-out 합의 Bhattacharyya + unique-info veto blend, temperature-free, config `CORROBORATION.ENABLE`, OFF→P31 byte-identical) + `val.load_model`에 P29~P32 structural threading 미러링(eval mismatch 수리) + config `b200-deliver_rgbdel_P32_physaug`(순수 ablation: P28 base+corroboration). 검증: py_compile·수식==검증도구(4.7e-6)·B200 GPU smoke.
**Phase 0(무학습 게이트) PASS**: P28 self-entropy [.77,.62,.30,.22] 재현 → corroboration이 event/LiDAR를 .54/.81로 반전. v2 재측정으로 신호형=corr_veto 확정(P31 depth workhorse .90→.28 붕괴를 veto가 .71 회복). 상세 [p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md).
**학습 결과(B200 4-GPU DDP)**: plateau **Test 53.45@ep40 / Val 61.65@ep30 < P28 55.27/63.40·P31 54.75/63.20** (−1.8/−1.8).
**분석(hinton GPU, `/mnt/HDD2/src/logs/P32_eval_20260706/`)**: **"신호는 맞고 라우팅 실패"** — corr_veto AUROC 양호(event .59/lidar .85)나 drop-modality Δ[img6.2,depth15.6,event~0,lidar~0]로 **event/LiDAR 여전히 미사용(Mode C)**. per-modal competence event/lidar≈0(viz: pred:event/lidar=노이즈). corroboration이 약모달을 up-weight해 P28 self-entropy(down-weight=쓰레기무시)보다 소폭 악화. per-domain spread 3.79(class-transfer 문제, Mode B). 구조적 사망 Bridge/Water/Wall/Other IoU~0. **교훈: signal AUROC ≠ routing 이득, hard selection 필요.**
**다음**: P32-C(PruneMem: modality dropout+hard pruning)로 drop-Δ≈0 직격.

---

### 2026-07-08 — 리포 전면 재구조화 (문서 IA + 코드 모듈화 + configs 재편 + Obsidian 볼트 수리)

**브랜치 `restructure/ia-taxonomy` (develop 기반 15커밋, draft PR). 사용자 승인 설계(Phase A 인벤토리 3종 → B 설계 → C 실행 → D 회의적 검증 PASS).**

- **문서**: `.claude_logs` 평면 번호체계 → 폴더 택소노미(status/models/experiments/det/datasets/research/decisions/infra/issues/meta/archive). 01 분할(current/history-H1/H2, diff 무손실 검증), 폴더별 00_MOC, `experiments/registry.md` 신설(실험↔config↔ckpt↔문서 허브), 00_INDEX에 구번호 매핑표. 루트 README(upstream 것 아카이브 후 교체)·CLAUDE.md·AGENTS.md·.cursorrules(타 프로젝트 잔재 제거) 갱신.
- **코드**: 메가파일 `sam_lora_image_encoder_seg.py`(375KB, 40클래스) → `lora_sam/` 패키지(base/heads/p08~p31/det/legacy + `MODEL_REGISTRY` 41종, `eval()` 제거), `sam_lola_utils.py` → `modules/{moe,fusion,reliability,common}.py`. 구경로 전부 re-export shim으로 무중단 (P9 ep131 ckpt 로드 missing/unexpected 0 + forward 검증). 위생: `.wandb_key` 추적중단(히스토리 purge 별도), val_mm.py 중복 제거, 데드 스크립트 `_archive/oneoff/` 이동.
- **configs**: 데이터셋별 재편(deliver 31/multiaqua 21/eval 39/archive 27, **파일명 불변** — output 매핑 보존), `profiles/` 서버별 경로 참조 + README(신규 명명 규칙: 서버접두어 금지). B200 학습 중 config는 구경로 symlink 보호.
- **Obsidian**: NAS 원본 볼트 수리(번호충돌 46→93·90_jepa→91, alias 14노트로 bare wikilink 해소, MOC 정비, `VAULT_CHANGELOG_2026-07-08.md`) + `scripts/sync_research_vault.sh` 신설로 repo 사본을 생성물화(손편집 금지).
- **데드 outputs**: 18개 디렉토리 ~176G **전량** → `/drone_nas/home/jemo_archive/MemorySAM_dead_outputs_20260708/` 이동 완료(검증 후 원본 삭제, HDD1 ~170G 회수). 이 과정에서 **ISSUE-023**(/mnt/HDD2 ENOSPC = NTFS MFT 레코드 고갈) 발견 → 판별 실험으로 원인 확정 → 아카이브 NAS 소산으로 완화 완료(쓰기 정상화 검증). `outputs/ARCHIVE_MANIFEST.md` 참조.
- **⚠️ 운영 주의**: 원격 서버(B200 P31 seg, jarvis det, bengio det final_full)는 **진행 중 학습 종료 전까지 이 브랜치 pull 금지** (taskboard R3). 병합 후 새 실행부터 config 신경로 사용.

*(분할 시점 2026-07-08 기준, 구 `01_project_status.md` history에는 2026-07-01 이후 날짜의 엔트리가 없었음 — 최신 엔트리 2026-06-24. 이후 이력은 여기에 누적.)*
