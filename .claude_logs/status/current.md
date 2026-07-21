---
legacy_id: 01
legacy_file: 01_project_status.md
split_from: 01_project_status.md
moved: 2026-07-08
---

> **역할**: 프로젝트 **현재 상태 스냅샷의 단일 출처(single source of truth)** — 매 갱신 시 아래 스냅샷 블록만 덮어쓴다.
> 과거 진행 이력(역시간순)은 [history-2026H2.md](history-2026H2.md)(2026-07-01~) · [history-2026H1.md](history-2026H1.md)(~2026-06-30)로 분리됨.

# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-07-08

**⚠️ 리포 재구조화 (2026-07-08, develop 병합)**: `.claude_logs` 폴더 택소노미·`lora_sam/` 패키지(MODEL_REGISTRY)·configs 재편. 구번호 매핑 = [00_INDEX.md](../00_INDEX.md), 규칙 = [meta/conventions.md](../meta/conventions.md). **원격 서버는 진행 중 학습 종료 전까지 pull 금지.**

---

## 📌 현재 상태 스냅샷 (CURRENT — 여기만 읽으면 됨)

> 이 블록은 **현재 상태의 단일 출처(single source of truth)**다. 매 갱신 시 이 블록만 덮어쓰고,
> 과거 진행 내역은 아래 "역시간순 진행 로그"에 엔트리로 남긴다. (아래 로그의 `## 현재 상태:` 같은 옛 헤더는 그 시점 스냅샷일 뿐 현재 아님.)

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
| **Det 객체검출** | 🎯 **best=bengio det_P29_egofill mAP50 0.8501** / event ablation 완주·final_full 진행 | **egofill(RUN-11) 🏁완주**: best **mAP50 0.8501@ep9**(공식 v2 test) — **목표 0.85 달성**(lidar egofill+2×데이터). **det_P29_event(RUN-14) 🏁완주**(07-07): best mAP50 **0.8427@ep14** → event≈egofill-lidar(−0.008, 모달 ablation 유의미). **det_P29_final_full(RUN-15) 🟢학습중**(07-08~): P29+egofill을 최종 annotation(_final_ann/instances_train_egofill.json)으로 재학습, EPOCHS50 ep0. det_P31_v3clip(RUN-10) 완료: mAP50 0.4724(v3clip=비공식). P30-Det 0.256. 모니터 RUN-10/11/14/15 |
| **SAM3 RBMA (포팅)** | **학습/디버깅 중** (DELIVER 25cls) | ✅6/21 decoder repurpose로 class-collapse 돌파: val 8.49→**16.27@ep22 (상승 중)**. 다음=ep40~60+ 상한 확인 |
| **P29 (SDC 조건 라우팅)** | **설계 완료 (구현 대기)** | Soft-MoE LoRA 라우팅 비특화 진단 → label-free image-derived 조건 latent+prototype→FiLM gate(헤드라인), RBMA 신뢰도를 라우팅으로 확장(P29-B). 상세 [models/arch-evolution.md](../models/arch-evolution.md) P29 / 노벨티 [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.7 |
| **P30 (class-token decoder + reliability-anchored router)** | **구현 완료 (학습 대기, P28 종료 후 GPU 2,3)** | P28 실패분석(rare-class collapse: Water/Bridge=0; event/LiDAR 미사용 Δ≈0) 직격 → ① class-token decoder(SAM3-RBMA class-collapse break 이식, m_feat에 class query cross-attn) ② reliability-anchored 학습 modality router(상수수렴 방지, per-class). 두 모듈 CPU smoke PASS, 모델 wiring compile-only. config `b200-deliver_rgbdel_P30_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P30 / 노벨티 [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.8 |
| **P31 (Calibrated Dual-Reliability RBMA + MS-HR class-token decoder)** | **구현+P31.1 수정 완료 (B200 자동 launch 대기)** — 2026-07-03, develop 반영 | doc 20 P31-Seg core 우선순위 ①② 구현: [Seg-A] per-modal temperature + correctness-contrastive **calibration loss**(event/LiDAR AUROC .30/.22 수리) / [Seg-C] `ClassTokenDecoderMS`(simple-FPN {4,8,16,32} + 학습형 ConvTranspose HR pixel-embed + training-only aux CE @H/4) / [레버①] Hiera 마지막 3 block unfreeze(LR×0.1) / [레버②] **router 'decisive' reg**(uniform 라우팅 해소) / [Seg-B] consistency bias·rel-AMF는 기본 OFF(AUROC>0.5 조건부). **P31.1 (비판 리뷰 `/mnt/HDD2/src/logs/P31_review_20260702/` 검증 반영)**: P30-seg 실측 붕괴(Val 49.76@ep136/Test 44.10@ep146 = P29 대비 −13.4/−10.2) 확인 → **CTD aux-only 강등**(최종 출력=SAM decoder 복원, CTD는 training-only aux CE) + **per-modal reliability AUROC/router-w 학습 중 로깅**(tb/wandb `p31/*`) + **SDC OFF**. B200 watcher가 P30(ep~194/200) 종료 시 GPU 4-7에 자동 launch. config `b200-deliver_rgbdel_P31_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P31 |
| **P32 (CoRB — Corroboration-Biased Memory Attention)** | 🟡 **학습 plateau + 분석 완료**(2026-07-06, branch `worktree-p32-corrb` 미병합) | RBMA 신뢰도 신호를 self-entropy → **무학습 cross-modal corroboration(corr_veto)** 으로 교체(`LoRA_Sam_P32._compute_bias_source`, λ만 학습). **Phase 0 게이트 PASS**(corroboration이 event/LiDAR AUROC .30/.22→.54/.81 반전, [experiments/analysis/p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md)). **학습 결과 미달**: Test **53.45@ep40** / Val **61.65@ep30** (P28 55.27/63.40·P31 54.75/63.20 대비 −1.8/−1.8, ep26~30 plateau). **분석 판정(핵심)**: "신호는 맞고 라우팅 실패" — corroboration AUROC는 좋으나 **drop-modality Δ[img6.2,depth15.6,event~0,lidar~0] = event/LiDAR 여전히 죽음(Mode C)**. soft attention-bias로는 feature/decoder가 약한 모달(competence≈0) 부활 불가, 오히려 P28 self-entropy(약모달 down-weight)보다 소폭 악화. 구조적 사망 class(Bridge/Water/Wall/Other IoU~0)=frozen-backbone ceiling. 산출물 `/mnt/HDD2/src/logs/P32_eval_20260706/`. **처방**: ① **P32-C(PruneMem: hard pruning+modality dropout)** 로 event/LiDAR 강제 사용(다음 단계) ② calibration 복원+corroboration 결합 ③ 구조적 사망=backbone unfreeze/CTD. config `b200-deliver_rgbdel_P32_physaug.yaml`. 상세 [models/arch-evolution.md](../models/arch-evolution.md) P32 / [experiments/analysis/p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md) |

**열린 블로커**
- SAM3 ViT single-scale 한계 → SAM2 P28(val~55) 대비 격차 규명 필요.
- SAM3 최소 클래스(Pedestrian/Pole/sign/Dynamic/Water) 여전히 0 → `decoder_high_res`(FPN skip) 후속 실험 후보.
- P28 multiaqua B200 config 경로 검증.

**다음 마일스톤**: ① SAM3-RBMA 수렴 곡선 확보 → ② SAM2 P28 B200 학습 → ③ RBMA ablation(SoftMoE LoRA / SQG / AMF 제거 robustness).

---

