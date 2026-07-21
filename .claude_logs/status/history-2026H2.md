---
split_from: 01_project_status.md
created: 2026-07-08
period: 2026-07-01 ~ 2026-12-31
---

> **역할**: 2026 하반기(2026-07-01~) 역시간순 진행 로그. **새 진행 엔트리는 이 파일 최상단(이 안내 블록 바로 아래)에 추가**한다.
> 현재 상태는 [current.md](current.md), 2026-06-30 이전 이력은 [history-2026H1.md](history-2026H1.md) 참조.

## 역시간순 진행 로그 (History — 2026H2)

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
