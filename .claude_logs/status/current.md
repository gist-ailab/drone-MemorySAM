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

**⚡ 2026-07-08 최신 (아래 표는 07-02 시점, P30~P31 시대의 기록임)**

| 트랙 | 상태 | 수치 / 다음 액션 |
|------|------|------------------|
| **P32 (CoRB) seg** | 🏁 **학습 완료 + 4축 독립 검증 완료** | 최종 **Day-Val 64.12@ep98(계보 최고) / Test 55.00**(P31 54.85 +0.15, P28 55.27에 −0.27 미달; 목표 갭 val −2.39/test −1.71). 검증 결론: CoRB attn-bias는 **유의한 순손해**(ΔmIoU −0.013, p=4.5e-22) — 신호는 유효, pre-softmax 주입은 무효. 지배 원인 = per-class 전이 붕괴(복구 상한 +7.9pt). 상세 [experiments/analysis/p32-verification-p33v2.md](../experiments/analysis/p32-verification-p33v2.md) + 볼트 `research/vault/P32_CoRB/P32_정량검증_실패분석_20260708.md` |
| **P33-v2 (CG-MoD 개정)** | ✅ **설계 완료 (구현 대기)** | 원안(doc 26) 적대적 비판 + 딥리서치 3축 반영: M0 무학습 진단 3종 → M1 class-transfer 복구(RCS+text-anchor+MIC consistency, night+**sun**) → M2 dropout+distillation → M3 soft gate(corr_veto) → M4 CoRB 제거. 기대 test 56.5~58. Global escape: val<65.5 → 카드 A(DINOv3-RBMA) 전환. 볼트 `research/vault/P33_CGMoD/P33_v2_설계개정_20260708.md` |
| **Det (국책과제)** | 🎯 **목표 달성** | egofill 데이터(2.01×)만으로 **mAP50 0.8501**@ep9 (official v2 test). 남은 서사 = 저조도 robustness delta. 상세 doc 19 E2.5 |
| **옵시디언↔repo 동기화** | ✅ 규약 제정 | NAS 볼트 canonical, `scripts/sync_vault.sh`(NAS→repo pull), 실험폴더 패턴 `P<N>_<이름>/`. `research/vault/README.md` §🔄 |

**진행 중 트랙 (2026-07-02 시점 기록 — 위 표가 최신)**

| 트랙 | 상태 | 최신 수치 / 다음 액션 |
|------|------|----------------------|
| **SAM2 RBMA seg (P32, B200 DELIVER)** | 🟢 **학습 중**(ep156/200, 07-08) — 🎯 **P32=확정 최선 seg**(val 64.12@ep98>P31 63.20 / test 55.00@ep154>P31 54.85; SOTA대비 val −2.39/test −1.71). 모니터 RUN-13. | <!-- 이하 P30/P31 히스토리 -->
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

