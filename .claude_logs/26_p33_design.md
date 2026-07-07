# P33 설계 — Competence-Gated Hard Fusion + Modality Dropout (CG-MoD)

> 근거: `25_p32_perimage_analysis.md`(ep108 best weight, test 전체 1897장). P32 결론 = **"신호는 맞고 라우팅은 실패"** + **융합이 competence 무시(misalloc 51.6%)** + **event/LiDAR 사망(competence≈16)** + **도메인 전이 붕괴**.
> 계보: P28(self-entropy RBMA) → P31(calibration+SDC router) → P32(corroboration bias) → **P33(competence-gated hard fusion + modality dropout)**. roadmap `23_seg_arch_proposals_P32.md`의 P32-C(PruneMem)를 흡수·구체화.

---

## 1. P32가 남긴 4개 실패와 P33 처방(1:1)

| # | P32 실패(정량) | 근본 원인 | P33 처방 |
|---|---|---|---|
| F1 | **misalloc 51.6%**, UAMM 균일(0.27/0.27/0.23/0.23) | 출력 융합이 비적응(`AMF_MODE:uniform`) | **C1. Competence-Gated Fusion** — reliability+corroboration로 구동되는 sharpened/top-k 융합 게이트 |
| F2 | event/LiDAR **competence 16.9/15.3**, drop-Δ≈0 | 약한 모달 feature/decoder에 정보 없음 | **C2. Stochastic Modality Dropout(PruneMem)** — RGB/depth 확률적 제거로 event/LiDAR 사용 강제 |
| F3 | corroboration Δlogit 0.25인데 **flip 0.046%**, ΔmIoU −0.013 | bias 권위 부족 + 비보정 디코더 위 작동 | **C3. Calibration 복원 + gate 승격** — corroboration을 attn-bias(soft)에서 **fusion gate 입력(hard)**로 승격, P31 calibration loss 부활 |
| F4 | Wall/Bridge/Water/TrafficLight = **val 생존/test 사망** | day→night class-transfer(용량 아님) | **C4. Thin-class 도메인 강건화** — class-balanced/focal loss + night-consistency + (옵션)backbone 마지막 stage unfreeze |

---

## 2. 아키텍처 (P32 대비 변경점만)

P33 = `LoRA_Sam_P33(LoRA_Sam_P32)` — P32의 corroboration 기구는 유지하되 **신호의 소비 지점**을 바꾼다.

### C1. Competence-Gated Fusion (F1 직격, 최우선)
- **문제**: 현재 `AMF_MODE:uniform`은 4모달 등가중 → depth(43.7)와 event(16.9)를 똑같이 섞음.
- **설계**: per-pixel per-modal 융합 가중치 `w_i(x)`를 **학습 게이트**로 산출.
  - 입력: `[reliability_i(calibrated), corroboration_i(veto), per-modal feature summary]`.
  - **Sharpening**: `w = softmax(g(·)/τ)`, τ<1로 뾰족하게(또는 학습가능 τ). 균일 붕괴 방지 위해 **entropy 정규화 페널티**(너무 균일하면 벌점) 또는 **top-k(k=2) hard selection**(하위 모달 0).
  - 초기화: uniform에서 시작해 competence 방향으로 학습(안정성).
- **config seam**: `AMF_MODE: competence_gate`(신규), `GATE.TAU`, `GATE.TOPK`, `GATE.ENTROPY_REG`.
- **기대**: misalloc↓, depth 지배 복원 → per-image mIoU 즉효 상승(가장 싼 레버).

### C2. Stochastic Modality Dropout / PruneMem (F2 직격)
- **문제**: RGB+depth만 쓰고 event/LiDAR는 미사용(drop-Δ≈0) → 두 모달이 학습 신호를 못 받아 competence가 안 큼(악순환).
- **설계**: 학습 시 각 스텝마다 확률 `p_drop`로 **주력 모달(img/depth) 중 1개를 memory/입력에서 제거** → 남은 모달로 세그를 맞추도록 강제. (memory token pruning = roadmap PruneMem의 hard 버전)
  - 스케줄: `p_drop` 0→0.3 warmup. event/LiDAR는 드롭 안 함(이미 약하므로 살리는 게 목표).
  - eval은 전모달(드롭 없음).
- **config seam**: `MODAL_DROPOUT.ENABLE`, `MODAL_DROPOUT.P`, `MODAL_DROPOUT.TARGETS:[img,depth]`, `MODAL_DROPOUT.WARMUP_EP`.
- **기대**: event/LiDAR competence↑, drop-Δ가 양수로 → C1 게이트가 실제로 라우팅할 정보가 생김. **F1과 상보**(게이트가 있어도 정보가 없으면 무의미, 정보가 있어도 게이트 없으면 무시 → 둘 다 필요).

### C3. Calibration 복원 + corroboration 승격 (F3)
- **문제**: P32는 P31 calibration loss를 버려 event/lidar 디코더가 anti-calibrated(self-entropy rel 0.29/0.36). corroboration은 attn-bias(soft)로만 작동해 flip 0.046%.
- **설계**:
  1. **P31 calibration loss 부활**(per-modal decoder logit shaping) → reliability가 신뢰 가능해짐(Phase 0: corr_veto는 **보정된 모델 위에서 최상**).
  2. corroboration을 **C1 게이트의 입력**으로 승격(soft attn-bias는 보조로 유지 or 제거). 결정경계에 직접 작용.
- **config seam**: `CALIBRATION.ENABLE:true`, `CORROBORATION.CONSUME: gate`(기존 `bias`에서).
- **기대**: 신호가 출력 결정에 도달(flip↑, ΔmIoU 양수화).

### C4. Thin-class 도메인 강건화 (F4)
- **문제**: Wall/Bridge/Water/TrafficLight val 생존/test 사망 = 일반화 실패(용량 아님).
- **설계**(직교, 저위험부터):
  1. **Class-balanced / focal loss** — thin·rare 클래스(Wall,Fence,TrafficLight,TwoWheeler,Ground…) 가중.
  2. **Night-consistency reg** — NIGHT_AUG 강화본과 원본 예측 일치(이미 physaug 존재, consistency term 추가).
  3. (옵션·고위험) **backbone 마지막 stage unfreeze**(P31 CTD 레버) — geometry 사망 클래스(Bridge/Water)용. 비용·과적합 주의 → ablation 후 결정.
- **config seam**: `LOSS.CLASS_BALANCED`, `LOSS.FOCAL_GAMMA`, `NIGHT_CONSISTENCY.ENABLE`, `UNFREEZE_LAST_N_BLOCKS`.

---

## 3. 우선순위 & Ablation 계획
P28 순수 ablation 관례 유지(한 번에 한 축). 순서 = **효과/비용 비**:

1. **P33.1 = C1 단독** (competence-gate, calibration 복원 포함) — 융합만 적응화. **misalloc·per-image mIoU 즉효 검증**. 가장 싸고 확실.
2. **P33.2 = C1 + C2** (게이트 + modality dropout) — event/LiDAR competence·drop-Δ 상승 검증. **Mode C 부활의 본 처방.**
3. **P33.3 = + C3 완전형** (corroboration을 gate 입력으로) — 신호→결정 도달 검증(flip↑).
4. **P33.4 = + C4** (thin-class/도메인) — 사망 클래스 회수.

각 단계 게이트: (a) test mIoU vs P32 54.79 / P28 55.27, (b) **misalloc% ↓**, (c) **modal competence(event/lidar) ↑**, (d) corroboration flip% ↑ — `viz_features_full.py`로 동일 측정 재사용.

## 4. 성공 기준 & 리스크
- **정량 목표**: test mIoU **> P28 55.27 돌파**(P32는 54.79로 −0.48), 궁극 DELIVER SOTA(DGFusion 56.71)·official test 56.71 지향. misalloc < 25%, event/lidar competence > 25.
- **리스크**:
  - C1 sharpening이 depth 단일모달 붕괴로 수렴 → entropy-reg/top-k로 방지, dropout(C2)이 견제.
  - C2 dropout이 주력(depth) 성능 일시 하락 → warmup·p_drop 상한으로 완화.
  - C4 unfreeze 과적합/비용 → 마지막 단계, ablation 후.
- **불변**: 실패해도 F1/F2는 독립 레버 → 부분 채택 가능.

## 5. 구현 seam (코드 위치)
- 모델: `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P33(LoRA_Sam_P32)`. 융합 게이트는 P32의 UAMM/AMF 융합 지점(`_fuse_outputs`/AMF) 오버라이드; corroboration은 이미 `_compute_bias_source` 존재(gate 입력으로 재배선).
- config: `configs/b200-deliver_rgbdel_P33_physaug.yaml` (P32 config 복제 + 위 seam 키).
- 검증 도구: `tools/viz_features_full.py`(per-image + corrb ON/OFF), `tools/module_diagnostics.py`, `tools/eval_reliability_auroc.py` — 모두 `--cfg/--model_path` 교체로 재사용.

## 6. 근거 데이터
`25_p32_perimage_analysis.md` + `/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/`. 핵심 수치: competence depth 43.7≫event 16.9≈lidar 15.3, UAMM 균일 0.27, misalloc 51.6%, corroboration flip 0.046%·ΔmIoU −0.013, dead-on-test/alive-on-val(Wall·Bridge·Water·TrafficLight).
