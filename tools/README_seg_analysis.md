# Segmentation analysis pipeline (model-agnostic)

## 🎯 표준 분석항목 1–4 ↔ 도구 매핑 (2026-07-12, 사용자 지정 — P31/32/33/34+ 공통)

**새 모델이 오면 도구를 새로 짜지 말 것.** 아래 4개 항목은 이 매핑의 도구/스테이지로 실행하고,
산출물은 `/mnt/HDD2/src/logs/<model>_eval_<YYYYMMDD>/`에 누적한다. 부족한 건 도구를 **확장**해서
(새 파일 대신) 여기 매핑을 갱신할 것.

| 항목 | 질문 | 도구 (파이프라인 스테이지) |
|---|---|---|
| **1. VFM adapter 모달 적응도** | non-RGB가 adapter로 얼마나 적응했나 | `modal_adaptation.py` (D3B: adapter on/off per-modal Δfeat/Δacc) + `adapter_health.py` (D3: 정적 per-site/expert ‖dW‖, CPU-only) |
| **2. 모달별 추출 피쳐** | 각 모달 피쳐가 어떤가 (수치+시각화, 전체 테스트셋) | `feature_stats.py` (D2N: full-testset norm/dead-ch/eff-rank/CKA + PCA png) + `viz_features.py` (D2/D4: per-image 패널) + `module_diagnostics.py` B (modal competence) |
| **3. fusion/제안 모듈 전후 비교** | 모듈이 수치·피쳐를 어떻게 바꾸나 | `module_ablation.py` (D5: toggle 전후 ΔmIoU per-class + fused-feat cos/shift + **no-op 모듈 감지**) + `module_diagnostics.py` D/E (UAMM alloc, drop-Δ) + `eval_reliability_auroc.py` (신호 대체 비교) |
| **4. 모델별 클래스×도메인 격차 → 극복 지점** | 어디를 극복해야 하나 | `eval_per_domain.py`→`analyze_per_domain.py` (D1) + **`compare_models.py`** (N모델 통합: STRUCTURAL/DESIGN-GAP/DOMAIN-GAP/SOLVED 자동 분류) |

한 방에 전부: `seg_analysis_pipeline.py --stages D1,D2,D2N,D3,D3B,D4,D5` (capability probe가
모델이 지원 안 하는 스테이지는 사유와 함께 자동 skip → 어느 P 버전이든 안전).
모델 간 비교는 각 모델의 D1 산출 후:
```bash
python tools/compare_models.py --run P29=<dir>[:ep146__*.log] --run P31=<dir> --out compare.md
```
`module_ablation.py`의 `pred_agreement≈1 & feat_cos≈1` 판정은 **죽은 모듈(ISSUE-022류,
훅 미도달/파라미터 미사용) 감지기**로도 쓴다 — 새 모듈 학습 전 1회 실행 권장.

---

One driver — `tools/seg_analysis_pipeline.py` — runs four analysis dimensions over **any**
DELIVER segmentation checkpoint (SAM2 family `LoRA_Sam_P8..P33` **and** SAM3-RBMA
`LoRA_Sam3_RBMA`), auto-detecting the model family and which diagnostic hooks the checkpoint
actually exposes, then running each dimension with **graceful skip + a logged capability
matrix** so a blank panel is never mistaken for "covered".

## The four dimensions

| Dim | Question | Tool(s) | Model-agnostic? |
|---|---|---|---|
| **D1** per-class / per-domain metrics | which classes drop, is it domain shift | `eval_per_domain.py` → `analyze_per_domain.py` | ✅ GT-based, any model |
| **D2** per-modality encoder info | what each img/depth/event/lidar encoder captures | `viz_features.py` (R2) + `module_diagnostics.py` (B) | 🟡 needs per-modal hooks |
| **D3** adapter / LoRA / router health | is the adapter actually adapting? dead layers? | **`adapter_health.py`** (static) + `module_diagnostics.py` (C/E/F) | ✅ adapter_health is agnostic |
| **D4** post-fusion feature info | fused feature, reliability/competence, UAMM alloc | `viz_features.py` (R2/R3/R5) + `module_diagnostics.py` (D) | 🟡 needs fusion hooks |

**Always run regardless of family:** D1 (metrics come from GT, not hooks) and the
`adapter_health.py` part of D3 (reads the state_dict statically — no forward, no GPU, no data).

## `adapter_health.py` — the model-agnostic gap-filler (NEW)

The previous toolkit could only infer "is the adapter adapting?" indirectly (drop-ΔmIoU).
This reads the checkpoint's state_dict directly and, for every injected qkv site, computes the
effective delta-weight `dW = B @ A`:

- **plain LoRA** (`*.linear_a_q/b_q/a_v/b_v.weight`) — SAM2 plain blocks **and** SAM3-RBMA
  (`sam3_lora_rbma.inject_plain_lora`, B init-0).
- **SoftMoE LoRA** (`*.experts_a.{i}/experts_b.{i}.weight`, `*.gate.weight`) — SAM2 P8+ SoftMoE.

Reports per site: `||dW||_F`, `||B||_F` (B is init-0 → `||B||≈0` after training = **dead adapter**),
ratio `||dW||/||W_base||` (if the frozen qkv is in the ckpt), and for MoE the per-expert `||dW||`
+ expert-usage CV (≈0 = collapse) + gate norm.

```bash
# instant, CPU-only, no data — works on ANY .pth (raw or {'model_state_dict':...})
python tools/adapter_health.py --ckpt <model.pth> --out health.json
```

## Full pipeline

```bash
python tools/seg_analysis_pipeline.py \
  --cfg configs/b200-deliver_rgbdel_P33_1_physaug.yaml \
  --model_path outputs/MMSamP33/.../best_checkpoint.pth \
  --dataset-root /path/to/DELIVER --out-dir <out> --gpu <free-gpu> \
  [--stages D1,D2,D3,D4] [--conditions cloud,fog,night,rain,sun] \
  [--viz-case sun --viz-contains RailTrack --viz-num 2] \
  [--max-imgs 120] [--skip-per-domain]     # skip the heavy 5-condition eval
```

Outputs under `<out-dir>/`:
- `report.md` — consolidated summary + **capability matrix** (which `_last_*` hooks are live) + per-stage status.
- `capability.json` — family, modals, live-hook map, forward-ok.
- `adapter_health.json` — per-layer LoRA health (D3, always).
- `per_domain/` + `per_domain_analysis.md` — per-domain × per-class IoU + failure classes (D1).
- `module_diag.json` — modal-competence / reliability-AUROC / UAMM alloc / drop-ΔmIoU / MoE (D2-4).
- `viz/panel_*.png` — per-modal encoder + fused + reliability + UAMM panels (D2/D4).

## Family behavior (why graceful skip matters)

- **SAM2 family** (`LoRA_Sam_P*`, incl. P32/P33.1) sets rich hooks
  (`_last_per_modal_feats/outputs`, `_last_uamm_spatial`, `_last_amf_weights`, `_last_moe_gates`)
  → D2/D3/D4 fully populate.
- **SAM3-RBMA** exposes **only** `_last_reliab`/`_last_reliab_logits`. The pipeline detects this
  via the capability probe and **skips** the UAMM/per-modal panels (logging the reason) rather
  than emitting silently-blank output. D1 + adapter_health still give a full report.

## Known remaining gaps (documented, not yet built)
- ~~Numeric per-modality encoder stats~~ → ✅ closed by `feature_stats.py` (D2N, 2026-07-12).
- ~~Fused-feature quantitative quality~~ → ✅ closed by `feature_stats.py` FUSED row + `module_ablation.py` feat-diff (2026-07-12).
- A SAM3-RBMA-native renderer for `_last_reliab` as competence/quality maps (RBMA UAMM-equivalent).
- Per-modality **MoE gate attribution** (which experts serve which modality) — 현재는 aggregate만
  (module_diagnostics F). encode 순서 chunking으로 가능하나 미구현.
- `module_ablation.py`의 toggle 레지스트리는 attr 이름 기반 — 새 모델(P32+)이 새 config-gated
  모듈을 추가하면 `make_toggles()`에 한 줄 등록할 것 (없으면 자동 skip일 뿐 에러는 아님).
