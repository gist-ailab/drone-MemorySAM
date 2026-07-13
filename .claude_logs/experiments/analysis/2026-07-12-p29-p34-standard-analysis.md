---
created: 2026-07-12
scope: P29/P31/P32/P34 표준 분석항목 1-4 (동일 프로토콜)
artifacts: /drone_nas/drone/analysis_logs/ (HDD2 ISSUE-023 재발로 NAS 대체)
tools: tools/seg_analysis_pipeline.py 스위트 (develop aecba1d)
---

# P29·P31·P32·P34 표준 분석 종합 (2026-07-12, lecun)

> **프로토콜**: DELIVER test 5-condition, 동일 파이프라인(batch 2, PHYSAUG off, GT-res D1 + WR-256 모듈 진단).
> 대상 ckpt: P29 test_ep146(54.34) · P31 test_ep182(54.85) · P32 test_ep158(55.01) · **P34 test_ep40(55.08, 학습 중 스냅샷)**.
> ⚠️ 기존 P29 per-domain 로그(2026-06-30, mean 59.06)는 **프로토콜 상이 확정**(재평가 52.21과 −6.9 편차) — 과거 표와 혼용 금지.
> ⚠️ P31/P32의 module_diagnostics·ablation은 WR-256 상대분석 수치 — 헤드라인 IoU는 D1(GT-res) 기준.

## 항목4 — 모델×클래스×도메인 (canonical: `compare_P29_P31_P32_P34_20260712.md`)

| model | cloud | fog | night | rain | sun | **mean** |
|---|---|---|---|---|---|---|
| P29 | 53.34 | 51.93 | 50.66 | 54.12 | 51.01 | 52.21 |
| P31 | 53.28 | 53.72 | 51.48 | 53.53 | 52.07 | 52.82 |
| P32 | 52.89 | 53.54 | 51.82 | 54.65 | 51.20 | 52.82 |
| **P34** | **54.14** | **55.07** | **52.43** | **55.30** | **52.87** | **53.96** |

- **P34(ReliaDINO, ep40 스냅샷)가 전 도메인 1위** (+1.14 vs P31/P32, +1.75 vs P29).
- **Water 부활**: SAM2 계열 0.0~0.2 (STRUCTURAL) → P34 **12.0**. TrafficLight 12~21 → **35.1**. → frozen-SAM2-backbone ceiling(ISSUE-008)이 두 클래스의 원인이었음이 실증.
- **잔여 STRUCTURAL** (전 모델 <10): **Bridge(~0)·Other·Wall** — 백본 교체로도 안 풀림 → 데이터/해상도/annotation 레벨 개입 필요.
- DOMAIN-GAP(타깃 증강 후보): RailTrack(spread 58), TwoWheeler(29), Fence, Water.
- P29 대비 P31/P32의 클래스 이동: RailTrack +7~10, TrafficSign +6 / Ground −4, Pedestrian∼.

## 항목1 — VFM adapter 모달 적응도

| | P31 (SAM2+SoftMoE-LoRA, 96 site) | P34 (DINOv3+per-modal LoRA, 48 site) |
|---|---|---|
| dead adapter | **0** | **0** |
| 정적 ‖dW‖ | site 평균 1.5~22.8 (모달 공유 MoE라 모달 분해 불가) | **per-modality**: img 11.5 / depth **13.4** / event 10.2 / lidar 11.3 |
| on/off Δfeat (rain) | img 0.61 / depth 1.59 / event 2.88 / lidar (대) | img 0.92 / depth **1.18** / event 0.76 / lidar 1.02 |
| on/off Δacc (rain) | img +3.8pt / depth **+13.4pt** / event·lidar 大 | (per-modal 출력 없음 — feat 지표만) |

**판정**: 두 계열 모두 non-RGB adapter가 RGB보다 크게 적응 (dead adaptation 없음). 적응 자체는 문제가 아니었고, 문제는 적응된 피쳐의 **질**(아래 항목2).

## 항목2 — 모달별 피쳐 통계 (night, 120img; full-testset 수치는 json)

| | P31 (SAM2) | P34 (DINOv3) |
|---|---|---|
| eff.rank (img/depth/event/lidar) | 6.1 / **1.1** / 5.6 / 6.0 (32ch) | **13.1 / 10.7 / 19.6 / 13.3** (1024ch) |
| FUSED eff.rank | **1.26** (붕괴) | 6.75 (256ch) |
| cross-modal CKA | 0.02~0.16 (모달 간 비정렬) | 0.80~0.91 (공통 표현으로 정렬) |
| dead channels | 0 | ~1/1024 |

**판정 (핵심 발견)**: SAM2 계열 피쳐는 **rank-1 수준으로 붕괴**(특히 depth 1.1, 융합 후 1.26)돼 있고 모달 간 표현이 정렬되지 않음 → 어떤 fusion 모듈을 얹어도 정보가 부족했던 근본 배경. DINOv3는 rank 풍부 + 모달 정렬 → 융합이 쉬워짐. **P29~P33의 fusion 모듈 반복이 한계에 부딪힌 이유의 피쳐-레벨 설명.**

## 항목3 — 모듈 전후 A/B (`module_ablation`, Δ>0 = 모듈 기여)

- **P31**: **learned router +10.7~+13.8 mIoU (유일한 대형 기여 모듈)** — 끄면 Bus −74/RoadLine −70/TwoWheeler −45 붕괴. RBMA bias ≈0 (**no-op**), temperature ≈0, CTD(aux-only)/SDC(off) = 설계 그대로 no-op 확인. ISSUE-022 수정으로 router가 실제 학습된 P31에서 router는 융합의 중추가 됨.
- **P32**: **모든 토글 |Δ|<0.05 — 제안 모듈(CoRB 계열) 전부 no-op** (P32-B 결론 재확인).
- **P34**: attn-bias(λ1)/consistency(λ2)/veto **≈0.00 (no-op)**, gate ±0.7 혼재, calib 평균 −0.3. **P34의 우위는 제안 모듈이 아니라 백본+per-modal LoRA+FPN 구조 자체에서 나옴.**
- reliability AUROC (night, 동일 프로토콜): P29 [.84,.63,**.26,.38**] / P31 [.43,**.92,.55,.97**] / P32 [.86,.70,.35,.34] — **P31 calibration loss만이 geometry 모달의 reliability를 실제로 수리** (대가: img 역보정 .43). 단, 수리된 신호조차 성능 기여는 router 경유로만 실현(직접 bias는 no-op).

## 설계 반영 (P35 인풋, 수치 근거)

1. **백본이 최대 지렛대** — P34 스냅샷이 이미 계보 1위 + Water/TrafficLight 부활 + 피쳐 rank/정렬 우위. P34 완주·튜닝이 최우선.
2. **pre-softmax additive bias 계열은 3세대 연속 no-op** (P32 CoRB, P31 RBMA-eval, P34 λ1/λ2) — novelty 서사를 bias가 아니라 **calibration(P31이 유일하게 AUROC 수리) × 강한 백본** 조합으로 재구성 필요.
3. **P31 router는 유효 기전** — P34에 learned router(P31식 decisive) 이식이 자연스러운 다음 ablation (P34 gate ±0.7 대비 P31 router +10~13).
4. **잔여 STRUCTURAL(Bridge/Other/Wall)** 은 모델 축으로 미해결 — 데이터·해상도·라벨 정합 조사 트랙 분리.

## 항목②③ 시각화 보강 (2026-07-13 추가 — P34 표준훅 미러링 후)

- **P34 fusion 내부가 표준 도구로 열림** (ReliaDINO에 `_last_per_modal_outputs`/`_last_uamm_spatial` 미러링): module_diagnostics 전 조건 완료 —
  - **rel AUROC (전 조건) = [img .85~.87, depth .78~.81, event .82~.87, lidar .68~.72] — 계보 최초로 4모달 전부 균형 보정** (P31은 img 희생 .43, P29/P32는 event/lidar 사망).
  - gate 할당 ≈ [.28,.26,.24,.22] (완만한 img 우위), **drop-Δ = [img 4~6, depth 15~17, event ≈0(음수 포함), lidar 0.2~1.7]** — depth 지배·event 무기여 구조는 P34에서도 지속.
  - **night misallocation top: RoadLine .79 / TrafficLight .62 / Wall .54** — gate가 유능 모달을 못 고르는 지점 = P31식 learned router 이식의 타깃 근거.
- **per-image 패널 전 조건 확장** (`P34_eval_20260712/viz/`, 5 cond × 2장): 입력 4모달 + 모달별 featPCA + FUSED PCA + (미러링 후) reliability·per-modal mask·gate맵 행 포함.
- **모듈 A/B 전후 패널 신규** (`module_ablation --viz-num`, `*_viz_viz/`):
  - **P31 router on/off (night)**: 불일치 2.7%가 **RoadLine(차선)·가는 구조 경계에 집중** — router +10~13 기여의 공간적 실체 = thin-class 경계 유지. rbma_off는 0.1% 무변화(no-op 시각 확정).
  - **P34 gate/calib/bias off (night/sun)**: 낮은 불일치 — 모듈 no-op 판정의 시각적 재확인.

## P34 최신 웨이트 재분석 (2026-07-13, test-best ep140 = Test 57.60 — 🏆 공식 양대 목표 최초 동시 달성)

> ckpt: `test_epoch140_57.6_top1` (Test 57.60 > 목표 56.71 **+0.89** / Val-best 68.19@ep120 > 66.51 +1.68). canonical 비교표 = `compare_P29_P31_P32_P34_20260713.md` (P34ep40/ep140 병렬).

**항목④ (동일 프로토콜)**: mean **55.65** (ep40 +1.69, P29 +3.44) — 전 도메인 55± (night 54.56). 클래스 이동(ep40→ep140): **Static 27.9→39.1(+11.2)**, Pole 39.5→48.6(+9.1), TrafficSign +5.4, RailTrack→61.0★, TwoWheeler→63.4★, TrafficLight→36.6★ — **retarget 클래스(M0-a 판정) 전부 계보 1위 달성**. ⚠️ 역행: **Water 12.0→5.4**, Wall 8.5→7.1 (test-best 운용점이 rare 클래스 일부를 희생 — val-best ep120와의 클래스 트레이드오프 확인 후보).

**항목①②③ (ep40 대비 수렴 변화)**:
- reliability AUROC 4모달 균형 유지([.85,.78,.87,.70] night) — 보정이 수렴에도 붕괴 안 함(P28/P29의 수렴-후-역보정과 대조).
- drop-Δ: depth 14.2→**10.6**, img 6.1→8.0 — **모달 의존 분산 진행**(robust화). event 여전히 ≈0(−0.11) = 4세대 공통 미해결.
- misallocation(night): RoadLine .79→.74, TrafficLight .62→.55, Wall .54→.42 — 개선되나 여전히 높음 → **learned-router 이식 여지 유지**.
- FUSED eff.rank 6.75→**10.18** — 융합 표현이 수렴하며 더 풍부해짐.
- 모듈 A/B: ep140에서도 bias/cons ≈0.00, gate/veto/calib ±0.3 — **no-op 판정 유지** (모듈 아닌 백본 우위 결론 불변).

## 산출물 맵 (후속 분석용)

- NAS `/drone_nas/drone/analysis_logs/` (HDD2 ISSUE-023 재발로 대체 canonical):
  - `P34_eval_20260713/` — **최신 ep140 풀 산출물** (동일 구성 + A/B 패널)
  - `{P29,P31,P32,P34}_eval_20260712/` — report.md, capability.json, per_domain/(5 cond 로그), per_domain_analysis.md, adapter_health.json, modal_adaptation.{json,md}, feature_stats.{json,md,_pca.png}, module_ablation.{json,md}, **module_diag.json(P29/P31/P32/P34 전부)**, viz/(패널 png, P34는 5 cond), `module_ablation_viz_viz/`(P31 router·P34 gate 전후 패널)
  - `compare_P29_P31_P32_P34_20260712.md` — 4모델 통합표+digest (구 3모델판은 P29 프로토콜 오염으로 폐기)
- lecun 원본: `/SSDb/jemo_maeng/analysis/` + 실행로그 `~/analysis_logs/*.log`
- 재현: `tools/README_seg_analysis.md` 매핑표 (P34는 `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34` 필요 — timm 1.0.24 사이드로드)
