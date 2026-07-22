---
name: det-analysis
description: MemorySAM detection 모델(P29-Det~P39-Det, poongsan)의 표준 분석을 실행하고 판정한다. det ckpt에 대해 "분석해줘 / 클래스별 성능 / 저조도에서 왜 떨어지나 / 모듈이 작동했나"를 물으면 이 스킬을 쓴다. 도구를 새로 짜지 말 것. seg 분석은 seg-analysis 스킬.
---

# Det 표준분석 (MemorySAM)

**원칙: 분석 코드를 새로 짜지 않는다.** 없으면 기존 도구를 확장하고 `tools/README_det_analysis.md`와 이 문서를 갱신한다.

## 0. 세 가지 질문 ↔ 도구

| 질문 | 도구 | 산출 |
|---|---|---|
| 클래스별 성능 | `det_eval_breakdown.py` (D1) | 클래스별 AP/AP50/n_gt + **야간 vs 정상** 분리 |
| 모듈이 일하나 | `det_module_ablation.py` (D2) | 토글 off 시 AP 변화 (없는 토글은 자동 skip) |
| 한 번에 | `det_analysis_pipeline.py --stages D1,D2` | 위 둘 + json/md |

```bash
python tools/det_analysis_pipeline.py --cfg <det cfg> --ckpt <ckpt> \
  --out-dir <out> --gpu <free> --mode test [--toggles auto] [--limit N]
```

`--limit N`으로 스모크. 야간 분리는 poongsan `final` 클립 기본값이며 `--lowlight-clips a,b`로 교체(파일명 substring 매칭).

## 1. 보고 규약 (중요)

- **항상 mAP / mAP50 / mAP75 세 값**을 함께 보고한다. **목표 지표 = mAP50**([[det-report-metrics-convention]], 목표 0.85).
- 모달 구성 명시: poongsan은 img/lidar(egofill)/thermal 조합이 실험마다 다르다. **3모달/RGB-only 등 구성을 반드시 기재**(seg의 모달 표기 의무와 동일).
- 알려진 결과: final-ann에서 **RGB-only ≥ 3모달**(mAP50 기준) — lidar/thermal 추가가 mAP50를 못 올린다([[det-final-ann-modality-ablation]]). 새 모달 주장은 이 기준선을 넘어야 한다.
- 데이터셋 split 혼동 금지: v2 / v3clip / final-ann은 서로 직접 비교 불가(과거 P31-Det가 v3clip 수치로 오비교된 사례).

## 2. 모듈 토글 (auto-skip)

`p36_router_det_off` · `p37b_classtoken_det_off` · `p36_router_off` · `p37a_cefr_off` · `attn_bias_off` / `consistency_off` · `p38_m2f_beta_off` · `p39_modalsrc_off` / `p39_anchored_off` / `p39_trunkexp_off` / `p39_query_off`.
새 모델 모듈이 `seg_model` / `.fusion` / `.m2f`에 스칼라·플래그를 노출하면 `det_module_ablation.make_toggles`에 `_attr(...)` 한 줄 추가하면 끝(seg 쪽 `module_ablation.py`와 동일 계약).

## 3. 판정 규약

seg와 동일한 읽기법을 쓴다: **off 시 하락 = 기여**, `|Δ|`가 노이즈 수준이면 **no-op**, 반대로 off 시 붕괴 수준이면 기여가 아니라 **co-adaptation 의존**. 소물체·야간처럼 **부분집합에서 부호가 갈리는지** 반드시 분해해서 본다(P30-Det가 큰 물체는 동률인데 소물체에서 붕괴한 사례).

## 4. 산출물·기록 (필수 — 여기까지가 완료)

- 원시 산출물 = NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/<model>_det_<YYYYMMDD>/`로 rsync 회수(서버 로컬에 두지 말 것).
- 판정 문서 = repo `.claude_logs/experiments/analysis/<날짜>-<주제>.md`, 그리고 `.claude_logs/experiments/registry.md`의 **Det 표**에 행 추가/갱신 + 필요 시 `status/current.md`.
- det 진단 서사는 `.claude_logs/det/diagnosis-plan.md`가 canonical.

## 5. 서버 운용

빈 GPU(≤2000MiB & ≤10%) 확인 후 실행. 학습과 동거 시 메모리 감시하고 임계 초과 시 **분석만** 중단(학습 보호). 기계적 실행·전송·회수는 sonnet 위임, **판정은 상위 모델**(CLAUDE.md §1.6).
