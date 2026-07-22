---
name: seg-analysis
description: MemorySAM seg 모델(P29~P39+, DELIVER·MUSES)의 표준 분석을 실행하고 판정한다. 새 ckpt가 나왔을 때 "분석해줘 / 모듈이 작동했는지 / 어디가 문제인지 / 왜 떨어졌는지"를 물으면 이 스킬을 쓴다. 도구를 새로 짜지 말고 여기 매핑대로 실행할 것.
---

# Seg 표준분석 (MemorySAM)

> detection 모델 분석은 **`det-analysis` 스킬**을 쓴다(도구·판정 규약이 다름).

**원칙: 분석 코드를 새로 짜지 않는다.** 필요한 게 없으면 기존 도구를 **확장**하고 이 문서와 `tools/README_seg_analysis.md`를 갱신한다.

## 0. 표준 분석항목 (user 지정 2026-07-12) — 매번 이 4개를 답한다

| 항목 | 질문 | 도구/스테이지 |
|---|---|---|
| ① adapter 적응도 | non-RGB가 adapter로 얼마나 적응했나 | D3 `adapter_health.py`(정적 dW) + **D3B `modal_adaptation.py`(on/off Δacc)** |
| ② 모달별 피쳐 | per-modal 피쳐 상태(수치+시각화) | D2N `feature_stats.py`(rank/CKA/dead-ch/PCA) + D2·D4 `viz_features.py` |
| ③ 모듈 전후 | 제안 모듈이 실제로 일하나 | **D5 `module_ablation.py`(토글 off-Δ)** + `module_diagnostics.py` |
| ④ 클래스×도메인 | 어디를 극복해야 하나 | D1 `eval_per_domain.py`→`analyze_per_domain.py`, 다모델은 `compare_models.py` |

## 1. 실행

```bash
# 공통 ENV (yeon/lecun): timm 사이드로드 필수
PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:<repo>/semseg/models/sam2 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1

# DELIVER 4모달 (img/depth/event/lidar) — test split
python tools/seg_analysis_pipeline.py --cfg <cfg> --model_path <ckpt> \
  --dataset-root <DELIVER> --out-dir <out> --gpu <free>

# MUSES 3모달 (img/lidar/event) — val split 필수(test GT 비공개)
python tools/seg_analysis_pipeline.py --cfg <cfg> --model_path <ckpt> \
  --dataset-root <MUSES> --split val \
  --conditions clear,fog,rain,snow,day,night --viz-case night \
  --out-dir <out> --gpu <free>

# MUSES 조합 셀(리더보드 채점 단위) — 실패 셀 정밀 진단
--conditions fog_night,clear_night,snow_night,rain_night,fog_day
```

**모달 구성은 config `DATASET.MODALS`가 단일 출처**다. 파이프라인이 그 값을 `adapter_health --modals`로 전달하므로 per-modality dW 라벨이 자동으로 맞는다(직접 호출 시엔 `--modals`를 반드시 줄 것 — 안 주면 DELIVER 순서로 오표기).

| 벤치 | 3모달 | 4모달 |
|---|---|---|
| MUSES | `img,lidar,event` | `img,lidar,event,radar` (P34 4모달 test 78.256 = 3모달 78.979 **−0.72**) |
| DELIVER | — | `img,depth,event,lidar` |

**4모달을 분석할 때 추가로 답할 것**: ① 추가 모달(radar/event)이 D3B Δacc·drop-modality에서 값을 하는가(dead면 제거 근거) ② 3모달 대비 기존 모달의 rank·기여가 깎였는가(모달 추가가 표현을 압축시키는지) ③ CKA로 정보 잉여인가.

소요: 조건당 ~10분, 전체 80~120분. 8스테이지(D3/D1×2/module_diag/D2N/D3B/D5/viz).

## 2. 판정 규약 (수치 → 결론)

- `miou_delta_when_off` **+ = 모듈 기여**. `|Δ|<0.5 & pred_agreement>0.99` → **no-op**(신규 모듈 조기 탈락).
- off-Δ가 **+20↑ & agreement<0.8** → 기여가 아니라 **co-adaptation 의존**(단일 실패점). "라우터가 +34 기여"로 읽지 말 것.
- **평균에 속지 말 것**: 조건별·클래스별로 부호가 갈린다(예: V5 query가 주간 +1.0인데 야간 −0.28). 항상 조건 축으로 펼쳐 본다.
- 라우팅류는 전역 평균이 uniform이어도 per-class로 갈라 재확인 → `probe_cefr_routing.py` 패턴(eval 스태시 훅 + per-class 집계).
- adapter 판정: Δacc가 크면 adapter는 **살아있다**. 여기에 rank가 낮으면 "죽음"이 아니라 **저rank 압축**이며, 융합이 그걸 쓰는지는 `drop_modality_dmiou`로 따로 본다.
- ckpt 선택: 논문 수치는 **val-best만**([[seg-report-sota-gap]]). 단 이 계보는 **val↔test 순위 역전**이 관측됐으므로 분석 판정은 test/조건 평균을 함께 본다.

## 3. 새 모델이 오면

1. 모듈 토글을 `module_ablation.py::make_toggles()`에 등록(+`seg_analysis_pipeline.py` 기본 토글 문자열). 미등록 시 조용히 skip된다.
2. 필요하면 모델 forward에 eval 스태시(`_last_*`) 추가 — 학습 영향 0으로.
3. **학습 직후 조기 즉검**: 첫 val-best 스냅샷에 `module_ablation.py`만 돌려 no-op 여부를 먼저 판정한다(완주 후 발견 금지).

## 4. 산출물·기록 (필수)

- 원시 산출물 = NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/<name>_<YYYYMMDD>/`로 rsync 회수.
- 판정 문서 = repo `.claude_logs/experiments/analysis/<날짜>-<주제>.md` + `experiments/registry.md` 행 + `status/current.md` 갱신.
- **모달 수(3모달/4모달) 명시 의무**(user 지정 2026-07-21).
- 그림: dataviz 팔레트(#2a78d6/#1baf7a/#eda100/#008300/#4a3aa7), 폰트 `Noto Sans CJK JP`, json 파싱 스크립트를 산출물 폴더에 함께 저장(재생성 가능하게).

## 5. 서버 운용

- 실행 전 빈 GPU 확인(≤2000MiB & ≤10%). 학습과 동거 시 메모리 감시 + 임계 초과하면 **분석만** kill(학습 보호).
- 분석은 서버의 격리 worktree(`/SSDb/jemo_maeng/src/dm_analysis`)에서 `origin/develop` detach로 실행 — 타 세션 체크아웃을 건드리지 않는다.
- 기계적 실행·전송·회수는 sonnet에 위임하고, **판정은 상위 모델이 한다**(CLAUDE.md §1.6).

## 6. 이미 반증된 것 (재확인 불필요, 새 모델에서 재발만 체크)

RBMA/CoRB attn-bias(4세대 무효) · reliability gate/calib/veto(3세대 no-op, 일부 조건에선 유해) · CEFR per-class 라우팅(미분화) · 무감독 threshold 마스크 게이트(P37b 영구 random) · zero-init 잔차 결선 일반(β·σ(a) 모두 "열리다 만" 고착). 상세 = `.claude_logs/experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md`.
