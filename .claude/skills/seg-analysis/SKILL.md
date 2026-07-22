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
| ① adapter 적응도 | non-RGB가 adapter로 얼마나 적응했나 | D3 `adapter_health.py`(정적 dW) + **D3B `modal_adaptation.py`(on/off Δacc)** + **T0↔T1 activation shift(§0.5)** |
| ② 모달별 피쳐 | per-modal 피쳐 상태(수치+시각화) | D2N `feature_stats.py`(rank/CKA/dead-ch/PCA) + D2·D4 `viz_features.py` — **§0.5 tap×method로 확장** |
| ③ 모듈 전후 | 제안 모듈이 실제로 일하나 | **D5 `module_ablation.py`(토글 off-Δ)** + `module_diagnostics.py` + **T4 피쳐-레벨 no-op 증거(§0.5)** |
| ④ 클래스×도메인 | 어디를 극복해야 하나 | D1 `eval_per_domain.py`→`analyze_per_domain.py`, 다모델은 `compare_models.py` |

## 0.5 피쳐 특성화 (user 지정 2026-07-22) — 분석의 1차 축

**철학 전환**: `module_ablation`의 Δ("어느 모듈이 몇 점")는 **결과지 원인이 아니다**. 먼저 **피쳐 자체를 특성화**해 병목의 물리적 실체(어느 tap에서 정보가 붕괴/중복/포화됐나)를 본 뒤, **그로부터 다음 모델 변경을 제안**한다. "어떤 Δ가 정확한가"를 쫓지 말고, encoder→adapter→fusion→제안모듈→head의 **피쳐가 무엇을·얼마나·어떻게 담는지**를 통계로 답한다.

### Tap 지점 (피쳐를 어디서 뽑나 — 파이프라인 순서, model.py 실측)
| tap | 무엇 | stash | 실행 |
|---|---|---|---|
| **T0** encoder raw per-modal | ViT+LoRA 출력 embed(모달별, full-dim 예 1024). **adapter가 인코더 내부**라 T0가 곧 post-adapter — 별도 T1 텐서 없음 | `_last_per_modal_feats` | ✅ 있음(feature_stats `mod*`) |
| **T2** fusion-input | **per-modal FPN 없음** — 인코더가 fusion에 직결. 즉 T2 ≡ T0 | (T0와 동일) | ✅ |
| **T3** fusion-output | `self.fusion(...)` 직후 fused(제안 모듈 잔차 이전) | `_last_fused_postfusion` | ✅ 추가됨(`FUSED_pf`) |
| **T4** fused-level 모듈 효과 | **CEFR·trunk_exp**(fused를 바꾸는 모듈)의 순효과 = **T3→T5 차분**. ⚠️ router/classtoken/m2f/arbiter는 **logit-level**(T5 이후 `_decode`·logits에서 작동)이라 여기 안 잡힘 → 그건 `module_ablation` Δ로 | (stage CKA `FUSED_pf~PREHEAD`) | ✅ |
| **T5** pre-head | `_decode` 직전 fused(모든 모듈 잔차 이후) | `_last_fused_prehead` | ✅ 추가됨(`PREHEAD`) |
| **FUSED** decode 피쳐 | `_decode` 반환 m_feat(head 직전 표현) | model return[1] | ✅ 있음(`FUSED`) |

→ **adapter 판정(T1 없음)**: T0 자체가 adapter 포함. adapter "효과"는 **LoRA on/off diff**(D3B `modal_adaptation`) 또는 정적 dW(`adapter_health`)로 본다 — T0 단일 tap에서 분리 불가. **T4(fused-level 모듈)**: `FUSED_pf~PREHEAD` stage CKA≈1 = **CEFR·trunk_exp가 fused를 안 바꿈**(피쳐-레벨 no-op). ⚠️ 이것만으로 router/classtoken/m2f까지 no-op이라 하지 말 것 — 그들은 logit-level이라 이 CKA에 안 나타난다(반드시 `module_ablation` Δ와 병행). 도구는 `feature_stats.py`를 **확장**(`--no-extra-taps`로 T3/T5 토글)했지 새로 짜지 않았다(§7).

### Method matrix (각 tap에 적용 — 산출 → 통계적 의미)
| 방법 | 산출 | 통계적 의미 | 현 도구 |
|---|---|---|---|
| activation 분포 | mean/std·‖f‖, per-ch mean\|act\|, **histogram·sparsity(%≈0)·kurtosis** | 용량 사용/포화/희소 | mean\|act\|·dead까지만 → **분포 지표 확장 대기** |
| dead channels | 데이터셋 전역 mean\|act\|≈0 채널 수 | 낭비된 용량 | ✅ `feature_stats` A |
| effective rank | participation ratio(cov eig) | 저차원 붕괴 vs 풍부 | ✅ `feature_stats` B |
| PCA | **설명분산비(top-k)·내재차원(90%분산 도달 k)** + 2D scatter | 정보의 실제 차원, 모달 분리도 | scatter만 → **정량화 확장 대기** |
| CKA | 모달간·**stage간** linear CKA | 중복(상보성 없음)/stage가 뭘 바꿨나 | ✅ 모달간만 → stage간 확장 대기 |
| cos(fused, modal) | 융합 기여 방향 | fusion이 어느 모달로 기움 | ✅ `feature_stats` D |

### 해석 규약 (피쳐 → 진단 → 처방)
- **eff_rank↓ + dead↑** = 그 tap에서 피쳐 붕괴(정보 부족). **T0 vs T1**로 원인이 adapter인지 인코더인지 가른다.
- **adapter 판정(동적)**: T0→T1 activation shift(‖Δ‖·낮은 CKA)가 크면 adapter **살아있음**. 정적 dW(`adapter_health`) + 동적 shift **둘 다** 본다. rank 낮아도 shift 크면 "죽음"이 아니라 저rank 압축.
- **CKA(modalA,modalB)→1** = 상보성 없음 → 그 모달 추가/유지가 무의미(예: event vs lidar depth 잉여 — [[p32-corroboration-rbma]]).
- **T2 vs T3 CKA** = "fusion이 실제로 뭘 섞었나". **T3 vs T5(또는 T4) CKA≈1** = 제안 모듈이 pre-head 피쳐를 **안 바꿈** → **피쳐-레벨 no-op 증거**(Δ-ablation과 교차검증).
- **평균 금지**(§2): 위 지표 전부 조건/클래스/샘플 축으로 펼쳐 본다.

### 모델 제안으로 연결 (framework 목적)
진단 → 처방 매핑 예:
- 특정 모달 T2 eff_rank 낮음 & CKA 높음 → 그 모달 **인코더/adapter 재설계 또는 드롭**.
- T3가 특정 모달 cos만 지배 → **fusion 편향** → 게이팅/정규화 조정.
- 제안 모듈 T4 출력이 T3와 CKA≈1 & 분포 무변화 → **모듈이 새 정보 안 만듦** → 재설계(피쳐-레벨 근거로 제안).

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
- **유해(harmful)**: `Δ ≤ −0.5`(끄니 **올라감**) → 모듈이 순손해. det 스위트 `ACTIVE(−)`와 동형. 단발성 노이즈 배제는 **조건 축 다수 조건에서 −부호 일관** 확인(단일 조건 −는 §"평균 금지"로 재검).
- off-Δ가 **+20↑ & agreement<0.8** → 기여가 아니라 **co-adaptation 의존**(단일 실패점). "라우터가 +34 기여"로 읽지 말 것.
- **평균에 속지 말 것**: 조건별·클래스별로 부호가 갈린다(예: V5 query가 주간 +1.0인데 야간 −0.28). 항상 조건 축으로 펼쳐 본다.
- 라우팅류는 전역 평균이 uniform이어도 per-class로 갈라 재확인 → `probe_cefr_routing.py` 패턴(eval 스태시 훅 + per-class 집계). **샘플-조건부 게이트(quantile 감쇠 등)는 per-sample 발동률**까지 집계(전역 평균이 숨긴다).
- 🔴 **토글이 목록에 안 뜨는 것과 no-op은 다르다.** `make_toggles()`는 모듈이 **실제로 결선돼 있을 때만** 토글을 등록한다(조건부):
  - `p38_m2f_off`는 **arbiter(P39 `core.arb_lambda`)가 없을 때만** 등록 — P39 계열에서는 β 경로 자체가 미사용이라 토글이 아예 안 뜬다. 이때 "no-op"이라고 쓰면 오판이다(뜨지 않음 = 그 결선이 존재하지 않음).
  - `p39_*`도 arbiter 존재 시에만 등록(플래그 attr는 구세대에도 있어 attr 존재만으론 판단 불가).
  - `p37_cefr_off`는 `fusion.cefr`가 있을 때만.
  → 실행 로그의 `available=[...] skipped=[...]` 줄을 **반드시 보고**하고, skip 사유를 "모듈 부재"로 적을 것.
- adapter 판정: Δacc가 크면 adapter는 **살아있다**. 여기에 rank가 낮으면 "죽음"이 아니라 **저rank 압축**이며, 융합이 그걸 쓰는지는 `drop_modality_dmiou`로 따로 본다. (동적 activation shift = §0.5 T0↔T1)
- ckpt 선택: 논문 수치는 **val-best만**([[seg-report-sota-gap]]). 단 이 계보는 **val↔test 순위 역전**이 관측됐으므로 분석 판정은 test/조건 평균을 함께 본다.

### 2.1 붕괴 런(학습 발산) 진단 프로토콜
val이 정점 뒤 급락한 ckpt(예: ep8 62 → ep12 45)는 **보고용이 아니라 진단 검체**다.
1. **두 시점 ckpt**(붕괴 전 val-best + 붕괴 후)를 모두 확보 — 차분이 핵심 신호.
2. **원인 모듈 off로 회복 테스트**: 붕괴-후 ckpt에 의심 모듈을 `module_ablation` off로 걸어 mIoU가 회복되면 **런타임 근인**, 회복 안 되면 **가중치 오염**(adapter/fusion이 co-adapt).
3. **피쳐 차분**(§0.5): 두 시점 T0~T5의 eff_rank·dead·activation 분포 대조 — 어느 tap에서 붕괴가 시작됐나.
4. **대조군**: 의심 모듈만 뺀 런이 같은 지점에서 **안 붕괴**함을 D1로 확정해야 귀속 성립(공통 학습 역학 배제). 예: RCA 붕괴는 P39.1 대조로 확증([[p34-reliadino]] 계열).

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
