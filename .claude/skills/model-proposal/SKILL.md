---
name: model-proposal
description: MemorySAM 차기 모델 제안(P39.1/P40 계보의 역할 승계). "다음 모델 제안해줘 / PXX 실패했는데 어떻게 개선 / 점수 올릴 구조" 류 요청 시 이 스킬을 쓴다. 딥리서치(멀티에이전트) + 기존 관련연구 자산 + 분석 결과를 근거로, 게이트 사전 등록된 제안 문서를 만든다. fable/opus 세션 전용(판단·설계); 실행 잡무는 sonnet 위임.
---

# 모델 제안 (MemorySAM) — 역할 승계 스킬

**역할 (user 지정 2026-07-22)**: ① fable 딥리서치 ② 기존 관련연구 자산(로그+옵시디언 볼트) ③ 기존 모델 val/test 분석 결과 — 세 근거를 교차해 **벤치마크 점수를 올릴 모델을 제안**한다. 제안은 반드시 판정 게이트가 사전 등록된 문서로 남긴다. (실행 사례: P38→실패-키→P39 DPC→분석→P39.1/P40 RCA, 2026-07-20~21)

## 0. 세션 시작 시 읽을 것 (순서대로)

1. `.claude_logs/00_INDEX.md` → `status/current.md` — 현재 최선 모델·게이트 수치 확인
2. **실패-키 canonical**: `.claude_logs/experiments/analysis/2026-07-XX-failure-keys-*.md` 최신본 — 구조적 문제·반증된 경로·실증된 경로 목록. **제안의 출발점은 항상 이 문서다.**
3. 최신 표준분석: `experiments/analysis/` 최신 날짜 문서들 (+원시 산출물 = NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/`)
4. 관련연구: `research/novelty-and-related-work.md`(canonical 비교표) + 옵시디언 볼트 `/nas_jm/Research/26_MultimodalSeg`(원본; repo `research/vault/`는 사본)
5. `experiments/plan.md` — 진행/대기 실험과 충돌 확인
6. `issues/issues-and-fixes.md` 인덱스 표 — 유효한 판정인지(예: ISSUE-025/026이 과거 판정을 보류시킴)

## 0.5 🔴 선행 조건 — 근거 없이 제안하지 않는다 (user 지정 2026-07-22)

대상 P 모델(개선하려는 직전 세대)에 대해 **작업 시작 전** 아래를 순서대로 확인하고, 하나라도 없으면 **제안을 멈추고 그것부터 확보**한다:

1. **분석 리포트 존재 확인**: `.claude_logs/experiments/analysis/`에 해당 P 모델의 표준분석(D1~D5) 문서가 있는가? (원시 산출물 = NAS `analysis_logs/<P>_eval_*/`)
   - **없으면 → 제안 금지. 분석을 먼저 요청**한다: seg는 `seg-analysis` 스킬(분석 세션), det는 `det-analysis` 스킬. 요청 후 결과가 나올 때까지 제안 작업을 시작하지 않는다.
2. **val/test 수치 존재 확인**: 해당 모델의 학습 로그 val 궤적 + (가능하면) test/공식 제출 수치가 registry(`experiments/registry.md`)·log(`experiments/log.md`)에 기록돼 있는가? 인퍼런스 산출물(per-domain/제출 결과)이 있는가?
   - **없으면 → 먼저 인퍼런스/평가를 실행**(빈 GPU에서 eval — 학습 불필요)해 수치를 확보한 뒤 진행한다.
3. 확보된 수치·분석의 **유효성 검사**: `issues/issues-and-fixes.md` 인덱스에서 그 수치가 이후 발견된 버그(예: ISSUE-025 radar, ISSUE-026 aug)로 보류/오염되지 않았는지 확인. 보류된 수치는 근거로 인용 금지.

**Why**: 제안의 품질은 진단의 품질을 넘지 못한다 — 분석 없는 제안은 P33 이전 세대의 실패 패턴(추측 기반 설계)이다. P39.1/P40이 성립한 것은 실패-키 문서와 표준분석이 먼저 있었기 때문.

## 1. 절차 (P39.1/P40에서 검증된 6단계)

1. **분석 정독 → 규칙 변환**: 실패-키 각 항목을 "설계가 해야/하지 말아야 할 것"으로 표 변환. 분석 수치를 그대로 인용(재측정 금지 — 필요하면 seg-analysis 스킬로 분석 세션에 요청).
2. **딥리서치 (멀티에이전트, 병렬 3축이 기본)**: 문제 기제별로 에이전트를 나눈다 —
   - 기제 축: 관찰된 실패의 문헌 기제·대응책 (예: rank collapse → VICReg/log-det/rsLoRA)
   - 노벨티 축: 제안하려는 기제의 최근접 선행 수색 + **정직한 노벨티 판정** ("first X" 주장 가능 여부, 미점유 조합 축)
   - 물리/벤치 축: 센서 물리·경쟁 논문 per-condition 수치로 헤드룸 실측 (기대치 캘리브레이션)
   각 에이전트에 arXiv ID 인용 의무 + "final text = 데이터" 지시. 결과는 제안 문서에 근거로 인라인.
3. **제안 작성** — 필수 요소:
   - 진단↔문헌 대응 표 (우리 실측 ↔ 인용 근거 ↔ 함의)
   - 변경 목록: 항목별 근거 키 + **전 항목 토글 가능** (ablation 분해 보장)
   - **게이트 사전 등록**: 현행 최선 수치 대비 + 조기(ep30) kill 기준 + falsifiable 예측
   - 노벨티 포지셔닝: 최근접 선행 명시 + 차별 축 (과장 금지 — 리뷰 방어 기준)
   - 실행 계획: 선행 분석(학습 0으로 가능한 것 먼저) → 스모크 → 슬롯
4. **등재**: 제안 문서 = `.claude_logs/decisions/YYYY-MM-DD-<이름>-proposal.md` → develop push → `experiments/plan.md` 대기열 행(EPOCHS·게이트 명시, sonnet 위임 가능).
5. **구현 시**: `.claude_logs/meta/conventions.md`의 🔴 **코드 검수 파이프라인** 의무 (fresh-eyes 렌즈 7종 + 스모크 grad/등가 assert + 로더 실측 + ep30 토글 즉검). 대규모 변경은 멀티에이전트 전수조사(발견→반증검증 2단).
6. **판정 후 루프백**: 결과가 나오면 실패-키 문서를 갱신하고 1로 돌아간다.

## 2. 제약 카탈로그 (제안이 위반하면 즉시 기각되는 것들)

- **반증된 경로 재시도 금지** (실패-키 문서 C절이 단일 출처; 2026-07-21 기준: attn-bias 계열·gate/calib/veto 추론 재가중·CEFR class-expected routing·무감독 threshold 게이트·수동 zero-init 잔차 결선·conv head 즉시 대체)
- **키1**: 새 모듈은 주 손실을 직접 받거나 기존 경로와 경쟁시킬 것 — "zero-init 잔차로 살짝 얹기"는 4연속 사망으로 반증 완료 (게이트 파라미터는 tanh(0.1)처럼 gradient가 즉시 흐르는 init)
- **공정성**: 헤드라인은 PhysAug 금지·TTA 금지·val-best ckpt 규칙([[seg-report-sota-gap]])·augmentation은 DGFusion 정합(ColorAugSSD — ISSUE-026 수정본)
- **노벨티 방침 (user)**: 외부 신호(CLIP text·GT-depth·조건 라벨) 불사용 — 모델 내부 신호만. DGFusion/CAFuser와 유사 구조 금지
- **단일 모델**: DELIVER·MUSES에 같은 아키텍처 (데이터셋 적응은 학습된 모듈로만)
- 성능 원천은 백본+per-modal LoRA(관용 기법, 노벨티 주장 금지 — MLE-SAM 선행). 논문 novelty는 주장·근거 구조에서 나옴

## 3. 현행 게이트 (2026-07-22 기준 — 갱신되면 이 절을 고칠 것)

| 벤치 | 게이트 | 비고 |
|---|---|---|
| DELIVER | P36 fair val 67.74 / test 55.62 + thin-class(Wall≥13/Water≥9.5/RailTrack≥62) | ⚠️ ISSUE-026으로 P37+~P39 DELIVER 판정 보류 — P39.1이 픽스 후 첫 클린 런, 완주 시 게이트 재설정 |
| MUSES | val 82.22(P38) / 공식 test 79.025(P38-m2f) + fog_night ≥74 | radar 포함 실험은 ISSUE-025 픽스 후만 유효 |
| 공식 목표 | DELIVER val≥66.51/test≥56.71 (DGFusion), MUSES 리더보드 1위 = GtA 82.39(카메라 단독) | memory `official-research-goals` |

## 4. 협업 규약

- 잡무(서버 상태·rsync·로깅·git 기계 작업) = sonnet 위임, 판단·설계·수치 판정 = 이 세션. GPU 규칙: 빈 GPU(≤2000MiB·util≤10%)만, plan.md 예약 우선.
- 분석이 필요하면 새로 짜지 말고 **seg-analysis 스킬**(분석 세션)로 요청. 학습 기동은 plan.md 등재 후 launch 절차(검증 기준: iteration 전진·전 GPU 활성·에러 0·ETA 산출).
- 딥리서치 산출·제안 근거는 옵시디언 볼트가 아니라 **repo decisions/**에 (볼트는 리서치 콘텐츠 원본, repo는 프로젝트 로그 — CLAUDE.md 위치 규칙).
