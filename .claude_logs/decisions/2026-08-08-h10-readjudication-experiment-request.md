---
created: 2026-08-08 17:00
author: fable (MMSAM discussion 세션)
type: 실험 의뢰서 (다른 세션이 이 문서만 읽고 실행 가능해야 함)
status: 의뢰 등재 — 실행 세션 미배정
---

# 실험 의뢰서 — H10 재판정: 인스턴스 감독 미니 실험

> **의뢰 취지**: 가설 원장 [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) 유일의 ⚠️ 미결 항목 H10을 실측으로 닫는다.
> P48 폐기 판정(2026-08-06)은 **게이트 적용 시점 오류 논란**이 있다 — 게이트(things PQ>30)는 "인스턴스 감독 **학습 후**" 모델에 적용해야 하는데 학습 전 모델에 적용됐다([experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md](../experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md) §3). 이 의뢰는 그 논란을 **올바른 시점의 미니 실험**으로 종결한다.

## 실행 스펙

| 항목 | 내용 |
|---|---|
| Base ckpt | **P47-MUB D-1 ep172** (MUSES 4모달 val 82.58) — config `configs/hpca100-muses_rgbelr_P47_d1_dgfproj_4modal.yaml`. ckpt 위치는 registry 행/기동 세션 기록에서 확인(실행 세션 몫) |
| 학습 | 기존 M2F 쿼리 헤드에 **인스턴스 단위 감독**(Hungarian matching을 per-instance mask로 — 설계는 [2026-08-05-p48-instance-supervision-proposal.md](2026-08-05-p48-instance-supervision-proposal.md) S3 참조) 추가, **LoRA+head만 10~20ep** 미세조정. 그 외 레시피 base 동결 |
| 선행조건 | MUSES `gt_panoptic` **train** split 존재 확인 (val은 08-06 PQ 채점에 사용됐으므로 확보돼 있음; train 미확보면 다운로드부터 — 용량 확인 후) |
| 측정 | `tools/eval_pq.py` `--geometry native`, MUSES val 250장 — **things PQ** (2026-08-06 측정과 동일 프로토콜, 비교선 = 학습 전 22.87) |
| 자원 | 4090 1~2장(jarvis), MUSES 1024² BS1, 총 ~반나절 |
| 실행 위임 | 기동·조회 = sonnet 가능 / **판정 = 상위 모델(fable/opus)** |

## 구현 (실행 세션이 결정할 것 — 보완 2026-08-08 18:00)

- **코드 변경 범위**: `MaskQueryLiteHead` 학습 타깃을 클래스단위 마스크 → per-instance 마스크(Hungarian per-instance matching)로 바꾸는 **학습 전용 토글** 1개 + MUSES panoptic GT 로더. 추론 경로 무변경(기존 `panoptic_inference()` 재사용). 설계 상세 = P48 제안서 S3.
- **구현 주체**: 이 의뢰서를 집은 세션이 직접 짜지 말고 **워커 위임**(GLM.md 규약 — 위임 직전 user에게 워커 선택 확인: labcode 권장, 정확도 중요). 지시문에 이 의뢰서 링크 + P48 S3 + "추론 불변·토글 가드" 요구사항을 명시.
- **검수 게이트 (기동 전 필수)**: code-review 규약 준수 — fresh-eyes 검수 + 스모크(grad 흐름·토글 off 시 base와 등가·로더 실측). 통과 전 기동 금지.

## 총 ETA

구현(워커 0.5~1일) + 검수(반나절) + 학습(4090 1~2장 ~반나절) + PQ 평가(~1h) ≈ **2일**.

## 폴백 (사전 정의 — 즉석 판단 금지)

- **gt_panoptic train split 미제공/미확보** → 의뢰 중단, 코디네이터 보고. (semantic GT에서 connected-component로 유사 인스턴스를 지어내는 대체는 **금지** — D2 검증이 오염됨.)
- 학습이 base 대비 semantic val을 −1.0 이상 훼손 → 조기 kill 후 그 사실 자체를 기록(인스턴스 감독과 semantic의 상충 증거).

## 사전 등록 게이트 (시점 = **인스턴스 감독 학습 완주 후**, P48 사태 재발 방지용으로 시점 명기)

- things PQ **> 33.6** (D2가 산출한 클래스단위 타깃 상한) → **D2 확인 + 쿼리 잠재력 실재** → P48 재개 검토 (단 PQ 축은 SOTA 비교 불가·논문 limitation 지위 유지 — 재개해도 우선순위는 코디네이터 판단)
- things PQ **≤ 33.6** → **H10 ✗ 확정** (감독을 줘도 못 깨움 = 능력 신설 필요) — 원장 H10 행을 ✗로 갱신하고 종결

## 결과 기록처 (실행 세션 의무)

1. `experiments/registry.md` 행 추가 (실험ID 예: `jarvis_muses_h10_instsup_mini`)
2. `experiments/analysis/2026-08-XX-h10-instance-supervision-readjudication.md` 신설 (수치 원표 + 게이트 적용)
3. [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) **H10 행 판정 갱신** + 이 문서 status 갱신
4. 완료 통보: `status/history-2026H2.md` 엔트리 (다음 세션이 자동 인지)

관련: [2026-08-05-p48-instance-supervision-proposal.md](2026-08-05-p48-instance-supervision-proposal.md) · [experiments/analysis/2026-08-06-pq-first-measurement-p48-gate.md](../experiments/analysis/2026-08-06-pq-first-measurement-p48-gate.md)
