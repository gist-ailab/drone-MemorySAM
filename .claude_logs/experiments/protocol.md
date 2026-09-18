---
created: 2026-09-18
owns: [judgment-protocol]
updated: 2026-09-18
---

# 판정·측정 규칙 (protocol)

> **역할**: 체크포인트 선택·측정 프로토콜·판정 게이트의 **목록·색인**. 정본은 아래 링크들이고 여기에 본문을 복제하지 않는다(2026-09-18 current.md 재설계, 감사 R2·R3).
> 스냅샷의 수치는 [headline.yaml](headline.yaml) 이 단일 정본이고, 현재 상태는 [../status/current.md](../status/current.md), 판정 이력은 [judgment-ledger.md](judgment-ledger.md).

## 1. 체크포인트 선택

- 1차 규칙 = **학습기 val-best(top1)** 고정(2026-09-16 생각정리 확정). N6 legal-val 재선택 수치는 **규칙 이름을 붙여 보조로만** 병기.
- **test-best 인용 금지** — 철회 사고 2회(P46 λ0.2 57.05, P49.1 57.68)가 이 규칙의 근거다.
- 기준선 척도 열 필수(val-best/final-iter·legal/트레이너 구분) — 카드 §0-1.
- MCubeS 예외 각주: 로더가 test split(102장)을 'val' 이름으로 읽으므로 val-best 선택이 test-best와 동치 — 논문 표에 각주 + final-epoch 보수치 병기(registry P52 MCubeS 행).

## 2. 측정 프로토콜(벤치별)

- **DELIVER legal** = `val.py` 1024·BS1·native GT + 하네스 가드 `tools/eval_harness_guard.py --check` 필수(채점 8파일 SHA256 동결 — ISSUE-033).
- 🔴 **ISSUE-036(2026-09-18)**: 하네스 v1(`val.py`의 nearest, floor 정렬)은 우리 수치를 약 −1.3 낮게 잰다. **2026-09-18 12:10 v2(`tools/legal_rescore_v2.py`, nearest-exact 중심 정렬) 채택** — 근거 = 같은 ckpt 재채점에서 하락 클래스 0·얇은 객체만 +2.7~2.9(측정 산물 확인), 기준선 변환(PIL 중심 정렬)과 정합. 헤드라인·SOTA 거리 비교는 v2 값으로, v1 값은 "(v1)" 꼬리표로만 병기. 카드 쌍 Δ(같은 하네스 v1 양쪽)는 재판정하지 않는다. 잔여: 확정 런 v2 재채점(E1 시드2·3, E13 시드1~3, val 전부)과 v2 래퍼의 가드 매니페스트 등재.
- **MUSES 공식** = `tools/eval_muses_official.py`(native 1080×1920 val 250장). test = Codabench(comp 14005) 제출이며 **user 승인 후에만** 올린다.
- **MCubeS** = 커뮤니티 표준 test split 102장(`semseg/datasets/mcubes.py` 로더 직접 검증).
- **결측·열화 모달 강건성(EMM/RMM/NM)** = `tools/missing_modality_eval.py`(2503.18445 프로토콜).
- **det(poongsan)** = `val_det.py`(score_thresh 0.0 명시 — 기본 0.3이면 기록값 재현 안 됨).
- 학습 @768 / 평가 @1024 해상도 mismatch는 논문·보고에 명시.

## 3. 실험 카드 판정 게이트(정본 = 카드 문서)

- **§0 스크린 규약**: 40ep 완주(val-best)에서만 판정. 통과 = Δtest ≥ +1.0 + 악조건 −0.5 미만 없음. 부기준 24클래스 Δ ≥ +0.5. **중간 epoch 비교 금지**, 조기 kill ep20 −1.5. 24클래스는 매번 원자료에서 재계산 — [decisions/2026-09-07-daily-cycle-experiment-cards.md](../decisions/2026-09-07-daily-cycle-experiment-cards.md) §0.
- **§0-1 확정 런(200ep) 규약**: 게이트 = 3페어 mean Δtest ≥ +1.0 그리고 24클래스 mean Δ ≥ +0.5. 분모(스크린 B0 / 확정 seed821) 이름 병기, 페어 수 표기, 완주 전 예비 재채점으로 최고 갱신 단정 금지 — 같은 문서 §0-1.
- **§0-2 SOTA 거리 게이트·모달리티 정합 비교**: 1차 = 같은 모달 집합(DELIVER DGFusion 56.71·CAFuser 55.6, MUSES DGFusion 79.5, MCubeS StitchFusion 55.9), 2차 = 적은 모달(MM SAM-adapter 57.35·81.07, GtA 82.39). 시드 평균과 단일 최고 병기, 손실 설정(레시피) 열 숨기지 않음 — 같은 문서 §0-2.
- **MUSES 조건별 표본 규칙**: 조건당 25~34장 — 3시드 반복 시에만 실재(DELIVER 조건별 379~380장에는 미적용) — 같은 문서 §5-31b.

## 4. 재현성·보고 규약

- 시드 3개 이상 mean±std 병기. 단일 런 최고만 인용하지 않는다. 무작위 시드 런(56.99 런)은 재현 안 되는 상단 꼬리로 취급.
- 실험 약어(E1·E13·N6·G4 등)는 쓸 때마다 같은 행에 풀어 쓴다(user 지시 2026-09-17) — 풀이 정본 [../meta/experiment-glossary.md](../meta/experiment-glossary.md).
- 진행보고 포맷 = user auto-memory `progress-report-format`(서버별 현황 표 + 남은 run 배치 + 벤치 baseline 표 2블록).
- 판정 주체 = "MMSAM | 생각정리" 세션만(CLAUDE.md §1 세션 분담). 판정 대장 append 규칙 = [judgment-ledger.md](judgment-ledger.md) frontmatter.
- 모든 학습·평가 전 GPU 가용성 확인(빈 GPU = memory ≤2000MiB·util ≤10%)·lecun 배치 금지 — CLAUDE.md 주의사항 참조.
