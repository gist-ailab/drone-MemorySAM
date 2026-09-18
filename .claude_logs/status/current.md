---
legacy_id: 01
legacy_file: 01_project_status.md
split_from: 01_project_status.md
moved: 2026-07-08
owns: [status-snapshot]
updated: 2026-09-18
---

> **역할**: 현재 상태 스냅샷 — 네 블록(벤치 표·지금 도는 것·블로커·판정 대기)만 담는다(상한 60줄, 2026-09-18 재설계 — 감사 R2). 이력·경위는 [history-2026H2.md](history-2026H2.md), 날짜 엔트리 적층 금지.
> 헤드라인 수치 정본 = [../experiments/headline.yaml](../experiments/headline.yaml) · 판정·측정 규칙 = [../experiments/protocol.md](../experiments/protocol.md) · 판정 이력 = [../experiments/judgment-ledger.md](../experiments/judgment-ledger.md)

# 프로젝트 현황

## ① 벤치별 최선 — 표는 생성물(`tools/gen_headline_tables.py`, 멱등·`--check` 검사)

<!-- headline:begin -->
| 벤치·분할 | SOTA(1차 같은 모달 / 2차 적은 모달) | 우리 최고(값·근거·규칙·상태) | 격차 | 근거 |
|---|---|---|---|---|
| DELIVER · test | DGFusion 56.71 / 2차 MM SAM-adapter 57.35 | **56.24** · best 56.72(legal v2) — 3시드 평균(감시 세션 보고값, std 미수신) · best = 시드2(56.72) · 시드별 {55.97, 56.72, 56.03} · 학습기 val-best(top1) · 유효(v2, 3시드) | mean −0.47 · best +0.01(DGFusion 56.71) · mean −1.11 · best −0.63(MM SAM-adapter 57.35) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) · [카드 W38 2026-09-18 legal v2 채택 항목](../decisions/cards/2026-W38-verdicts.md) |
| 〃 · val | CAFuser-CAA 68.79 / 2차 MM SAM-adapter 69.6 | **69.51** · best 69.68(legal v2) / 67.89(v1 최고 시드3) — 3시드 평균(감시 세션 보고값, std 미수신) · best = 시드3(69.68) · 시드별 {69.39, 69.45, 69.68} · 학습기 val-best(top1) · 유효(v2, 3시드) | mean +0.72 · best +0.89(CAFuser-CAA 68.79) — 같은 4모달 계열 val 1위 · mean −0.09 · best +0.08(MM SAM-adapter 69.60) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) |
| MUSES · test | DGFusion 79.5 / 2차 GtA 82.39 | **79.29** ±0.71 · best 79.788(MUSES 공식) — 2시드 평균 ± 표준편차 · best 단일 런 병기 · 학습기 val-best(top1) · 유효 | mean −0.21 · best +0.29(DGFusion 79.5) · best −2.60(GtA 82.39) · best −1.28(MM SAM-adapter 81.07) | [judgment-ledger 2026-09-18 행](../experiments/judgment-ledger.md) · [analysis 시드20260825 판독](../experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md) |
| MCubeS · test | StitchFusion 55.9 | **58.07** ±0.49(커뮤니티 test 102장) — 3시드 {57.93, 57.67, 58.62} 평균 ± 표준편차 · 학습기 val-best(registry N4 행 표기) · 유효(1등 확정) | +2.17(StitchFusion 55.9) · +3.42(published 최고 Mul-VMamba 54.65) | [registry N4 행(yeon_mcubes_rgbadn_P39_1_rank)](../experiments/registry.md) |
| det(poongsan) | — | **0.9321**(val_det.py 재현) — ep6 · 재현 패키지 기록값 · 유효(종결 국면) | 목표 0.85 대비 +0.08 — 달성 완료 | [det 인증 문서 D1 스윕 표](../det/det-cert-D1-realtime.md) |
| MULTIAQUA | — | **82.10**(챌린지 서버) — P9 Val 93.29/Test 70.91(재제출 #16710) · P22 Val 93.42/Test 70.77(#16932) · 챌린지 제출 마스크 기준 · 유효(종료·고정) | —(챌린지 종료·고정) | [registry MULTIAQUA 표](../experiments/registry.md) |
<!-- headline:end -->

> 상태 표기 = 유효 / 보류 / 철회 / 불일치. 수치·근거·병기값은 headline.yaml 이 정본이다.

## ② 지금 도는 것

- 실행 중 런·대기열·GPU 점유·ETA = [../experiments/plan.md](../experiments/plan.md) "실행 중"·"GPU 예약·점유 현황" 표(여기 복제 금지).
- 링크: 상황판 artifact https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 · 노션 논문 페이지 https://app.notion.com/p/gistailab/Drone-Object-Detection-for-RGB-IR-Fusion-33d05310a165408ab0b8ec4427d1fe2c

## ③ 열린 블로커 (2026-09-18)

1. 🔴 **ISSUE-036** legal 하네스 재샘플 floor 정렬 편차(우리 수치 약 −1.3 편향) — 2026-09-18 12:10 **legal v2(nearest-exact) 채택**, DELIVER 헤드라인 = 56.39(DGFusion −0.32, SOTA 미달). 잔여: E1 시드2·3·E13 시드1~3 test 재채점 5건 + val 재채점(감시 세션 대기열), v2 래퍼 가드 등재. [../issues/issues-and-fixes.md](../issues/issues-and-fixes.md) 상단 표.
2. **ISSUE-034** eval 예측 덤프 파일명 평탄화 — test 1270/1897장만 남음(jarvis 12런). 덤프를 쓴 과거 이미지별 분석 점검 필요.
3. **ISSUE-035** 헤드라인 ckpt 경로 기록 — 56.99 런 ckpt는 NAS 정본 이관 완료(registry 표기), 이슈 표 갱신 대기.
4. **lecun 배치 금지**(user 2026-09-17, `scripts/servers.conf` policy off 유지).
5. **hpca100 공유 볼륨 감시** — 09-15 Errno 28로 6런 사망 전례([../experiments/plan.md](../experiments/plan.md) GPU 표 비고).

## ④ 다음 판정 대기 (판정 = "MMSAM | 생각정리" 세션)

- legal v2 잔여 재채점 5건(E1 중간층 4탭 카드 시드2·3, E13 4탭+센서별 prototype 시드1~3)·val 2건 회신 → 헤드라인 표·판정 대장 갱신(카드 쌍 Δ 는 같은 하네스라 판정 유지).
- MUSES 200ep 풀 런 셋 완주(09-18~19) — E7(PhysAug-off 기준선) 공식 val 81.88 완료·E1M(4탭 읽기 MUSES판) 81.70 완료·E13M(4탭+센서별 prototype MUSES판) 09-18 23:40 → 공정선 헤드라인 test 제출 판정(user 승인 후).
- E17(E1 4탭 읽기 + 고해상도 세부 가지) 40ep 스크린 G1~G4 판정 — 09-18 13:05 완주 예정.
- E-LoRA 3판(센서별 r16 / 완전공유 r16 / 공유 r8+센서별 잔차 r8) A/B/C 3시드 판정 — 09-20 완주.
- DELIVER 확정 넷째 페어(4탭 읽기 시드4 · C3-only 기준선 시드4 · 4탭+센서별 prototype 시드4) 판정 — 09-20~21 완주.
