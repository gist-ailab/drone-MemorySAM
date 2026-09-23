---
legacy_id: 01
legacy_file: 01_project_status.md
split_from: 01_project_status.md
moved: 2026-07-08
owns: [status-snapshot]
updated: 2026-09-23
---

> **역할**: 현재 상태 스냅샷 — 네 블록(벤치 표·지금 도는 것·블로커·판정 대기)만 담는다(상한 60줄, 2026-09-18 재설계 — 감사 R2). 이력·경위는 [history-2026H2.md](history-2026H2.md), 날짜 엔트리 적층 금지.
> 헤드라인 수치 정본 = [../experiments/headline.yaml](../experiments/headline.yaml) · 판정·측정 규칙 = [../experiments/protocol.md](../experiments/protocol.md) · 판정 이력 = [../experiments/judgment-ledger.md](../experiments/judgment-ledger.md)

# 프로젝트 현황

## ① 벤치별 최선 — 표는 생성물(`tools/gen_headline_tables.py`, 멱등·`--check` 검사)

<!-- headline:begin -->
| 벤치·분할 | SOTA(1차 같은 모달 / 2차 적은 모달) | 우리 최고(값·근거·규칙·상태) | 격차 | 근거 |
|---|---|---|---|---|
| DELIVER · test | DGFusion 56.71 / 2차 MM SAM-adapter 57.35 | **56.24** ±0.42 · best 56.73(legal v2) — 3시드 평균 ± std(감시 세션 보고값) · best = 시드2(56.73) · 시드별 {55.97, 56.73, 56.03} · 24클래스 평균 56.85(std 0.36) · 얇은 4클래스 49.10(std 1.67) · 학습기 val-best(top1) · 유효(v2, 3시드) | mean −0.47 · best +0.02(DGFusion 56.71) · mean −1.11 · best −0.63(MM SAM-adapter 57.35) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) · [카드 W38 2026-09-18 legal v2 채택 항목](../decisions/cards/2026-W38-verdicts.md) |
| 〃 · val | CAFuser-CAA 68.79 / 2차 MM SAM-adapter 69.6 | **69.51** ±0.15 · best 69.68(legal v2) / 67.89(v1 최고 시드3) — 3시드 평균 ± std(감시 세션 보고값) · best = 시드3(69.68) · 시드별 {69.39, 69.45, 69.68} · 학습기 val-best(top1) · 유효(v2, 3시드) | mean +0.72 · best +0.89(CAFuser-CAA 68.79) — 같은 4모달 계열 val 1위 · mean −0.09 · best +0.08(MM SAM-adapter 69.60) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) |
| MUSES · test | DGFusion 79.5 / 2차 GtA 82.39 | **79.29** ±0.71 · best 79.788(MUSES 공식) — 2시드 평균 ± 표준편차 · best 단일 런 병기 · 학습기 val-best(top1) · 유효 | mean −0.21 · best +0.29(DGFusion 79.5) · best −2.60(GtA 82.39) · best −1.28(MM SAM-adapter 81.07) | [judgment-ledger 2026-09-18 행](../experiments/judgment-ledger.md) · [analysis 시드20260825 판독](../experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md) |
| MCubeS · test | StitchFusion 55.9 | **58.07** ±0.49(커뮤니티 test 102장) — 3시드 {57.93, 57.67, 58.62} 평균 ± 표준편차 · 학습기 val-best(registry N4 행 표기) · 유효(1등 확정) | +2.17(StitchFusion 55.9) · +3.42(published 최고 Mul-VMamba 54.65) | [registry N4 행(yeon_mcubes_rgbadn_P39_1_rank)](../experiments/registry.md) |
| det(poongsan) | — | **0.9321**(val_det.py 재현) — ep6 · 재현 패키지 기록값 · 유효(종결 국면) | 목표 0.85 대비 +0.08 — 달성 완료 | [det 인증 문서 D1 스윕 표](../det/det-cert-D1-realtime.md) |
| MULTIAQUA | — | **82.10**(챌린지 서버) — P9 Val 93.29/Test 70.91(재제출 #16710) · P22 Val 93.42/Test 70.77(#16932) · 챌린지 제출 마스크 기준 · 유효(종료·고정) | —(챌린지 종료·고정) | [registry MULTIAQUA 표](../experiments/registry.md) |
<!-- headline:end -->

> 상태 표기 = 유효 / 보류 / 철회 / 불일치. 수치·근거·병기값은 headline.yaml 이 정본이다.

## ② 지금 도는 것

- 실행 중 런·대기열·GPU 점유·ETA = [../experiments/plan.md](../experiments/plan.md) "실행 중"·"GPU 예약·점유 현황" 표(여기 복제 금지).
- 링크: 상황판 artifact https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 · 노션 논문 페이지 https://app.notion.com/p/gistailab/Drone-Object-Detection-for-RGB-IR-Fusion-33d05310a165408ab0b8ec4427d1fe2c

## ③ 열린 블로커 (2026-09-22)

1. 🔴 **ISSUE-038** hpca100 repo(cddc319)의 R1/R2·QAF 구현이 **미커밋 로컬 수정(+566줄)** 로만 존재 — pull 금지(날아감), develop 커밋 시급. [../issues/issues-and-fixes.md](../issues/issues-and-fixes.md)
2. 🔴 **ISSUE-036** legal 하네스 재샘플 floor 정렬 편차 — legal v2(nearest-exact) 채택 완료, v2 래퍼 가드 등재 잔여.
3. **yeon GPU0-3 상실**(09-22 리부트 직후 타인 선점, [../experiments/plan.md](../experiments/plan.md) GPU 표) — 4장 운용, e1scr_s903 재개 대기. ~~bengio GPU 전면 고장~~ → 09-23 감시 세션 실측으로 **정정**: bengio 에서 40ep 스크린 6런이 완주했고 legal v2 재채점 6건이 GPU 0·1·2·3·4·7 에서 돌고 있다(인수인계 문서의 "고장"은 낡은 정보). e1scr_s903 은 인수인계 §1 기준 jarvis 에서 완주(65.80@40, 재채점 대기).
4. **ISSUE-034** eval 예측 덤프 파일명 평탄화 — test 1270/1897장만 남음. 과거 이미지별 분석 점검 필요.
5. **ISSUE-035** 헤드라인 ckpt 경로 기록 — NAS 정본 이관 완료, 이슈 표 갱신 대기.
6. **lecun 배치 금지**(user 2026-09-17) · **hpca100 공유 볼륨 감시**(09-15 Errno 28 전례).

## ④ 다음 판정 대기 (판정 = "MMSAM | 생각정리" 세션)

> 🔴 **09-23 15:30 클로드 코드 재개용 인수인계 정본 = [handoff-2026-09-23-zcode.md](handoff-2026-09-23-zcode.md)** — R1/R2 legal v2 확정 수치·사고·남은 일 전부 거기에.

- **R1(경계 prior)·R2(연결 성분 손실) legal v2 1차 수치(시드821, 잠정)** — R1 test 56.58(jarvis 사본)·56.80(hpca100 사본) / val 68.65·69.09, R2 test 56.56·val 68.88. 짝(E1 스크린) 55.94/68.56. ⚠️ 생각정리 판정(09-23): "DGFusion 초과" 문구 철회 — 같은 시드의 R1 이 두 사본(jarvis test 56.58 / hpca100 56.80)으로 존재하고 유리한 쪽만 고를 수 없음, 얇은4 56.39·50.81 은 TrafficSign 을 넣은 비정본 정의(정본 = Pole·Pedestrian·Static·TrafficLight). 판정은 3시드 후([judgment-ledger 09-23 행](../experiments/judgment-ledger.md)). 판정 잔여 = 정본 얇은4 + "찾고도 못 그린 비율" 러너(21dfb04) + RMM depth Δ + 시드 902·903.

- **P54 Q2(증류 대조군)·Q3(QAF 본 카드)** 40ep — jarvis 09-21 저녁 기동. Q1b-2 probe(RGB 연산자별 분해) 결과 회수 후 RGB 품질 헤드 범위 확정.
- **DGFusion 기준선 (b) 열화 재학습 완주(09-22 02:47)** — val-best 사후 스윕(ckpt 20개) + Q0 잔여(EMM/NM 기준값) → G-robust-vs-DGFusion 판정 재료.
- **muphys_824·825**(MUSES PhysAug-off 공정선 3페어) ~09-25 오전 완주 · muphys_826 미기동(autoplace queue) · **e1scr_s903·E1-shared 3시드** 스크린 미기동(배치 승인 대기).
- **DELIVER 넷째 페어**(E1·E13·C3-only 시드4) — 완주·재채점 완료, 판정 대기.
- MUSES 공정선 제출본(E13M 81.95, 09-20 생성 `submission/muses/20260920_…zip`) 제출 여부 — user 직접.
