---
legacy_id: 01
legacy_file: 01_project_status.md
split_from: 01_project_status.md
moved: 2026-07-08
owns: [status-snapshot]
updated: 2026-09-29
---

<!-- PI:BEGIN -->
paper_target: 단일 아키텍처(ReliaDINO)로 DELIVER·MUSES·MCubeS 전 모달 융합 계열 1위 — 논문 트랙 (출처: CLAUDE.md 프로젝트 개요)
headline: DELIVER test **57.74**(A2 3시드 {58.61, 57.09, 57.52}, best 58.61, legal v2, 2026-10-27 adopt) · 24클래스 58.56 · val 미확인(시드903 69.86) · 결측 val EMM15 57.85 · MUSES test 79.29±0.71(3모달 구기록, 논문 미사용; 4모달 P39.1 79.571) · MCubeS 58.07±0.49(P39.1) (출처: 아래 ① 표 = headline.yaml)
sota_gap: DELIVER test mean **+1.03** vs DGFusion 56.71(best +1.90) · 결측 val EMM15 +7.92 vs RMMSS 49.93 · MUSES test(4모달 P39.1) +0.07 vs DGFusion 79.5(논문 모델 A2 미측정) · MCubeS +2.17 vs StitchFusion 55.9(P39.1, A2 미측정) (출처: ① 표)
next_gate: ① 메인 3표 완성 — MCubeS A2 전이 DRN-261017-02 2시드(yeon 대기열) · MUSES 4모달 A2 DRN-261019-02 2시드(교사 E1M4 공식 val 후 TEACHER_CKPT) → ② ablation(L1 평가·L2·L4·L5·L6·단순 융합) + 비용표 DRN-261027-02 → ③ 논문 초고 동결 2026-11-10(DRN-261027-01, IEEE Sensors Journal) · 병행: PRB-DRN-261020-01 교차 확인 DRN-261022-01 통과 시 S2 1런(이번 투고엔 미포함). A2 3시드 판정 adopt 완료(10-27)
blocker: MUSES test 제출 보류(user 2026-09-29: 4모달 논문 모델로만 — MUSES 4모달 A2 완주·공식 val 뒤 user 가 직접 제출) · IEEE Sensors Journal 1회 투고 목표(user 2026-10-21): 분석형+성능 틀, ablation 7런·백본 사다리 3런 대기열
last_updated: 2026-10-27 (A2 3시드 adopt → 헤드라인 57.74 교체, 강건성 adopt, 논문 집필 착수)
<!-- PI:END -->

> **역할**: 현재 상태 스냅샷 — 네 블록(벤치 표·지금 도는 것·블로커·판정 대기)만 담는다(상한 60줄, 2026-09-18 재설계 — 감사 R2). 이력·경위는 [history-2026H2.md](history-2026H2.md), 날짜 엔트리 적층 금지.
> 헤드라인 수치 정본 = [../experiments/headline.yaml](../experiments/headline.yaml) · 판정·측정 규칙 = [../experiments/protocol.md](../experiments/protocol.md) · 판정 이력 = [../experiments/judgment-ledger.md](../experiments/judgment-ledger.md)

# 프로젝트 현황

## ① 벤치별 최선 — 표는 생성물(`tools/gen_headline_tables.py`, 멱등·`--check` 검사)

<!-- headline:begin -->
| 벤치·분할 | SOTA(1차 같은 모달 / 2차 적은 모달) | 우리 최고(값·근거·규칙·상태) | 격차 | 근거 |
|---|---|---|---|---|
| DELIVER · test | DGFusion 56.71 / 2차 MM SAM-adapter 57.35 | **57.74** · best 58.61(legal v2) — 3시드 평균(분석 문서 tasks/DRN-261018-01/analysis.md 표) · 시드별 {58.61, 57.09, 57.52} · best = 시드821 · 24클래스 평균 58.56(Q2 57.06 대비 +1.50) · 얇은 4클래스 Δ vs Q2 +3.04 · std 미기재(분석 문서에 없음) · 학습기 val-best(top1) · 유효(v2, 3시드, 40ep 스크린) | mean +1.03 · best +1.90(DGFusion 56.71) · mean +0.39 · best +1.26(MM SAM-adapter 57.35) | [DRN-261018-01 analysis.md(3시드 표, 시드903 legal)](../tasks/DRN-261018-01/analysis.md) · [DRN-261012-02 analysis.md(시드821·902)](../tasks/DRN-261012-02/analysis.md) · [lab-plan verdict DRN-261018-01 adopt 2026-10-27](../plan/plan.yaml) |
| 〃 · test | DGFusion 56.71 / 2차 MM SAM-adapter 57.35 | **56.24** ±0.42 · best 56.73(legal v2) — 3시드 평균 ± std(감시 세션 보고값) · best = 시드2(56.73) · 시드별 {55.97, 56.73, 56.03} · 24클래스 평균 56.85(std 0.36) · 얇은 4클래스 49.10(std 1.67) · 학습기 val-best(top1) · 유효(v2, 3시드) | mean −0.47 · best +0.02(DGFusion 56.71) · mean −1.11 · best −0.63(MM SAM-adapter 57.35) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) · [카드 W38 2026-09-18 legal v2 채택 항목](../decisions/cards/2026-W38-verdicts.md) |
| 〃 · val | CAFuser-CAA 68.79 / 2차 MM SAM-adapter 69.6 | **69.51** ±0.15 · best 69.68(legal v2) / 67.89(v1 최고 시드3) — 3시드 평균 ± std(감시 세션 보고값) · best = 시드3(69.68) · 시드별 {69.39, 69.45, 69.68} · 학습기 val-best(top1) · 유효(v2, 3시드) | mean +0.72 · best +0.89(CAFuser-CAA 68.79) — 같은 4모달 계열 val 1위 · mean −0.09 · best +0.08(MM SAM-adapter 69.60) | [judgment-ledger 2026-09-18 확정 6건 v2 행](../experiments/judgment-ledger.md) |
| MUSES · test | DGFusion 79.5 / 2차 GtA 82.39 | **79.29** ±0.71 · best 79.788(MUSES 공식) — 2시드 평균 ± 표준편차 · best 단일 런 병기 · 학습기 val-best(top1) · 유효 | mean −0.21 · best +0.29(DGFusion 79.5) · best −2.60(GtA 82.39) · best −1.28(MM SAM-adapter 81.07) | [judgment-ledger 2026-09-18 행](../experiments/judgment-ledger.md) · [analysis 시드20260825 판독](../experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md) |
| MCubeS · test | StitchFusion 55.9 | **58.07** ±0.49(커뮤니티 test 102장) — 3시드 {57.93, 57.67, 58.62} 평균 ± 표준편차 · 학습기 val-best(registry N4 행 표기) · 유효(1등 확정) | +2.17(StitchFusion 55.9) · +3.42(published 최고 Mul-VMamba 54.65) | [registry N4 행(yeon_mcubes_rgbadn_P39_1_rank)](../experiments/registry.md) |
| det(poongsan) | — | **0.9321**(val_det.py 재현) — ep6 · 재현 패키지 기록값 · 유효(종결 국면) | 목표 0.85 대비 +0.08 — 달성 완료 | [det 인증 문서 D1 스윕 표](../det/det-cert-D1-realtime.md) |
| MULTIAQUA | — | **82.10**(챌린지 서버) — P9 Val 93.29/Test 70.91(재제출 #16710) · P22 Val 93.42/Test 70.77(#16932) · 챌린지 제출 마스크 기준 · 유효(종료·고정) | —(챌린지 종료·고정) | [registry MULTIAQUA 표](../experiments/registry.md) |
<!-- headline:end -->

> 상태 표기 = 유효 / 보류 / 철회 / 불일치. 수치·근거·병기값은 headline.yaml 이 정본이다.

## ② 지금 도는 것

- 런·대기열·GPU 점유·ETA = [plan.md](../experiments/plan.md)(복제 금지).
- 링크: 구조·노벨티 https://claude.ai/artifact/Am5i5hF7CuPQCRuJWibNwc · 상황판 https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 · 노션 https://app.notion.com/p/gistailab/Drone-Object-Detection-for-RGB-IR-Fusion-33d05310a165408ab0b8ec4427d1fe2c

## ③ 열린 블로커 (2026-09-29)

1. hpca100 공유볼륨 89%(여유 244G, [artifact-locations](../infra/artifact-locations.md) 2b).
2. ISSUE-036 legal 재샘플 편차 — v2 채택 완료, 래퍼 가드 등재 잔여.
3. **기준선 강건 측정 완료**(DGFusion 공식·(b) 10-01, CAFuser (b) 10-04, 61케이스): Q2 3시드가 결측·depth 부분·노이즈 축 우위, RGB 부분 축은 CAFuser (b) 와 동률.
4. ISSUE-034·035 후속 정리 대기. RGB-only 융합 크래시 수정(10-22, 스모크 통과).
5. jarvis `/ailab_mat2` 금지(sshfs 정지). lecun·jarvis GPU0 금지 09-30 해제.

## ④ 판정 현황 (판정 = "MMSAM | 생각정리" 세션, 근거 = judgment-ledger 2026-09-23~29 행)

- **2모달 교사(10-02)**: RGB+Depth 40ep legal v2 test 56.56 = 4모달 스크린 55.94 동률(Δ24 +0.36) → clean 가산 이득 0, 4모달 가치는 강건 축.
- **종료(기각)**: R1 · Q1/Q3 · E17 · R2 · P55 · **P56 A·B·C(10-04~08)** → 논문 모델 = E1 구조 + Q2 레시피 고정.
- **채택(바탕 레시피)**: Q2(두 패스 clean CE + 열화 CE + 동결 E1 교사 증류). clean Δ24 +0.41(개선 주장 안 함) · 결측 15조합 48.06 vs E1 40.01 vs DGFusion (b) 45.52·공식 가중치 31.14 · depth 부분 저하 (b) 의 절반 · RGB 부분 결측 이득 없음.
- **보류**: E1-shared(전 센서 공유 LoRA r16) — clean·강건 게이트 둘 다 통과했으나 clean 분산이 커 단독 채택하지 않음, P56-C 재료.
- **실행 중**(상세=[history](history-2026H2.md), 설계서 decisions/2026-09-30-p56-bc): P56-A(모달충돌학습) hpca100 s821 GPU1(ep20)·s902 GPU2(ep17) — ISSUE-040(2장 DDP 정체)로 10-01 10:02 1장 원복, ETA 10-03 22시경. P56-B(거리조건화attn) s821 hpca100 GPU3·s902 yeon GPU1. P56-C(상태조건부LoRA) s821 yeon GPU0·s902 yeon GPU2.
- **N-RGBX-T**(2모달교사, DRN-261001-02) **3레그 legal v2 전부 완료 + NAS 보존 완료**: RGB+Depth(val 68.02/test 56.56) · RGB+LiDAR(val 61.68/test 55.31) · RGB+Event(val 62.27/test 55.68). 참조(4모달 E1 스크린821, NAS ckpt로 재현) test 55.94 — RGB+Depth만 +0.62 높고 나머지 둘은 낮아 clean 가산성 반증 조건 발동(생각정리 예비판정 1ce181b). ckpt+train.log+legal로그 전부 NAS `ckpts/bengio_deliver_rgb{d,l,e}_..._20261001/`에 md5와 보존. bengio GPU 전원 반납. oracle_cond_test+E1스크린821 val leave-one-out은 yeon 완료(headroom 1.87~2.48).
- **대기열**: CAFuser (b) 재개(DRN-260926-58, 스모크 선행) → MUSES 기준선 DGFusion 공식가중치 val 재현(DRN-261001-01, 데이터 스테이징 선행).
- MUSES: PhysAug-off 공정선 3시드 공식 val 82.30±0.14. test 제출 보류(user 09-29).
- **적체 판정 완료(09-30)**: 이관분 219건 전부 lab-plan verdict 기록. 미판정 없음.- **CAFuser (b) 완료(10-04)**: clean 55.79 · 결측 15조합 46.50 · depth 부분 저하 2.6/3.1/3.5 → Q2 우위 유지(EMM·depth·노이즈), RGB 부분 축은 동률.
