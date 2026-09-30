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
headline: DELIVER test 56.24±0.42(best 56.73, legal v2 3시드) · val 69.51±0.15 · MUSES test 79.29±0.71(best 79.788, 2시드) · MCubeS 58.07±0.49 (출처: 아래 ① 표 = headline.yaml)
sota_gap: DELIVER test mean −0.47 vs DGFusion 56.71 · val mean +0.72 vs CAFuser-CAA 68.79 · MUSES test mean −0.21 vs DGFusion 79.5 · MCubeS +2.17 vs StitchFusion 55.9 (출처: ① 표)
next_gate: P56-A(모달 충돌 학습: Q2 레시피의 열화 패스에 "한 센서 영역이 자신 있게 틀린" 표본 추가) 40ep 스크린 2시드 — 사전 기준 = clean 24클래스 Δ vs Q2 ≥ −0.3 그리고 (실제 과노출·저노출·모션블러 케이스 mIoU 평균 Δ ≥ +1.0 또는 RGB 부분 결측 저하 20% 이상 감소), lab-plan DRN-260929-01
blocker: MUSES test 제출 보류(user 2026-09-29: 새 4모달 모델로 다시 만들 것)
last_updated: 2026-10-01 (헤드라인 불변. 감시/판정 구역 분담, 8KB 재압축)
<!-- PI:END -->

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

- 런·대기열·GPU 점유·ETA = [plan.md](../experiments/plan.md)(복제 금지).
- 링크: 구조·노벨티 https://claude.ai/artifact/Am5i5hF7CuPQCRuJWibNwc · 상황판 https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 · 노션 https://app.notion.com/p/gistailab/Drone-Object-Detection-for-RGB-IR-Fusion-33d05310a165408ab0b8ec4427d1fe2c

## ③ 열린 블로커 (2026-09-29)

1. **hpca100 공유 볼륨 89%**(여유 244G, [artifact-locations](../infra/artifact-locations.md) 2b). 계속 감시.
2. **ISSUE-036** legal 재샘플 편차 — v2(nearest-exact) 채택 완료, 래퍼 가드 등재 잔여.
3. **기준선 강건 측정 완료**(10-01, DGFusion (a)·(b) 61케이스 같은 프로토콜): Q2 3시드가 (b) 대비 clean +1.8·결측 15조합 +2.5·depth 부분 저하 절반. CAFuser·MM SAM-adapter 는 미측정.
4. **ISSUE-034** eval 예측 덤프 파일명 평탄화 · **ISSUE-035** 헤드라인 ckpt 경로 기록(이슈 표 갱신 대기).
5. jarvis `/ailab_mat2` 접근 금지(sshfs 정지). lecun·jarvis GPU0 금지 09-30 해제.

## ④ 판정 현황 (판정 = "MMSAM | 생각정리" 세션, 근거 = judgment-ledger 2026-09-23~29 행)

- **종료(기각)**: R1(depth 경계 prior, Δ24 −0.09) · Q1/Q3(합성 열화 품질 헤드, clean −1.31) · E17(세부 가지) · R2(연결 성분 손실, Δ24 ≈ 0).
- **채택(바탕 레시피)**: Q2(두 패스 clean CE + 열화 CE + 동결 E1 교사 증류). clean 3시드 Δ24 +0.41(개선 주장 안 함) · 결측 15조합 평균 48.06 vs E1 스크린 40.01 vs DGFusion 재학습판 45.52·발표판 31.53(같은 프로토콜 10-01) · depth 부분 저하 0.98/2.04/2.64 vs DGFusion (b) 2.51/2.88/3.47 · RGB 부분 결측 이득 없음.
- **보류**: E1-shared(전 센서 공유 LoRA r16) — clean·강건 게이트 둘 다 통과했으나 clean 분산이 커 단독 채택하지 않음, P56-C 재료.
- **P55(자기감독 게이트) 계열 종료**: 오라클 순 여유 조건별 1.1~1.8 · 분할 손실만으로 학습한 팔 test 57.59, 자기 leave-one-out 손실을 목표로 준 팔 test 57.67(게이트가 사실상 상수) — 둘 다 Q2 57.73 과 같은 수준. 곱셈 게이트는 주 후보에서 제외.
- **실행 중**(상세=[history](history-2026H2.md), 설계서 decisions/2026-09-30-p56-bc): P56-A(모달충돌학습) hpca100 s821 GPU1·s902 GPU2(완주 10-03). P56-B(거리조건화attn) s821 hpca100 GPU3·s902 yeon GPU1. P56-C(상태조건부LoRA) s821 yeon GPU0(교사버그 수정 재기동)·s902 yeon GPU2. B/C s902 04:11 착수(게이트 앞당김), 검증 PASS.
- **신규 착수(10-01)**: N-RGBX-T(2모달교사, DRN-261001-02) bengio GPU1,2,3,5 DDP 연쇄(`n_rgbx_t_chain`). oracle_cond_test+E1스크린821 val leave-one-out → yeon GPU3,4,5(ETA ~5h).
- **재배치(user 10-01 승인)**: P56 6런을 2장 DDP 로 재개(A s821 hpca100 1+3 · B s821 jarvis 0+1 · yeon 3런 2장씩) → 첫 판정 10-02 오후 예상. bengio 1·2·3·5 N-RGBX-T 연쇄, 그 외 잔여 GPU 는 rNs.
- **대기열(bengio, 10-01 등록)**: N-RGBX-T(E1 2모달 교사 3런, DRN-261001-02, P30, 위에서 기동 완료) → CAFuser (b) 재개(DRN-260926-58, P40, 스모크 선행) → MUSES 기준선 DGFusion 공식 가중치 val 재현(DRN-261001-01, P35, 데이터 스테이징 선행).
- **MUSES**: PhysAug-off 공정선 3시드 공식 val 82.30±0.14(824·825·826). test 제출은 보류(user 09-29).
- **적체 판정 완료(09-30)**: 이관분 219건 전부 lab-plan verdict 기록. 미판정 없음.
