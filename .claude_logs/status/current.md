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
last_updated: 2026-10-01 (헤드라인 수치 변동 없음. P56-B/C 둘째 시드 착수 반영)
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

- 실행 중 런·대기열·GPU 점유·ETA = [../experiments/plan.md](../experiments/plan.md) "실행 중"·"GPU 예약·점유 현황" 표(여기 복제 금지).
- 구조·노벨티 정리(09-29) https://claude.ai/artifact/Am5i5hF7CuPQCRuJWibNwc
- 링크: 상황판 artifact https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 · 노션 논문 페이지 https://app.notion.com/p/gistailab/Drone-Object-Detection-for-RGB-IR-Fusion-33d05310a165408ab0b8ec4427d1fe2c

## ③ 열린 블로커 (2026-09-29)

1. **hpca100 공유 볼륨 89%**(여유 244G: 09-29 체크포인트 32개 이관 + 09-30 덤프·캐시 150.9G 삭제. [../infra/artifact-locations.md](../infra/artifact-locations.md) 2b 절). 공유 볼륨이라 계속 감시.
2. **ISSUE-036** legal 하네스 재샘플 편차 — legal v2(nearest-exact) 채택 완료, v2 래퍼 가드 등재 잔여.
3. **기준선 강건 측정 완료**(10-01, DGFusion (a)·(b) 61케이스 같은 프로토콜): Q2 3시드가 (b) 대비 clean +1.8·결측 15조합 +2.5·depth 부분 저하 절반. CAFuser·MM SAM-adapter 는 미측정.
4. **ISSUE-034** eval 예측 덤프 파일명 평탄화 · **ISSUE-035** 헤드라인 ckpt 경로 기록(이슈 표 갱신 대기).
5. jarvis 에서 `/ailab_mat2` 접근 금지(sshfs 정지). lecun·jarvis GPU0 금지는 09-30 해제(빈 GPU 규칙만 적용).
6. ~~ISSUE-038 hpca100 미커밋 코드~~ → 09-23 회수·develop 커밋으로 종결.

## ④ 판정 현황 (판정 = "MMSAM | 생각정리" 세션, 근거 = judgment-ledger 2026-09-23~29 행)

- **종료(기각)**: R1(depth 경계 prior, 3시드 Δ24 −0.09) · Q1/Q3(합성 열화 감독 품질 헤드: clean −1.31, 실제 전이 실패) · E17(세부 가지) · R2(연결 성분 손실, Δ24 ≈ 0).
- **채택(바탕 레시피)**: Q2(두 패스 clean CE + 열화 CE + 동결 E1 교사 증류). clean 3시드 Δ24 +0.41(개선 주장 안 함) · 결측 15조합 평균 48.06 vs E1 스크린 40.01 vs DGFusion 재학습판 45.52·발표판 31.53(같은 프로토콜 10-01) · depth 부분 저하 0.98/2.04/2.64 vs DGFusion (b) 2.51/2.88/3.47 · RGB 부분 결측 이득 없음.
- **보류**: E1-shared(전 센서 공유 LoRA r16) — clean·강건 게이트 둘 다 통과했으나 clean 분산이 커 단독 채택하지 않음, P56-C 재료.
- **P55(자기감독 게이트) 계열 종료**: 오라클 순 여유 조건별 1.1~1.8 · 분할 손실만으로 학습한 팔 test 57.59, 자기 leave-one-out 손실을 목표로 준 팔 test 57.67(게이트가 사실상 상수) — 둘 다 Q2 57.73 과 같은 수준. 곱셈 게이트는 주 후보에서 제외.
- **실행 중**: P56-A(모달 충돌 학습) hpca100 시드821 GPU1(09-29 07:08 UTC 기동)·시드902 GPU2(12:24 UTC 기동), 에폭당 약 2.1시간, 완주 예상 10-03. **P56-B(거리 조건화 attention) 시드821 hpca100 GPU3 기동(09-30 08:20 UTC, 감시 세션 검증 7항목 통과, P56-A 무영향 확인)** — 설계·구현 검수는 생각정리 세션이 마쳤고(develop e0ca300·4b5e1b4) user 승인 09-30("응 기동해"). **P56-C(센서 상태 조건부 LoRA 전문가 혼합) 시드821 yeon GPU0 기동(09-30 21:4x KST, 검증 6항목 통과)** — hpca100·jarvis 빈 GPU 부재 + lecun/yeon 공유 `MMSS_SAM` env의 timm 0.4.12(DINOv3 미지원, ISSUE-039)로 4차 시도 끝에 성공. timm 1.0.24로 상향 완료(lecun도 동일 조치, SAM2 계보는 timm 미사용이라 무관 확인). 🔴 **09-30 22:31 KST 재기동**: 최초 기동분은 교사가 학생과 같은 state_routed 아키텍처로 빌드돼 블록7~24 LoRA 없이 로드된 결함 발견(missing=151/unexpected=72, KD 목표 오염 → 무효). `train_reliadino.py` 수정(교사는 항상 E1 아키텍처로 빌드, 키 불일치 시 즉시 RuntimeError, `tools/tests/test_teacher_arch.py` 검사 추가, develop 1f9b83a) 후 같은 GPU0에서 kill+재기동, `missing=0 unexpected=0` 확인. 무효 런 산출물은 `..._P56C_invalid_teacher_20260930`로 보존(삭제 안 함). P56-B는 교사 missing=2(RangeBias만 빠짐, λ≈0)라 유효 — 영향 없음. 🔴 **10-01 04:11 KST 둘째 시드(902) 착수**(생각정리 세션 지시 2026-10-01 — "둘째 시드는 A 판정 뒤" 게이트는 GPU 부족 전제였고, yeon에 7장 빈 GPU가 생겨 사전등록·레시피 그대로 2시드 평균 판정으로 진행, 설계서 decisions/2026-09-30-p56-bc-modality-aware-design.md): **P56-B s902 yeon GPU1**·**P56-C s902 yeon GPU2**, config는 hpca100 seed902 원본에서 경로만 교체(파생 검증 완료, develop eb28f76), 교사 로드 `missing=0 unexpected=0` 양쪽 확인, 검증 4항목 전부 PASS(파라미터 수·에러 없음·iter 전진·GPU 활성화 메모리). GPU0(s821)은 영향 없음 확인. yeon 체크아웃에 encoder/fusion/model/train_reliadino unstaged 변경 + p56c_router.py 등 untracked 모듈이 남아 있어 develop 병합 잔여(추후 처리 필요, 이번 두 런은 그 로컬 소스로 기동).
- **신규 착수(10-01)**: **N-RGBX-T**(E1 레시피를 2모달 교사 RGB+X로 좁혀 모달별 순기여 분리, lab-plan DRN-261001-02, 생각정리 지시·user 직접 승인) bengio GPU1,2,3,5 4GPU DDP 순차 연쇄(tmux `n_rgbx_t_chain`, 04:25:48 KST 기동) — RGB+Depth→RGB+LiDAR→RGB+Event. 1번째 런 iteration 전진·GPU 활성화 확인, 첫 eval 미도달(추후 재확인). **oracle_cond_test**(Q2 s821·E1확정s1 test leave-one-out) + **E1스크린821 val leave-one-out**을 hpca100→yeon GPU3,4,5로 이관 기동(04:22 KST), ETA 약 5시간(배치당 75~78초×238/251배치).
- **MUSES**: PhysAug-off 공정선 3시드 공식 val 82.30±0.14(824·825·826). test 제출은 보류(user 09-29).
- **적체 판정 완료(09-30)**: 이관분 219건 전부 lab-plan verdict 기록. 미판정 없음.
