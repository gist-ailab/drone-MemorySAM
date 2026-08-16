---
legacy_id: 01
legacy_file: 01_project_status.md
split_from: 01_project_status.md
moved: 2026-07-08
---

> **역할**: 프로젝트 **현재 상태 스냅샷의 단일 출처(single source of truth)** — 매 갱신 시 아래 스냅샷 블록만 덮어쓴다.
> 날짜 붙은 진행 엔트리(📝/🏆/⚠️/🛠)는 **이 파일에 쌓지 말고** [history-2026H2.md](history-2026H2.md) 최상단에 append한다 (2026-08-08 재확립 — 이전에 22개 엔트리가 여기 적층돼 스냅샷 기능을 잃었던 사고의 재발 방지).
> 과거 이력: [history-2026H2.md](history-2026H2.md)(2026-07-01~) · [history-2026H1.md](history-2026H1.md)(~2026-06-30)

# 프로젝트 현황 (Project Status)

> 최종 업데이트: **2026-08-08** (스냅샷 전면 재작성 — 문서 정리 경위는 history 2026-08-08 엔트리 참조)
> 📊 **사용자용 상황판(artifact)**: https://claude.ai/code/artifact/11924e8a-12fc-4dbc-a174-ead7259b0228 — 갱신 규약 [meta/conventions.md](../meta/conventions.md) §4 (판정 변화 시 `meta/status-report.html` 갱신 + 동일 URL 재배포)

---

## 📌 현재 상태 스냅샷 (CURRENT — 여기만 읽으면 됨)

**연구 정체성 (2026-08-08 개정)**: 계보 12세대의 공통 가설은 "모달 신뢰도/유용도에 따른 적응적 가중". 검증 결과 — **추론 경로 안의 가중(학습 게이트·SoftMoE·RBMA attn-bias·추론 재가중)은 전부 반증**됐고, 성능을 실제로 움직인 축은 ① frozen 백본 + per-modal LoRA(SAM2→DINOv3, 계보 최대 단일 변수 +11.6) ② 트렁크 rank 복원(P39.1 gated-MLP+VICReg) ③ **학습 전용 손실**(P46-C3 prototype, deep supervision)이다. 구 정체성 문구(RBMA attn-bias)는 폐기 — 근거는 [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md) §1과 SOTA 진단 artifact(2026-08-08).

**🎯 공식 목표 (user 2026-07-03 설정, 기준선 2026-08-08 갱신)**: ① **Seg = 논문 publish** — DELIVER 현행 SOTA = **MM SAM-adapter val 69.60 / test 57.35** (구 기준 DGFusion 66.51/56.71은 이미 상회); MUSES test SOTA = **GtA 82.39(camera-only)**, 융합계보 기준 DGFusion 79.5. ② **Det = 국책과제 mAP50 0.85 — 달성 완료**(0.9321). ③ MULTIAQUA 확장 예정.

### 벤치별 현재 최선 (legal 프로토콜: val-best 또는 final-iter만, test-best 금지)

| 벤치 | 우리 최선 | vs SOTA | 판정 |
|---|---|---|---|
| DELIVER | **P46 C3-only 본run** val-best@ep70, **@1024 평가**: val **69.44** / test **56.99** (RailTrack 67.69, base 4.02) | MM SAM-adapter 대비 **−0.16 / −0.36** | 사정권 — 격차 < 단일런 편차(0.59). 구 SOTA(DGFusion)는 no-tradeoff 상회(+2.93/+0.28) |
| MUSES | **P39.1-rank seed2 3모달** Codabench test **79.788** (val 82.13; day 80.246/night 76.818, fog_night 69.610 최악) | GtA(camera-only) −2.60 / **융합(4모달)계보 1위**(79.571 > DGFusion 79.5) | 정면 돌파 비현실 → 포지셔닝 전환(융합계보 1위 + adverse robustness 인과 실증) |
| MUSES PQ | things 22.87 / All 35.55 (P47-D1 ep172) | SOTA(CAFuser) 59.26 −23점대 | PQ 축 비교 불가 — limitation 절 소재 |
| Det | D1-recovered(ViT-L) AP50 **0.9321**@ep6 | 목표 0.85 **+0.08** | 종결 국면 |
| MULTIAQUA | P9 ep131 / P22 ep120 M-score **82.10** | (챌린지 종료, 고정) | 고정 |

### 통일 아키텍처 확정 (2026-08-16, §6 규칙 집행)

**P49-AIR 계열 종결** (DELIVER P46 미달 + MUSES 공식 val 81.16 < 82.13 + G-4M 실패). **통일 = ReliaDINO 계보(P39.1-rank 추론 그래프) + 진단-짝 학습손실** — 양 벤치 현 최고(P46-CTR 56.99 / P39.1-rank 79.788)가 이미 동일 추론 그래프(C1/C2/C3 전부 학습 전용). 판정 상세 [experiments/analysis/2026-08-16-p49-1-muses-official-verdict.md](../experiments/analysis/2026-08-16-p49-1-muses-official-verdict.md). 남은 결정전 = **C2**(DELIVER SOTA 산술 경로 + 레시피 통일 여부).

### 활성 런 / 대기 (2026-08-08, 커밋 기준 — 실시간은 [experiments/plan.md](../experiments/plan.md)·registry)

- 🔴 **P46 C3-only @1024² 학습 — 게이트 미달 확정 (2026-08-09 02:35, 잠정→확정 격상)**. elice-b200 val-best@ep70: val **69.79**(+0.35 vs 768² 본run) 인데 legal test **56.50**(−0.49 vs 내부최고 56.99, −0.85 vs SOTA 57.35) — **val↑/test↓ 역발산**.
  yeon λ0.05 병행런도 동일 패턴(ep60→62 사이 val 68.72→69.03 소폭↑, test 56.58→**54.29** 급락 −2.29).
  **확정 근거**: ep70 이후 24 epoch(ep90~94) 동안 val 이 69.79→69.73→69.51→69.23 로 **단조 하락** — 정체가
  아니라 명백한 하락 국면. 동 구간 raw test 는 58.41@ep92 로 SOTA 를 크게 웃도는 값이 찍혔으나 **비legal**
  (val 하락 중 발생) — val-test 상관이 낮다는 추가 증거이지 돌파 근거가 아니다.
  **해석**: 1024² 학습이 val 은 밀어올리지만 test 는 별개로(과적합 방향으로) 움직인다 — 08-08 22시 리포트의
  "1024² 가 정점을 밀어올린다" 잠정 해석은 **기각**. elice 잔여 epoch 는 참고용 관찰로 격하.
  → **RA-L 포지셔닝("no-tradeoff 우위 + MUSES 융합계보 1위 + adverse robustness 인과") 이 현재 더 현실적 경로.**
  ⚠️ 게이트 확정 미달로 3-seed 재현 확인의 실익이 낮아짐(이미 2개 독립런이 val↑/test↓ 재현) — jarvis 5장·
  yeon 3장(6주기·12시간 미배치) 용도를 **C2(MCC) 순기여 측정(hpca100 A100×2 유휴) 또는 RA-L 소재 확보**로
  전환 검토. (jarvis GPU6,7 분신은 08-08 새벽 ep18에서 사망 → 그 슬롯은 seedB로 전환, 아래.)
- ✅ **P39.1-rank @1024² 대조 — 목적 달성, ep126/200 에서 조기 종료(2026-08-08 14:50, user 지시, GPU 4장 회수)**.
  **해상도 순효과 확정** (양쪽 val.py@1024 직접 실측, 환산 없음):

  | 학습 해상도 | ckpt (val-best) | val | test |
  |---|---|---|---|
  | 768² | P39.1-rank ep106 | 66.72 | 53.68 |
  | 1024² | P39.1-rank ep54 | **67.87** | **55.69** |
  | **순효과** | | **+1.15** | **+2.01** |

  → 전 세대(P39.1~P47)가 768² 로 학습된 동안 **test 약 2점을 해상도만으로 손해**보고 있었다.
  조기 종료 근거: val-best 가 ep54 이후 **70 epoch** 갱신 없음 + legal 값이 위 실측으로 확정 → 잔여 epoch 정보가치 없음.
  부수 검증: 실측(67.87/55.69)이 학습로그 환산 추정과 소수점까지 일치 → 오프셋(val −2.58/test −1.79)이 **이 런에 대해** 정확. 런별 값이므로 외삽 금지.
- 🆕 **P46 C3-only @1024² seedB** — jarvis GPU6,7 (2f3ff6e). **첫 진짜-시드 런**(`TRAIN.SEED`=20260808).
  ⚠️ 그 전까지 'seed2/seed3' 런은 시드가 실제로 달라진 적이 없다 — `fix_seeds(3407)` 하드코딩이고 `MODEL.C3.SEED` 는 C1 off 시 inert.
  따라서 기존 편차 0.59 는 GPU 비결정성만 반영한 값 = 참 시드 분산의 **하한**이고, SOTA 격차 −0.36 은 그 하한보다도 작다.
- ⏸ **P47-2 UniBal** — 구현·스모크 완료, A100급 4장 대기.
- 🔴 **CEA 프로브(조건-전문가 상한) — 폐기 확정 (2026-08-08 16:00, 제안 세션)**: 7런 완주, oracle Δ(fog_night) **+0.21 < 게이트 +1.0**(5배 미달), night가 주야갭 4.33 중 **+0.02만 회수** → "평균 최적성 함정" 가설까지 반증. **적응 가설 계열(융합 가중→추론 재가중→추출 전문화) 3단계 완결 폐쇄** — 남은 격차의 원인은 배분이 아니라 **정보**. 재제안 금지. canonical = **[research/hypothesis-ledger.md](../research/hypothesis-ledger.md)(가설 원장, 신설)** + [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md) §6·§7. 음성 결과는 MUSES 포지셔닝의 oracle 상계로 논문 회수.
- ⚠️ **jarvis GPU2-5 완전 유휴 (08-08 15:13 실측)** — GPU-never-idle 규칙 위반 상태. 투입 후보(학습 0 우선): ① RGB-D 2모달 @1024 fair-eval(config e055aab 준비됨) ② elice ep28 val-best ckpt 회수 후 **조기 fair-eval**(게이트 조기 판정 가능 — val-best가 ep28에서 18ep째 정체 중이라 이미 확정됐을 수 있음).

### 논문 트랙 (CVPR 2027 마감 ~2026-11 중순 / RA-L rolling)

- **분기 게이트 = P46 @1024² 판정(08-09)**: 돌파 + 3-seed 재현 → CVPR 도전 / 미달 → RA-L 확정.
- 스토리(2026-08-08 논의): "test 전이 실패는 단일 병리가 아니다 — 클래스축(DELIVER)·조건축(MUSES)은 다른 처방을 요구한다" — 기둥 = per-modal LoRA 트렁크(P39.1) + 학습 전용 prototype 손실(P46-C3) + drop-modal 인과 분석. C3의 MUSES 이식 실패(−0.765)는 대조 실험으로 재활용.
- 🔴 **RA-L 초안(ReliaDINO v1, 볼트 `_paper_submission/`) 재중심화 필요** — 현재 RBMA 중심 서사는 반증된 상태. [research/ral-paper-plan.md](../research/ral-paper-plan.md)의 슬롯 3(MUSES 제출)·5(multi-seed)는 이미 충족됨(문서에 미반영).

### 열린 블로커 / 미결

0. ✅ **DELIVER 채점 프로토콜 확정 완료(2026-08-14)** — MM-SA(현 SOTA)=native GT(**우리와 동일, SOTA 비교 유효**) / CMNeXt·CAFuser·DGFusion 계열=1024-리사이즈 GT(낙관 지표 — 이들 대비 우리 수치는 과소). 잔여 작업 = P46 ep70의 1024-GT 재채점 1건(학습 0, DGFusion 비교 각주용). [analysis/2026-08-14-p49-1-fair-eval-metric-protocol.md](../experiments/analysis/2026-08-14-p49-1-fair-eval-metric-protocol.md)

1. **P48 폐기 판정 재확정 필요** — 08-06 게이트 적용 시점 오류 지적([experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md](../experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md)) 후 상위 재판정 기록 없음. 논문 스코프 밖으로 두되 기록은 닫을 것.
2. **C2(MCC) 순기여 미측정** — 유일하게 결과를 모르는 조합 (40GB급 필요).
3. **RGB-D 2모달 fair-eval(학습 0)** — SOTA 최고 구성 대비 직접 비교, 기존 ckpt @1024 재평가만 남음.
4. MUSES RGB-L 2모달 런(~1일) — 상위권 실구성과 직접 비교.
5. 반증 확정(재제안 금지) 목록 = artifact D절: attn-bias 계열·추론 재가중·CEFR·zero-init 잔차·rank/η² 개입·모달 드롭·gradient 균형화·radar(MUSES)·NORM_ALL.

### 재현성 규약 (전 세션 공통, 논문 표 작성 시 재검증)

- **test-best ckpt 인용 금지** (철회 사고 2회: P34 57.60, P46 57.05). val-best 선택 민감도 큼(ep20→26에서 test −2.76) → 3-seed mean±std 필수.
- 학습 @768 / 평가 @1024 mismatch는 논문에 명시 (P46 @1024² 학습 완주 시 해소).
- 진행보고 포맷 = user auto-memory `progress-report-format` (2블록 + 벤치 baseline 표).
