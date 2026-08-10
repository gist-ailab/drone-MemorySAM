---
created: 2026-08-10
type: MUSES 공식 test 판독 (제출 #9, RGB-L 2모달)
---

# MUSES 공식 test — RGB-L 2모달 79.571 (2026-08-10, 제출 #9)

> 제출물 = `muses_P39_1_rank_rgbl_2modal_ep136_submission.zip` (val-best ep136, val 82.00). Codabench 수치 = user 수신 원문. 판정 = fable(discussion 세션).

## 1. 헤드라인 + 모달 스펙트럼 완성

| 구성 | 공식 test | Δ |
|---|---|---|
| 3모달 (img/lidar/event, seed2) | **79.788** | 계보 최고 |
| **2모달 RGB-L (본 제출)** | **79.571** | 3모달 −0.217 |
| 4모달 (+radar, seed2) | 79.571 | 3모달 −0.217 |

- **2모달과 4모달이 소수점 셋째 자리까지 동률(79.571).** 모달 한계기여: event +0.217, radar −0.217 — **MUSES에서 RGB+LiDAR 이후의 모달은 사실상 평탄**. "모달을 늘리면 좋다" 가정의 최종 해부(논문 modality-efficiency 표 완성).
- val에서는 동급(82.00 vs 5-seed 평균 82.03)이었는데 test에서 −0.217 — val→test 전이 낙차의 일관 패턴 재확인.
- ⚠️ **동일 구성 직접 비교에서는 열세**: MM SAM-adapter RGB-L 81.07(내부 기록 기준) 대비 **−1.50** — "우리 스택의 병목은 모달 수가 아니라 RGB-L 기본기"라는 H12′(대형 백본 전제·스택 열세 정직 공개)와 정합.

## 2. 조건별 (2모달, 750장)

| 축 | 수치 |
|---|---|
| 날씨 | clear 80.004 / fog 78.545 / rain 78.562 / snow 77.708 (spread 2.30) |
| 주야 | day 80.497 / night 76.786 — **갭 3.71** (3모달 3.43과 유사) |
| 최악 조합 | **fog_night 69.819** (3모달 69.610보다 +0.21 — event 없이도 fog_night 동급) |
| 역전 재현 | **snow_day 70.127 < snow_night 75.160** — 계보 3번째 재현 (기존 2회 + 본 건) |
| 특이 | fog에서 train IoU 100.00 (표본 소수·단일 인스턴스 추정 — 통계 취약, 인용 주의) |

per-class 전문·조건 조합 전체 = user 수신 원문(Codabench detailed_results), NAS 제출 인덱스에 보존.

## 3. 판정

1. **"2모달 = 상위 구성과 동급" 확정** (내부 계보 기준): RGB+LiDAR만으로 계보 최고 −0.217. event는 fog_night 방어에도 불필요함이 확인됨(69.819 ≥ 69.610).
2. **DELIVER와의 대비 완성**: DELIVER는 2모달이 −2.85 붕괴(같은 날 fair-eval) ↔ MUSES는 2모달 동급 — **벤치별 모달 요구 상이**가 양방향 실측으로 닫힘. 논문 modality 분석 절의 골격.
3. 단 **랭킹 관점 이득은 없음**: 79.571 < 79.788이므로 리더보드 대표는 여전히 3모달 seed2. 이 제출의 가치는 순위가 아니라 ablation 행.

관련: [2026-08-10-rgbd-2modal-fair-eval.md](2026-08-10-rgbd-2modal-fair-eval.md)(DELIVER 대비쌍) · [MUSES_TEST_RESULTS_INDEX.md](MUSES_TEST_RESULTS_INDEX.md) · registry `jarvis_muses_rgbl_P39_1_rank_2modal`
