# 학습0 검증 2건 판정 — router per-class/coverage (P43~P45 제안 §7, 2026-07-25)

**도구**: `tools/analyze_router_coverage.py` (develop 7b053e0) · **대상**: P38-m2f ep156 / P39-DPC ep146, MUSES val (clear n=60 / fog n=58 / night n=60, drop n=24) · **원시**: NAS `analysis_logs/router_coverage_20260725/`

## 판정 요약

| 검증 | 질문 | 판정 |
|---|---|---|
| **V1 (§7-a)** | 클래스마다 실제 다른 모달을 고르나? | **강한 해석("클래스별 모달 특화") 기각 · 약한 해석("전역 오배분 방지 + 조건 적응의 클래스 해상도") 실증** |
| **V2 (§7-b)** | 커버리지 밖에서 lidar 가중이 떨어지나? | **🔴 실패 확정 — 오히려 밖에서 높음(uniform 퇴화). V-1 presence 재정규화 필수 근거** |

## V1 증거

- **drop-modality ΔmIoU (P38)**: img +25.2/+16.9/+14.4 (clear/fog/night) vs lidar +0.21/+0.30/**+1.72**, event ≈0~0.7. P39도 동형(lidar ≤1.3). → 비RGB 인과 기여는 전 조건 미미 = 기존 진단 재확인.
- **단 per-class로 보면 클래스 구조 실재(소폭)**: clear rider +8.0(lidar), night traffic sign +7.3·vegetation +4.0·bicycle +3.9(lidar), night traffic light +10.2(event). "비RGB가 이기는 클래스"는 없지만 **한계 기여가 클래스-구조적**임은 확인.
- **router argmax가 비RGB인 클래스 수(P38)**: clear **4** → night **11** → fog **13** — router는 조건×클래스로 실제 분화하며 방향도 물리적으로 정합(stuff/기하 클래스 위주: road, sidewalk, terrain, vegetation).
- **🔴 핵심 발견 — 가중↔인과 괴리**: fog에서 13클래스가 비RGB argmax인데 drop-lidar는 +0.30뿐. **router 가중은 존재하나 인과 사용으로 전환되지 않는다** = "가중이 아니라 사용이 문제" → P44(학습시 강제: MMPareto/국소마스킹)의 전제를 독립 재확인. 추론시 게이트류가 무력했던 계보 이력과도 정합.

## V2 증거

| (P38) | 안(inside) | 밖(outside) | cover_frac |
|---|---|---|---|
| clear | 0.097 | **0.352** | 0.181 |
| fog | 0.123 | **0.367** | 0.114 |
| night | 0.184 | **0.374** | 0.151 |

(P39 동형: 안 0.10~0.15 vs 밖 0.25~0.26.)

- 밖 값이 ≈1/3에 밀집 = **zero-fill 입력의 무정보 feature에서 softmax가 uniform으로 퇴화** — §7-b가 예측한 "router가 무효 데이터를 모른다"의 정확한 실현이며, 예상(안 떨어짐)보다 나쁨(역전).
- 안쪽 lidar 가중은 조건 악화에 따라 상승(P38 clear 0.097→night 0.184) — 커버리지 안에서는 적응 방향 정상.
- **처방 확정**: P44 V-1(presence 재정규화, 결정론·무학습) — 부재 픽셀 가중 0 + 잔여 재정규화. **P44 본학습 config에서 V-1 기본 on으로 전환**(이 판정 근거, eval 토글 `p44_validity_off`로 ablation 분리 가능). MULTIAQUA 확장 시 선행 조건 재확인.

## 논문 서사 확정 (§7-a 반영)

- 주장: "per-class router는 **전역 조건 게이트의 클래스별 오배분을 방지**하고 조건×클래스 해상도로 분화한다(4→13 클래스)" + "가중↔인과 괴리 측정이 학습시 균형화(P44)의 동기" — 두 측정 모두 경쟁 논문이 보고하지 않는 진단.
- 금지: "클래스마다 우세 모달이 다르다"(clear 인과 기여로 반증), "router 가중 = 사용"(괴리 실증).
