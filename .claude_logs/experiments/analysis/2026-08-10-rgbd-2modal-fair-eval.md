---
created: 2026-08-10
type: fair-eval 판정 (RGB-D 2모달 @1024)
---

# RGB-D 2모달 fair-eval — 열세 확정, "2모달 충분" 가설 기각 (2026-08-10)

> 판정 = fable(discussion 세션). 실행 = sonnet(yeon GPU2/6, val 2005장 + test 1897장, 각 ~1h). legal 프로토콜(val-best ckpt).

## 수치

| 구성 | val | test | 비고 |
|---|---|---|---|
| **RGB-D 2모달** (P46 C3 λ0.05, ep66 val-best, **@1024 평가**) | **65.79** | **54.14** | 본 측정 |
| MM SAM-adapter RGB-D (SOTA 최고 구성) | 69.60 | 57.35 | −3.81 / −3.21 |
| 우리 4모달 최고 (P46 C3 본run @1024 평가) | 69.44 | 56.99 | −3.65 / −2.85 |

로그: yeon `drone-MemorySAM-p38/logs/rgbd_eval1024_{val_20260810_042129,test_20260810_043002}.log` · 산출 `outputs/ReliaDINO/yeon_deliver_rgbd_P46_c3only_lam005_2modal/`

## 판정

1. **"DELIVER는 RGB-D 2모달로 충분하다" 가설 기각.** 우리 스택의 RGB-D는 SOTA의 RGB-D 구성(57.35)에 −3.21, 우리 4모달에도 −2.85로 크게 열세. MM SAM-adapter가 2모달로 도달하는 지점을 우리는 4모달 없이 못 간다.
2. **모달 기여의 위치가 정밀화됨**: drop-modal ablation(추론 시 제거)에서 event/lidar Δ≈0였는데, **학습부터 빼면 −2.85**다. 즉 event/lidar의 기여는 추론 시 폴백이 아니라 **학습 시 표현 형성**에 있다 — "무기여" 서사를 "추론 무기여·학습 유기여"로 교정해야 한다(논문 modality 분석 절의 핵심 뉘앙스; MUSES RGB-L=3모달 동급과 대비되는 벤치 차이이기도 함).
3. **@1024 평가 이득이 2모달에선 부재/역전** — 4모달은 @1024 평가로 +1.64를 벌었는데 2모달은 학습로그 66.46 대비 65.79로 오히려 낮다. 해상도 이득이 모달 수와 상호작용할 가능성(관찰 1건, 단정 금지).

## 교란 요인 (명시)

- λ0.05 런(우리 4모달 최고는 λ0.2 본run) — 단 λ 0.05~0.2는 평탄 확인돼 있어 −2.85를 설명 못 함.
- ep66 vs ep70, 단일런(시드 분산 미확보).

## 논문 사용

ablation 표의 "modality 축소" 행으로 사용: DELIVER는 4모달 필요(−2.85), MUSES는 2모달 동급(RGB-L 82.00) — **벤치마다 모달 요구가 다르다**는 축-특이성 주장의 모달 버전.

관련: [2026-08-06-p46-c3only-fair-eval-final.md](2026-08-06-p46-c3only-fair-eval-final.md) · registry `yeon_deliver_rgbd_P46_c3only_lam005_2modal_eval1024` 행
