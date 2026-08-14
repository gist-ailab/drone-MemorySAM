---
created: 2026-08-14
type: fair-eval 판정 + 측정 프로토콜 규명
---

# P49.1 fair-eval 최종 + 채점 해상도 프로토콜 발견 (2026-08-14)

> 판정 = discussion 세션(fable). 실측 = sonnet(yeon, val.py 4종 + 트레이너 evaluate 단독 재현 하니스).

## 1. 수치 (ep126 val-best ckpt, 전부 동일 ckpt·동일 추론 해상도 768)

| 측정기 | test | 채점 방식 |
|---|---|---|
| 트레이너 내부 eval (하니스로 재현, 57.6800 정확 일치) | **57.68** | 예측·GT 모두 **768²로 리사이즈 후** IoU |
| val.py (오프라인) | **55.66** | 예측을 **native 1042²로 복원**, 원본 GT에서 IoU |
| val.py test@1024 | 56.19 | native GT |
| val.py val@1024 | 62.29 | native GT (@1024 평가가 P49.1엔 유해 — P46과 반대) |
| val.py test@768 **INJECT-off** | 42.96 | A/B — **주입 제거 시 −12.7 붕괴** |

## 2. 판정

1. **측정기 불일치 해소** — 모델·ckpt·추론은 동일하고, **채점 해상도 프로토콜 차이**가 전부다(트레이너는 testset을 `return_meta` 없이 만들어 768-리사이즈 GT로 채점; val.py는 orig_label로 native 채점). per-class 이동이 thin 클래스(Pole +8.2, Fence +2.7, TrafficLight +3.5)에 집중 — 다운샘플-GT 지표의 전형적 낙관 편향(MUSES 81.02→80.86 사고와 동일 계열).
2. **"57.68 SOTA 돌파" 영구 철회** — 내부 정본(val.py native-GT, P46 legal 56.99와 동일 자)으로 P49.1 test = **55.66@768 / 56.19@1024 → P46 미달(−1.33/−0.80), SOTA 미달.** P49.1의 DELIVER 성능 도전은 실패로 판정.
3. **기제는 생존** — INJECT-off 붕괴(−12.7)로 비대칭 주입이 추론 필수 부품임은 확정(키1 흡수 없음, γ 0.019의 절대 기여 실재). 성능 미달과 기제 생존은 분리 기록.
4. 🔴 **공개표 비교 가능성 이슈 격상 (논문 블로커)** — CAFuser 코드의 `CMNEXT_EQUIVALENT_EVAL` 주석("GT를 1024 NEAREST 리사이즈 — 원조 DELIVER/CMNeXt 코드베이스와 동일")에 따르면 **커뮤니티 관례가 resized-GT 채점일 가능성**. 그렇다면 MM-SA 57.35·DGFusion 56.71 등 공개 수치는 낙관 지표이고, 우리 native-GT 수치(P46 56.99)는 **그들보다 보수적인 자로 잰 것 = 비교 자체가 불공정하게 불리**했을 수 있다. 07-15 경고의 재부상 — **DELIVER 공식 채점 프로토콜 확정 + 우리 대표 ckpt들의 protocol-matched 재채점(학습 0)이 논문 전 필수.** 결과에 따라 공개표 대비 서열이 (우리에게 유리하게) 바뀔 수 있으나, **내부 서열(P49.1 < P46)은 동일 자 비교라 불변.**

## 3. 재발 방지

- 트레이너 [Val]/[Test] 로그 수치는 **어떤 대외 비교에도 사용 금지** (768-GT 낙관 지표) — 판정은 항상 val.py native(내부) 또는 protocol-matched(대외).
- registry·보고에 "학습로그 값" 표기 의무 유지.

원시: yeon `drone-MemorySAM-p49/logs/p491_*_20260814_011725.log` + `tools/diag_trainer_eval.py`(yeon 전용 하니스) + `/tmp/diag_trainer_eval_run.log`

관련: [2026-08-06-p46-c3only-fair-eval-final.md](2026-08-06-p46-c3only-fair-eval-final.md) · monitor-log L720(CMNEXT_EQUIVALENT_EVAL 경고 원문) · registry `yeon_deliver_rgbdel_P49_1_air_768_g01`

## 4. 공개표 프로토콜 확정 (2026-08-14, 3개 repo 1차 소스 — 블로커 #0 해소)

| Repo | IoU 채점 | 함의 |
|---|---|---|
| CMNeXt(원조)·CAFuser(명시적 CMNeXt-equivalent, 기본 ON)·(DGFusion 등 그 계열) | **1024², GT nearest-다운샘플** | 구 리더보드 수치(DGFusion 56.71 등)는 리사이즈-GT 지표 |
| **MM SAM-adapter(현 SOTA)** | **native 1042², GT 원본**(mmseg 관례, 예측 bilinear 업샘플) | **우리 val.py와 동일 프로토콜** |

**판정**:
1. **vs MM-SA(57.35) 비교는 처음부터 유효했다** — 우리 native-GT 정본과 동일 프로토콜. P46 −0.36, P49.1 미달 판정 전부 유지.
2. **vs DGFusion 계열 비교는 우리가 불리한 자로 재 왔다** — 그들의 1024-리사이즈-GT는 낙관 지표. protocol-matched 재채점(P46 ep70 @1024-GT, 학습 0) 시 우리 수치는 상승만 가능 → "구 SOTA no-tradeoff 상회(+0.28)" 마진이 커진다. 재채점 1건만 하면 논문 표 각주 완결.
3. **공개표 자체가 두 프로토콜 혼재**(MM-SA native vs CMNeXt 계열 1024-GT) — 이 발견 자체가 논문 프로토콜 절/각주 소재(우리가 양 프로토콜 수치를 모두 공개하는 것이 최선의 방어).
