# P38-m2f 표준분석 (항목①~④) — test-best ep62, DELIVER 5-cond (2026-07-19)

**대상**: `test_epoch62_55.05_top1_checkpoint.pth` (hpca100 학습 중 스냅샷, ep78+ 진행 중; val-best 65.19@ep28)
**실행**: yeon GPU0, worktree dm_analysis @ develop cc68e14, `tools/seg_analysis_pipeline.py` 전 스테이지(D1,D2,D2N,D3,D3B,D4,D5) ok, 82분
**프로토콜**: P29~P36 분석과 동일 (5-cond, D1 120장/조건, D5 40장/조건, test-best ckpt 계보 비교)
**산출물**: yeon `/SSDb/jemo_maeng/analysis/P38_eval_20260719/` → NAS `analysis_logs/P38_eval_20260719/`
**⚠️ ckpt 주의**: test-best ckpt는 분석용(user 지시). 논문 수치는 val-best 규칙([[seg-report-sota-gap]]) 적용.

## 한줄 판정

**m2f(MaskQueryLite) 분지는 추론 시 사실상 no-op(off Δ+0.04~+0.12)이고, P36 fair 대비 동일 프로토콜 mean −1.63·thin-class(Wall/RailTrack) 퇴행 — "1-변수 비교" 게이트(val 67.74/test 55.62) 기준 현재까지 m2f 도입 이득 없음.** 단, router 의존도가 +38~42 → +1.6~2.4로 급감한 것은 m2f deep-supervision이 학습 동역학을 실질적으로 바꿨다는 유일한 구조적 신호.

## ④ 클래스×도메인 (D1, 동일 프로토콜 계보 비교)

| model (test-best ckpt) | cloud | fog | night | rain | sun | **mean** |
|---|---|---|---|---|---|---|
| P29 | 53.34 | 51.93 | 50.66 | 54.12 | 51.01 | 52.21 |
| P34 ep140 | 54.14 | 55.07 | 52.43 | 55.30 | 52.87 | **55.65** |
| P35 (fair) | — | — | — | — | — | 54.53 |
| P36 (fair+router) | — | — | 53.42 | 56.61 | — | 55.29 |
| **P38 ep62** | 54.05 | 54.63 | 52.05 | 54.47 | 53.09 | **53.66** |

- **P38 mean 53.66 = P36 −1.63 / P35 −0.87 / P34 −1.99** (P29·P32보단 위). 계보 게이트(P36 fair) 미달.
- per-domain spread 2.58 — 도메인시프트 아닌 per-class transfer 문제 (계보 공통 패턴 유지).
- **thin-class 게이트도 미달**: Wall 2.6~10.2(≈7, P36 13.3 대비 −6★), RailTrack 36~81(sun 35.9 붕괴, P36 62.5 대비 하락★), Water 4.8~21.3(≈11, P36 9.5와 유사). **mask-query 헤드가 thin-class를 구하리라는 가설 불발.**
- 도메인-불변 사망 동일: Other(≤5.2), Bridge(≤0.9). 도메인-민감 상위: RailTrack(spread 45), TwoWheeler(19), Water(17), Fence(15), Bus(15).

## ③ 모듈 전후 (D5, 신설 `p38_m2f_off` = m2f.beta→0)

| toggle | ΔmIoU(off) 5-cond | pred agree | 판정 |
|---|---|---|---|
| **p38_m2f_off** | **+0.04 ~ +0.12** | 0.998 | **추론 기여 no-op** — β=0.133로 열렸지만 sem_q 잔차가 로짓을 거의 못 바꿈 |
| p36_router_off | +1.55 ~ +2.43 | 0.963~0.973 | 최대 기여 모듈이나 **P36의 +38~42에서 1/20로 급감** — m2f deep-sup이 router 우회 경로를 학습시킴 |
| p34_gate_off | −0.03 ~ +0.53 | 0.992+ | ≈no-op (P34 계보 재현) |
| p34_calib_off | −0.14 ~ +0.32 | 0.992+ | ≈no-op |
| p34_veto_off | −0.03 ~ +0.00 | 1.000 | no-op |

- **P38 우위(vs P37b +2.7 test)의 원천은 추론 시 m2f 로짓이 아니라 학습 시 deep-supervision**(마스크 예측기가 supervised라 P37b의 random-mask 버그가 없음)일 수밖에 없음 — 그러나 그 효과조차 P36 fair를 못 넘으므로, 현 시점 m2f는 "P37b 버그를 고친 것"이지 "P36 대비 전진"이 아님.
- router 의존 급감은 양날: co-adaptation 해소(강건성↑ 가능) vs router가 담당하던 thin-class 회복(Wall/Water/RailTrack)이 함께 사라짐 — ④의 thin-class 퇴행과 정합.

## ① adapter 적응도 (D3/D3B)

- mm_lora 48 site 전부 활성(dead 0), dW mean: **lidar 18.90 > depth 17.14 > img 15.09 > event 13.99**.
- adapter on/off Δacc (D3B): **lidar +0.082~+0.143 (전 조건 최대·필수)**, depth +0.017~+0.038, img +0.012~+0.052(fog 최대), **event −0.003~+0.020 (cloud/fog에서 음수 — 사실상 dead adaptation)**.
- **event 모달 무기여가 P38에서도 재현** — P32(depth 잉여)·P34(event AUROC 최저 .70)·det ablation과 일관. 4-modal 구성의 event는 계보 전체에서 정당화 근거 없음.

## ② 피쳐 통계 (D2N, 120장/조건)

- eff.rank: img 19~22, depth 16~18, event 21~28, **lidar 6.9~7.5(최저)**, FUSED 7.1~7.7/256.
- dead ch 1/1024(무시 수준), 노름 균형(10~15) — DINOv3 계보의 건강한 피쳐 유지, SAM2류 rank-collapse 없음.
- cross-modal CKA 0.79~0.91 — 모달 간 중복 높음(P34 계보와 동일 범위). event가 rank는 높지만(27) Δacc는 0 — "표현은 다양하나 태스크 기여 없음"의 전형.

## 종합 판정

1. **P38은 현재 스냅샷 기준 게이트 미달**: 게이트(P36 fair val 67.74/test 55.62) 대비 val 65.19(−2.55)/test 55.05(−0.57), 동일 프로토콜 D1 mean −1.63. 학습이 ep78+ 진행 중이나 val은 ep28 조기포화·test는 plateau라 역전 가능성 낮음.
2. **m2f의 실체**: 추론 모듈로는 no-op(+0.07 평균), 학습 정규화로는 P37b 버그 수정분만큼만 유효. "Mask2Former-lite로 thin-class 회복" 가설은 반증(Wall −6, RailTrack sun 붕괴).
3. **구조적 관찰**: router 의존 +38~42→+1.6~2.4 급감 — deep-supervision이 router 우회 표현을 만들었다는 점은 P36 router 의존성(단일 실패점) 문제의 해법 후보로 기록 가치. 단 성능이 따라오지 않으면 무의미.
4. **다음 설계에 주는 수치 근거**: (a) event 모달 제거/교체 검토(전 세대 일관 무기여), (b) thin-class는 router 잔차가 유일하게 회복한 실적이 있으므로 m2f와 router의 결합 방식(현재는 m2f가 router를 대체·희석) 재설계, (c) physaug 복원이 P34 우위의 실체였음을 감안하면 P38+physaug 완주 비교 전까지 m2f 채택 보류.

## 재현

```bash
PY tools/seg_analysis_pipeline.py --cfg configs/hpca100-deliver_rgbdel_P38_m2f.yaml \
  --model_path <ckpt> --dataset-root <DELIVER> --out-dir <out> --gpu 0
# p38_m2f_off 토글: develop cc68e14+ (m2f.beta→0 잔차 차단)
```
