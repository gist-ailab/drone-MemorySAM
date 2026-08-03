# MUSES 공식 test 결과 — P46 C3-only λ0.2 (2026-08-03)

제출: `muses_P46_c3only_lam02_3modal_ep136_submission.zip` (P46 C3-only λ0.2, 3모달, ckpt ep136 val 81.65)

**Overall mIoU: 79.023** (750장)

## per-class (Full test)

| class | IoU |
|---|---|
| road | 97.03 |
| sidewalk | 86.74 |
| building | 93.14 |
| wall | 81.41 |
| fence | 65.37 |
| pole | 63.02 |
| traffic light | 69.58 |
| traffic sign | 72.69 |
| vegetation | 89.30 |
| terrain | 78.56 |
| sky | 96.65 |
| person | 70.12 |
| rider | 59.61 |
| car | 93.29 |
| truck | 75.71 |
| bus | 94.25 |
| train | 92.24 |
| motorcycle | 54.49 |
| bicycle | 68.24 |

## 조건별

| condition | mIoU |
|---|---|
| clear | 77.982 |
| fog | 78.637 |
| rain | 78.604 |
| snow | 77.947 |
| day | 79.096 |
| night | 75.951 |

## 조합 셀

| cell | mIoU |
|---|---|
| clear_day | 78.268 |
| clear_night | 73.667 |
| fog_day | 76.785 |
| fog_night | 69.334 |
| rain_day | 78.866 |
| rain_night | 73.619 |
| snow_day | 69.306 |
| snow_night | 73.711 |

## 주요 per-class 조건별 (발췌)

| class / condition | day | night |
|---|---|---|
| motorcycle | 37.26 | 60.58 |
| truck | 77.85 | 51.15 |
| bus | 96.29 | 79.75 |

fog train: 100.00 | fog rider: 40.66

## 대조표 (내부 val vs 공식 test)

| 제출 | 내부 val | 공식 test |
|---|---|---|
| P39.1-seed2 3모달 ep208 | 82.62 | **79.788**(우리 최고) |
| P43-pdual ep156 | 82.51 | 79.351 |
| **P46-C3only λ0.2 ep136** | **81.65** | **79.023** |
| P38-m2f ep156 | 82.22 | 79.025 |
| P34 3모달 ep276 | 81.02 | 78.979 |

## 판정

C3는 seed2 대비 test **−0.765**로 2~4위권이며 P38(79.025)과 사실상 동률. val −0.97 → test −0.765로 **낙차가 val보다 작다**. 🔴 **조건별 손실이 day(−1.15 vs seed2 80.25)에 집중되고 fog는 −0.07로 동급** — val 분해(Δclear −1.72/Δday −1.29/Δfog +0.16)와 **동일 패턴이 공식 test에서 재현**됨. 이는 P47-MUB의 진단(병목=clear/day RGB under-optimization, modality laziness)을 **공식 test 수치로 확증**한다.

이상: snow_day 69.306 < snow_night 73.711 **역전 재현**(P34·P38에 이어 3회째, 원인 미규명). day motorcycle 37.26 < night 60.58도 역전.
