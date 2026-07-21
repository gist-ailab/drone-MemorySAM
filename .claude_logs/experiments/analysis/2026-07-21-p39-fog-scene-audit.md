# MUSES fog per-scene 감사 — 파국 장면 가설 기각 (2026-07-21)

**대상**: P39-DPC ep146 vs P38-m2f ep156, MUSES val, jarvis GPU6, `tools/p39_fog_scene_audit.py` (신규 도구, develop). CASE 조합명(fog_night 등)은 이 경로에서 미지원이라 fog(n=58)/night(n=100) 단일 조건으로 실행. **산출물**: NAS `analysis_logs/P39_fog_scene_audit_20260721/`.

## 결과

| 셀 | 모델 | n | mean | median | worst5 범위 | skew(med−mean) |
|---|---|---|---|---|---|---|
| fog | P39 | 58 | 79.27 | 79.96 | 53.0~62.5 | +0.69 (균일) |
| fog | P38 | 58 | 80.89 | 80.74 | 51.2~68.1 | −0.15 (균일) |
| night | P39 | 100 | 70.00 | 68.18 | 46.1~51.4 | −1.82 (균일) |
| night | P38 | 100 | 71.87 | 71.52 | 43.1~52.4 | −0.34 (균일) |

(수치 = per-image present-class mIoU의 평균 — D1의 조건 aggregate 클래스평균 mIoU와 정의가 다름, 직접 비교 금지)

## 판정

1. **파국 장면 가설 기각**: fog 58장 분포가 조밀(worst ~51+, skew≈0) — "소수 깨진 장면이 평균을 끌어내린다"는 시나리오는 없다. night도 동일.
2. **fog 약점의 실체는 클래스 축**: per-image 지표(fog ~79)와 D1 조건 mIoU(62.36)의 큰 괴리는 fog 약점의 상당 부분이 **희소 클래스의 조건부 전멸**(traffic light/rider/train 0@fog — 대부분 이미지에 부재한 클래스가 aggregate 클래스평균을 끌어내림)에 있음을 시사. 장면 품질 문제가 아니다.
3. **fog 헤드룸 하향 조정**: 리서치(07-21)의 "초과결손 8~9pt" 추정은 metric 정의 차이(문헌 PQ aggregate vs 우리 mIoU aggregate의 rare-class 민감도)를 감안해 보수적으로 볼 것. 현실 목표 = ①공식 test fog_night 셀의 P38 수준 복원(−12.05 회복, 이건 per-scene가 아니라 lidar 대체능력 문제로 실재) + ②fog 희소 클래스 회복(일부는 소표본 아티팩트 가능 — GT 존재 확인 필요).
4. **P39.1 투입 판단: GO** — 이 감사는 투입을 막지 않는다. rank 수리의 근거(공식 test fog_night −12.05, drop-lidar 6.33→1.24)는 per-scene 문제가 아니므로 유효.

## 재현

`CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34 python tools/p39_fog_scene_audit.py --cfg configs/jarvis-muses_rgbel_P39_dpc.yaml --ckpt P39=<ep146> --ckpt P38=<ep156> --conditions fog,night --split val --gpu 0 --out <out>` (도구의 이미지-ID dict 출력 버그는 후속 커밋에서 수정됨)
