# MUSES P39.1-rank 4-modal seed2 — drop-radar ablation (radar 무익 확정)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 결과

| | val mIoU |
|---|---|
| (a) 4모달 그대로(rgb+lidar+event+radar) | **82.35** (학습 로그 재현 정확 일치, ep260 best) |
| (b) radar zero-fill 3모달(rgb+lidar+event) | **82.22** |
| **dMIoU = (a) - (b)** | **+0.13** |

- ckpt: `outputs/ReliaDINO/hpca100_muses_rgbelr_P39_1_rank_4modal_seed2/MUSES_ReliaDINO-ViTL16_iler/epoch260_82.35_top1_checkpoint.pth` (완주 300/300, Total Training Time 12:12:38).
- 도구: `tools/eval_reliadino_ckpt.py --drop-modality radar`(신규 플래그, 커밋 66b67b8) - `tools/feature_stats.py --drop-modality`와 동일 시맨틱으로 radar 텐서를 zero-fill(모델 구조는 그대로, 입력만 0). 체크포인트 로드 양쪽 다 missing=0/unexpected=0(아키텍처 완전 일치).
- 실행: hpca100 GPU2,3(물리, forward-only), env `RELIADINO_LOCAL_BACKBONE=<local dinov3 safetensors>`(hpca100 HF 우회) + `LD_LIBRARY_PATH`(venv 번들 cuDNN) + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. 타테넌트(GPU0,1)는 실행 전후 메모리/util 변화 없음(무간섭 확인).

## Per-class Δ (4모달 - 3모달, + = radar 도움)

| Class | 4모달 | 3모달 | Δ |
|---|---|---|---|
| rider | 59.52 | 58.18 | **+1.34** |
| wall | 75.17 | 74.56 | +0.61 |
| traffic sign | 74.17 | 73.76 | +0.41 |
| terrain | 79.49 | 79.18 | +0.31 |
| truck | 90.65 | 90.43 | +0.22 |
| building | 93.06 | 92.90 | +0.16 |
| bus | 93.75 | 93.60 | +0.15 |
| fence | 71.43 | 71.55 | -0.12 |
| car | 93.11 | 93.18 | -0.07 |
| traffic light | 77.87 | 78.13 | -0.26 |
| motorcycle | 80.51 | 80.86 | -0.35 |
| bicycle | 71.86 | 72.28 | **-0.42** |
| road / sidewalk / pole / vegetation / sky / person / train | ~동일 | ~동일 | ±0.1 이내 |

## 판정 (코디네이터)

- **radar는 MUSES에서 무익하다고 확정.** dMIoU +0.13은 노이즈 수준(≈0) - rider 한 클래스(+1.34)만 실질적 신호이고, bicycle/motorcycle/TL은 오히려 radar 없이 더 낫다(방향 비일관).
- **3-seed plateau(<82.62)와 정합**: 4모달(82.35) < 3모달 seed2 기록(82.62) - radar를 추가한 4-modal 경로가 오히려 3-modal 표현을 희석시켜 원래 3-modal 기록에도 못 미친다. **"4-modal이 SOTA로 가는 길"이라는 가설이 이 실험으로 반증됨.**
- DELIVER의 depth/lidar 잉여 현상(별도 분석에서 확인된, 추가 모달이 정보이득 없이 표현만 희석)과 동형 - 멀티모달 융합에서 "모달을 더 넣는다고 항상 좋아지지 않는다"는 패턴이 두 데이터셋에서 재현됨.

## 후속

- hpca100 GPU2,3는 이 결과로 확보된 A100×2 - C2+C3 본학습 착수 대기(코디네이터 지시 예정, @1024 C3-only test 결과가 마지막 설계 입력).
