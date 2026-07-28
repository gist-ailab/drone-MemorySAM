# `certification/` — 인증 평가 스크립트 세부 참조

> **먼저 읽을 것: 리포 루트의 [`../README.md`](../README.md)** — 환경 구성, 실행법,
> 체크포인트, 산출물 경로, 데이터셋, 트러블슈팅이 모두 거기에 모여 있다.
> 이 문서는 인증 **수치의 출처**와 이 디렉터리 파일들의 세부만 다룬다.

## 인증 수치 (원 기록)

인증 대상 모델 **D1 ViT-S+** = ReliaDINO(frozen DINOv3 **ViT-S+/16** + per-modal LoRA
+ ReliabilityGatedFusion + SimpleFPN) **+ RF-DETR NMS-free head**. 3-modal
(RGB + LiDAR depth + Thermal), 10 classes, 768². 웨이트 = `det_D1_vitsp_20260723`
(epoch 11, md5 `b78f614cba1375bb54dfeacd5e58cef3`).

poongsan_v2 final test, 3,239장, **predicted-scope**:

| 지표 | 값 |
|---|---|
| mAP50 (전체) | **0.9166** |
| mAP50 (night) | 0.8765 |
| mAP50 (normal) | 0.9418 |
| FPS | 7.38 (RTX 3090) / 약 10–12 (RTX 5080) |
| VRAM | 0.76 GB |

재현/배치 수치는 `tools/det_eval_breakdown.py`, FPS는 `tools/det_fps_bench.py`로도 낼 수 있다
— `cert_eval.py`와 동일한 `tools/_det_common.py` 기반이라 값이 일치한다.

## split 사실관계

poongsan final split = **clean capture-holdout**:
train 5클립(`112051` / `113007` / `113534` / `115206` / `120059`, 12,681장) vs
eval 3클립(`114021` / `114808` / `115624`, 3,239장), **frame·clip 모두 겹침 0**
→ 인접프레임 누수조차 없음. 이 사실은 `check_split.py`로 재확인할 수 있다.

night 판정 기본 클립 = `capture_20260618_114021` + `capture_20260618_115624` (1,768장),
`capture_20260618_114808`(1,471장)은 normal.

## 파일

| 파일 | 역할 |
|---|---|
| `run_cert.sh` | 엔터 한 번 러너. timm shadow(`pylibs_p34`) prepend + protobuf 가드 + `runs/cert_D1/console_<ts>.log` tee 후 `cert_eval.py` 호출. 인증 config(`configs/det/det_D1_vitsp_jarvis.yaml`) 고정 |
| `cert_eval.py` | 시작 배너(클래스/모달리티/ckpt md5/장치) → ENTER → 이미지별 추론·GT 비교 스트리밍 + 오버레이 저장 → mAP/FPS/VRAM 리포트 + `cert_report.json` |
| `check_split.py` | 모델·GPU 없이 annotation만 비교하는 학습/평가 분리 검증. frame-level + clip-level, exit 0(PASS)/1(FAIL), `split_check.json` 기록 |

실행 명령과 옵션은 루트 README 3절을, 산출물 파일 목록은 5절을 볼 것.

> ℹ️ `cert_eval.py` docstring의 예시에 나오는 `certification/configs/det_D1_vitsp_dronedemo.yaml`은
> 이 패키지에 포함돼 있지 않다. 인증 config는 `configs/det/det_D1_vitsp_jarvis.yaml`이며
> `run_cert.sh`가 이것을 사용한다. 다른 머신에서는 `--data-root`로 경로만 갈아끼운다.

## YOLO 대조군

인증 모델과 독립된 baseline이다. 경로와 실행법은 루트 README 3-(d)절 참조
— `objdet/yolov5m-lowlight/`, `objdet/yolo11m-rgb/`, `objdet/gistolo/`.
