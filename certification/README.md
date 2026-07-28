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
| `cert_eval.py` | 시작 배너(클래스/모달리티/ckpt md5/장치) → ENTER → 이미지별 추론·GT 비교 스트리밍 + 2단 패널 오버레이 저장 → mAP/FPS/VRAM 리포트 + `cert_report.json` |
| `check_split.py` | 모델·GPU 없이 annotation만 비교하는 학습/평가 분리 검증. frame-level + clip-level, exit 0(PASS)/1(FAIL), `split_check.json` 기록 |

실행 명령과 옵션은 루트 README 3절을, 산출물 파일 목록은 5절을 볼 것.

## 오버레이 레이아웃 (`viz/<NNNN>_<파일stem>.png`)

인증 모델은 3모달(RGB + LiDAR depth + Thermal) 입력인데 예전 오버레이는 RGB 한 장뿐이라
야간 프레임에서는 거의 검은 화면에 박스만 떠 있어 **왜 검출됐는지, 멀티모달을 쓰긴 하는지**가
보이지 않았다. 그래서 기본 산출물을 2단 패널(1536×1280) 한 장으로 바꿨다.

```
┌───────────────────────┬───────────────────────┐
│   GT  (RGB + GT 박스) │  Pred (RGB + 예측)    │  768² × 2
├───────────┬───────────┼───────────┬───────────┤
│   RGB     │  LiDAR    │  Thermal  │           │  512² × 3
└───────────┴───────────┴───────────┴───────────┘
```

- 상단 왼쪽 `GT` = GT 박스만(녹색 + 클래스명), 오른쪽 `Pred` = 예측만(클래스색 + score,
  `val_det.draw_detections` 그대로). 좌우를 눈으로 바로 대조하라고 나눠 놓은 것이다.
- 하단은 **모델에 실제로 들어간 텐서**를 그대로 그린다(파일 재로딩 없음). 순서는 config의
  `DATASET.MODALS`.
- LiDAR/Thermal은 값 범위가 서로 다르고(depth m, thermal raw) 그대로 찍으면 새까맣게 나오므로
  **타일별 min-max 정규화** 후 컬러맵(LiDAR=`INFERNO`, Thermal=`JET`)을 입힌다.
  채널이 3개면 평균내 1채널로 만든다. **표시용 변환일 뿐 모델 입력에는 영향이 없다.**
  `cv2`가 없으면 grayscale로 폴백한다(cv2는 optional).
- `--viz-mode rgb`를 주면 예전 단일 768² 오버레이(예측 클래스색 + GT 얇은 녹색)로 돌아간다.
  파일명·저장 경로 규칙은 두 모드가 동일하다.
- 모달 키가 없으면 있는 것만 그리고, viz가 어떤 이유로 실패해도 평가는 계속된다
  (`inference_log.txt`에 `[viz warn]` 한 줄만 남는다). 렌더링은 latency 측정 구간 **밖**이라
  FPS에 영향을 주지 않는다.

> ℹ️ `cert_eval.py` docstring의 예시에 나오는 `certification/configs/det_D1_vitsp_dronedemo.yaml`은
> 이 패키지에 포함돼 있지 않다. 인증 config는 `configs/det/det_D1_vitsp_jarvis.yaml`이며
> `run_cert.sh`가 이것을 사용한다. 다른 머신에서는 `--data-root`로 경로만 갈아끼운다.

## YOLO 대조군

인증 모델과 독립된 baseline이다. 경로와 실행법은 루트 README 3-(d)절 참조
— `objdet/yolov5m-lowlight/`, `objdet/yolo11m-rgb/`, `objdet/gistolo/`.
