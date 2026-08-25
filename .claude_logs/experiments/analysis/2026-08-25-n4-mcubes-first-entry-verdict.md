---
created: 2026-08-25
type: N4 MCubeS 첫 진입 판정 + N4b 예측 정정
---

# N4 MCubeS 판정 — published 최고 +3.28, 단 plaster는 벤치 공통 난제 (2026-08-25)

> 실측 = 모니터링 세션(yeon 6,7, 통일 레시피 C3-off, val-best 57.93@ep140). 문헌 대조 = 딥리서치 에이전트(코드 수준 split 검증 포함). 판정 = discussion 세션(fable).

## 1. 비교 유효성 — split 검증 완료

- MCubeS 커뮤니티 관행: CMNeXt·MMSFormer 공식 로더 모두 `split='val'` → **`list_folder/test.txt`(102장)** 로드. **우리 `semseg/datasets/mcubes.py` L140도 동일** — 같은 test split, 비교 유효.

## 2. 수치 — published 최고 대비 +3.28

| 방법 | 백본 | 4모달 mIoU |
|---|---|---|
| CMNeXt (CVPR'23) | MiT-B2 | 51.54 |
| MLE-SAM (2412.04220) | **SAM2 Hiera-B+** | 51.02 |
| MemorySAM (2503.06700) | **SAM2** | 52.88 |
| MMSFormer | MiT-B4 | 53.11 |
| StitchFusion+FFMs | MiT-B2 | 53.92 |
| Mul-VMamba (KBS'25) | VMamba 55M | **54.65 (published 최고)** |
| **우리 N4 (통일 레시피, C3-off)** | frozen DINOv3-L | **57.93 (+3.28)** |

- 🔑 **백본 반론에 대한 역설적 방어 자료**: SAM2급 대형 백본 방법들(MLE-SAM 51.02·MemorySAM 52.88)이 **소형 MiT-B4(53.11)·VMamba(54.65)보다 낮다** — MCubeS에서 "큰 백본 = 승리"가 성립한 적이 없다. DINOv3-L급 published 수치는 부재 → 우리가 대형 백본으로 이 벤치를 실제로 넘은 첫 사례. "백본이 커서 이겼다"는 반론에 "대형 백본 선행 2편은 못 넘었다"로 응수 가능(물론 백본 각주는 유지).
- 주장 지위: 수정 없는 통일 레시피(주행 센서→편광/NIR)로 첫 시도 +3.28 = **modality-agnostic 일반성의 강한 실증**. 단 단일런 — DELIVER 교훈(H18)대로 시드 확인 전 "SOTA" 단정 금지, 재현 런 필요.

## 3. 🔴 plaster 판정 정정 — 벤치 공통 난제, class-transfer 붕괴 아님

| 방법 | Plaster IoU |
|---|---|
| MCubeSNet 3.0 · CMNeXt 0.8 · MMSFormer 0.5 · U3M 1.2 | 전부 0~3 |
| **우리** | **0.40 (동일 대역)** |

- **이전 프레임(내 것) 정정**: "plaster 0.40 = class-transfer 붕괴 → C3가 회복 예측"은 **틀렸다.** 전 방법이 붕괴하는 클래스 = DELIVER의 Wall/Water/Bridge 유형(복구 불가 공통 난제)이지, RailTrack 유형(남은 되는데 우리만 죽는 회복 가능 격차)이 아니다.
- **N3 검출기의 필수 정교화**: 붕괴를 두 유형으로 분리해야 한다 — ① **회복 가능**(published 대비 우리만 크게 낮음, RailTrack형) → C3 대상 ② **공통 난제**(전 방법 붕괴, plaster형) → C3 무관/한계 명시. 검출기가 이 구분 없이 "낮은 IoU = C3 켜라"고 하면 MCubeS에서 오발한다.

## 4. N4b 재설계 (기동 보류 → 사전 예측 재등록 필요)

1. **선행 작업**: 우리 per-class 전체 vs published per-class(CMNeXt/MMSFormer 표) 대조 → **RailTrack형 격차(우리 ≪ published) 존재 여부** 판정.
2. **예측 등록 후 기동**: RailTrack형 격차 존재 → "C3가 그 클래스 회복" 예측 / 부재 → "C3 중립~유해(MUSES형)" 예측. **어느 쪽이든 N3 검증 데이터**가 되지만 예측을 먼저 박아야 진단-구동 서사가 성립.
3. plaster는 N4b의 primary 지표에서 **제외**(공통 난제로 limitation 절 소재).

## 5. N1 seed824 병기 (예비)

MUSES seed824 val-best 82.05 (지표 기준 확인 중 — trainer면 vs 82.62 Δ−0.57 / 공식이면 vs 82.13 Δ−0.08). 어느 쪽이든 DELIVER(3.7pt outlier 구조)와 달리 **MUSES는 시드 안정 대역** — seed825로 확정 예정.

관련: [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md) H8(C3 축-특이) · plan N4/N4b · 문헌 출처 = 딥리서치 보고(CMNeXt 2303.01480·MMSFormer 2309.04001·MLE-SAM 2412.04220·MemorySAM 2503.06700·StitchFusion 2408.01343·Mul-VMamba KBS'25)
