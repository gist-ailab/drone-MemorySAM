# 벤치마크 & 모달리티 확장 로드맵 (모든 세션 공유)
updated: 2026-07-17 (P38 구현 세션) · 배경: 사용자 지시 — "실험 계획 시 DELIVER/MUSES 밖 벤치(MCubeS 등)와 모달리티 ablation까지 고려, 모든 세션이 알도록 로깅"

## 원칙
- 논문 서사: 모델 내부 신호(신뢰도) 기반 modality routing — 외부 신호(CLIP text·GT-depth) 불사용. DGFusion/CAFuser와 같은 잣대(공정 aug, val-best ckpt, PQ) 위에서 비교.
- 새 벤치 착수 게이트: DELIVER P38 판정 통과 후 (실패 시 head 진단이 우선).

## Tier-1 — 논문 주표 (DGFusion/CAFuser 직접 대응)
| 벤치 | 모달리티 | 지표 | 상태 |
|---|---|---|---|
| DELIVER | img/depth/event/lidar | mIoU (val/test) | P36 fair 67.74/55.62 · P38 대기 |
| MUSES | frame/lidar/radar/event | **PQ**(주) + semantic mIoU | semantic 78.979 제출됨 · PQ는 P38 head로 가능해짐 — P38-DELIVER 판정 후 MUSES 학습·제출 |

## Tier-2 — 일반화 표 (관련연구 공통 벤치, 소형·저비용 순)
| 벤치 | 모달리티 | 지표 | 비고 |
|---|---|---|---|
| MCubeS | RGB/AoLP/DoLP/NIR | mIoU | CMNeXt·MAGIC·StitchFusion 공통 표 — 500장 소형, 1~2일/런, Tier-2 1순위 |
| FMB | RGB/Thermal | mIoU | RGB-T 표준 |
| PST900 | RGB/Thermal | mIoU | 소형, 저비용 |
| MULTIAQUA | RGB/Thermal/LiDAR | mIoU | 프로젝트 기원 — 챌린지·도메인 확장 |

## 모달리티 ablation 계획 (논문 표 재료)
1. **leave-one-out** (벤치별): full-modal 학습 모델에 eval-time 모달 마스킹; 여력 시 재학습 행 병기.
2. **MUSES radar 서사**: radar는 lidar와 잉여(SOTA조차 +0.6) — 우리 −0.72는 radar 무능이 아니라 잉여성의 발현. 신뢰도 라우팅이 lidar-degraded 조건에서 radar를 "대체 range 신호"로 쓰는지 **per-condition(night/rain/fog/snow) 수치**로 검증 — 이 주장 방어에는 누적 ablation + per-condition 표가 필수.
3. **DELIVER 센서 열화 조건**(Motion Blur/Over-/Under-Exposure/LiDAR-Jitter/Event Low-res 등) per-condition 표.
4. **MAGIC식 random modality-drop robustness** (eval-time, 학습 0).

## 실행 순서 (P38 기준)
P38 DELIVER(대기) → 판정 → [통과] MUSES P38(PQ 학습·제출) + MCubeS/FMB 병렬 착수 → ablation 표 → [미통과] head 진단 후 P38.1
