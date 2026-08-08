---
created: 2026-08-08 18:30
author: fable (MMSAM discussion 세션)
type: 실험 의뢰서 (conventions §4 의뢰서 규약 — 이 문서만 읽고 실행 가능해야 함)
status: 의뢰 등재 (코디네이터 승인 2026-08-08) — 실행 세션 미배정
---

# 실험 의뢰서 — ProbeA2: 백본 스케일링 프로브 (상한 H+ / 하한 S+·B, RGB 단독)

## ① 의뢰 취지

한 프로브로 두 질문을 동시에 닫는다:
- **상한**: DINOv3-L(현행) 위로 표현력 천장이 남았는가 — "남은 격차 = RGB 표현력" 서사([research/hypothesis-ledger.md](../research/hypothesis-ledger.md) H5′·H6)의 직접 검증. 양성이면 차기 본 모델 = 백본 승급이 확정된다.
- **하한(공정성 방어)**: 경쟁군 Swin-T(28M)와 용량 정합인 **DINOv3-S+(~29M)** 에서 우리 스택이 어디까지 가는가 — "백본이 커서 잘 나오는 것" 비판(내부 P34 판정에서 이미 제기)에 대한 논문 Table 방어 행 + 스케일링 곡선(그림 재료).

근거 맥락: 적응 가설 계열 폐쇄로(원장 H1~H4) 남은 이득 축이 표현력·해상도·학습신호로 좁혀짐. 표현력은 계보 최대 단일 변수(ProbeA1: SAM2→DINOv3-L **+11.6**).

## ② 실행 스펙

| 항목 | 내용 |
|---|---|
| 프로토콜 | **ProbeA1 통제 프로브 재사용** — frozen backbone feature + 동일 경량 seg head, 동일 학습 예산. 원 프로토콜/코드 = NAS `analysis_logs/ProbeA1_dinov3_20260712/`(리포트에 실행 스크립트 명시돼 있는지 실행 세션이 먼저 확인) |
| 백본 | DINOv3 **S+(~29M) / B(~86M) / L(~300M, 기준선 — ProbeA1 접점) / H+(~840M)**. 7B은 옵션(§⑥ 게이트 중간대역일 때만) |
| 모달 | **RGB 단독** (표현력 축 분리 — 융합 개입 없음) |
| 데이터 | **MUSES** train 서브셋(2~4천 장, head 학습용) + val 250(측정). DELIVER 반복은 선택(시간 남으면) |
| 실행 방식 | **2단 분리**: (1) feature 추출 = 추론 전용 bf16, 활성화 미보존 → 24GB에 H+ 여유, 7B도 가능 추정(OOM 시 §⑤ 폴백) (2) 캐시 feature 위 head 학습 = 메모리 사소 |
| 자원 | jarvis 4090 1~2장 (GPU1~5 유휴 확인됨 2026-08-08 15:13). **A100/B200 불필요** — 본 모델 학습만 B200(P46 완주 후 슬롯) |
| 캐시 위치 | feature 캐시는 로컬 SSD(`/mnt/SSD2` 데이터셋 캐시 규약) — sshfs에 쓰지 말 것 |

## ③ 구현

- **코드 변경 범위**: ProbeA1 스크립트가 재사용 가능하면 백본 로더 4종 분기만 추가(구현 ~0). 재사용 불가면 feature 추출 스크립트 + cached-feature head 학습 스크립트 신규(소규모).
- **구현 주체**: 신규 작성 필요 시 **워커 위임**(GLM.md 규약 — 위임 직전 user에게 워커 선택 확인; 소규모·단순이라 glmcode도 가능, 정확도 우선이면 labcode).
- **검수 게이트(기동 전)**: code-review 규약 — fresh-eyes + 스모크(백본 4종 로드 확인·**RANDOM INIT 0건**·feature shape 실측·head 학습 1 epoch 손실 하강). ⚠️ HF 다운로드 필요 — `HF_HUB_OFFLINE=1` 상태로 fresh 로드 금지([[hpca100-cudnn-fix]] 사고: offline이 RANDOM INIT 유발).

## ④ 총 ETA

가중치 확보(~1h) + 구현/검수(0~1일, 재사용 여부에 따라) + 추출(백본당 1~4h, 2장 병렬) + head 학습(백본당 ~1h) + 측정 ≈ **1~2일**.

## ⑤ 폴백 (사전 정의 — 즉석 판단 금지)

- H+/7B 가중치 확보 실패(미공개·라이선스) → **확보 가능한 최대 크기까지만** 측정하고 그 사실을 기록. 대체 백본(EVA 등)으로 바꾸지 말 것(통제 변수 오염).
- 7B 추출 OOM@24GB → H+까지로 종료(7B은 A100 확보 시 후속).
- ProbeA1 코드가 회수 불가/재현 불가 → head 사양을 새로 고정하되 **4종 전부 동일 사양**으로 — 이 경우 L 수치가 ProbeA1의 +11.6과 직접 비교 불가함을 결과 문서에 명시.

## ⑥ 사전 등록 게이트 (시점 = 4종 head 학습 완료 후 MUSES val 측정치로)

- **G-A2-상한**: mIoU(H+) − mIoU(L) **≥ +1.5** → 백본 승급 본설계 착수(B200 슬롯, 4모달 per-modal LoRA) / **< +0.5** → 표현력 축 소진으로 원장 H6에 기록(차기 방향에서 백본 승급 제외) / **+0.5~1.5** → 7B 추가 측정 후 재판정.
- **G-A2-하한**: mIoU(L) − mIoU(S+) **≤ 3.0** → 용량 정합 본런(S+ 전체 레시피)을 논문 슬롯으로 확정(방어 Table 행) / **> 3.0** → 본런 없이 "방법 기여 주장은 대형 백본 전제"로 논문 스코프를 명시(정직 공개) — 어느 쪽이든 스케일링 곡선 4점은 논문 그림으로 회수.
- ⚠️ 프로브 절대치를 본 모델 수치·리더보드와 비교 금지(head가 다름). **백본 간 Δ만** 유효.

## ⑦ 결과 기록처 (실행 세션 의무)

1. `experiments/registry.md` 행 추가 (예: `jarvis_muses_probea2_backbone_scaling`)
2. `experiments/analysis/2026-08-XX-probea2-backbone-scaling.md` 신설 (4종 수치 원표 + 게이트 적용 + 스케일링 곡선)
3. [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) **H6 행 갱신**(스케일링 실측 반영) + 이 문서 status 갱신
4. `experiments/plan.md` #9 행 상태 갱신 + `status/history-2026H2.md` 완료 통보 엔트리

관련: [2026-08-08-condexpert-adapter-probe-proposal.md](2026-08-08-condexpert-adapter-probe-proposal.md)(적응 계열 폐쇄 — 본 의뢰의 배경) · ProbeA1 = NAS `analysis_logs/ProbeA1_dinov3_20260712/` · [meta/conventions.md](../meta/conventions.md) §4 의뢰서 규약
