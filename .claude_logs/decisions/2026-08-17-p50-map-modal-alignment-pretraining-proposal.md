---
created: 2026-08-17
author: fable (MMSAM discussion 세션, model-proposal 절차 + 딥리서치 3축 A/B/C)
status: 🟢 승인(2026-08-17 user) — pseudo-모달 생성 파이프라인 구현부터 착수, 학습은 λ0.1 시드런 완주 후 슬롯
---

# P50-MAP 제안 — Modal Alignment Pretraining (2026-08-17)

> **한 줄**: 융합 구조를 더 바꾸지 않는다(캠페인이 소거 완료). 대신 VLM의 실제 교훈 — **"융합 모듈은 2차, 정렬 사전학습이 1차"** — 를 센서 도메인에 이식해, **LoRA+트렁크를 pseudo-모달 대량 데이터로 정렬 사전학습**한 뒤 타깃 파인튠한다. "정보 축"의 정공법이며 원장 H1~H15 어디에도 저촉되지 않는다.

## 1. 근거 (딥리서치 3축 수렴, 2026-08-17)

| 축 | 발견 | 출처 |
|---|---|---|
| C (VLM) | connector 선택은 "comparatively negligible"(MM1 ablation) · MLP가 Q-Former를 더 적은 데이터로 상회(LLaVA-1.5) · 압축형(Q-Former/resampler)은 dense 과제에서 역효과(Idefics2·Cambrian) → **우리 MLP 트렁크는 VLM 정통, 공백은 stage-1 정렬 부재** | 2403.09611 · 2310.03744 · 2405.02246 · 2406.16860 |
| C (센서) | 동일 패턴: StitchFusion(공유 trunk+adapter, DELIVER 68.20)이 정교 융합(GeminiFusion 66.9)을 이김 · Q-Former/이산 토큰화 이식 사례 0 + dense 역효과 예측 | 2408.01343 · 2406.01210 |
| A | **OmniSegmentor(NeurIPS'25)**: ImageNet 1.2M을 5 pseudo-모달로 렌더링(ImageNeXt) → 융합 스택 사전학습 → 모달 이득 +2.6, KITTI-360 **+5.1이 사전학습만으로** · DFormer(ICLR'24)·MultiMAE 선행 정합. 검증: 그들 DELIVER는 val-레인지(68.0, 소형 백본)라 우리 기준선 위협 아님 — **기제 증거**로 사용 | 2509.15096 · 2309.09668 · 2204.01678 |
| B (참고) | event 스트림 인코딩 천장 ≤+1(문헌 한계기여 +0.4~0.9와 일치) → 후순위 · radar 물리 게이트는 각주급 | 2401.12761 ablation 등 |
| 우리 실측 | LoRA+트렁크는 타깃 3~4천 장으로 **from scratch** — 사전학습 0. 융합 정교화 상한 ~1.2pt(MM-SA Tab.9 + 우리 12세대) | 원장 H1~H4·H14 |

## 2. 설계

- **사전학습 데이터**: 🟢 **코퍼스 = Places365 확정(user 2026-08-17)** — 공개 다운로드·장면 중심(주행 도메인 정합). 원본 tar = `/ailab_mat2/personal/jemo_maeng/dset/Places365/`(전 서버 공유), 생성 작업은 서버 로컬 SSD에서 후 tar로 회수(sshfs 소파일 금지 규칙). 서브셋 200k에 Omnidata depth + pseudo-LiDAR range-view + N-ImageNet event. thermal 불요(우리 벤치에 없음). 시작 규모 **200~300k장**(프로브), 통과 시 확장.
- **사전학습 대상**: **백본 frozen 유지**(DINOv3-L) — LoRA(모달별) + 융합 트렁크 + FPN 헤드만 학습(≈50M trainable → 4090 함대 가능). 목적함수 = pseudo-라벨 seg(ImageNet엔 seg GT가 없으므로: SAM/DINO 기반 pseudo-mask 또는 MultiMAE식 cross-modal masked reconstruction — 프로브에서 두 팔 비교).
- **파인튠**: 기존 통일 레시피 그대로(DELIVER는 +C3). **추론 아키텍처 무변경** — 단일-모델 원칙 유지, 순수 초기화 개선.
- OmniSegmentor와의 차별(노벨티): ① frozen foundation 위 **어댑터·트렁크만** 정렬(그들은 소형 백본 전체) ② MUSES 최초 적용 + 야간/조건축 분해로 "사전학습이 정보 결핍을 채우는가" 분석(우리 체계) ③ 진단-짝 손실과 결합. 공정성: 사전학습은 ImageNet 파생 합성 — 경쟁 백본들의 사전학습과 동급 지위, 벤치 GT 무관.

## 3. 게이트 (사전 등록)

| 단계 | 게이트 |
|---|---|
| **프로브** (200k×~30ep 사전학습 → DELIVER 파인튠 1런) | base(무사전학습, 56.99 계열 레시피) 대비 legal test **≥ +0.5 → 본 사전학습 확장** / +0.0~0.5 → 스케일 판단 보류 / **< 0 → 폐기** |
| 본 실험 | DELIVER test ≥ 57.35(SOTA) · MUSES val 82.13 초과 → 공식 test 제출 판단 · 야간 서브셋 이득 ≥ 전체 이득(정보 결핍 명제 검증) |
| falsifiable | 사전학습 있/없 쌍의 유일 변수 = 초기화 — 이득이 나면 원인 귀속이 자명 |

## 4. 비용/실행

pseudo-모달 생성(기성 도구 GPU 추론, ~수일) → 사전학습 프로브(4090 2~4장 × 1~2일) → 파인튠(기존 레시피). λ0.1 시드 완주 후 jarvis 슬롯. 생성 파이프라인 구현은 워커 위임 + 검수 규약.

관련: 원장 [research/hypothesis-ledger.md](../research/hypothesis-ledger.md)(H1~H15 저촉 없음 확인) · [2026-08-10-p49-air-asymmetric-injection-proposal.md](2026-08-10-p49-air-asymmetric-injection-proposal.md)(§6 단일-모델 원칙 — 본 제안은 추론 그래프 무변경이라 자동 충족)
