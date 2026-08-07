---
created: 2026-08-08
author: fable (MMSAM discussion 세션)
status: 프로브 **기동됨** (2026-08-08 06:38 KST, jarvis GPU1/6/7) — 결과 대기
---

> 🔴 **기동 시 발견 — 게이트 적용 전 반드시 읽을 것 (2026-08-08, opus 세션)**
>
> **1. 로그 mIoU 를 그대로 G-P1 에 넣지 마라. 조건 서브셋에서는 희석된다.**
> 평가기는 등장하지 않는 클래스도 0.00 으로 19 로 나눈다. fog_night val 은 **25장뿐**이라
> 19 클래스 중 **8개가 아예 등장하지 않는다**(traffic light·person·rider·truck·bus·train·
> motorcycle·bicycle). 실측: 존재 11클래스 평균 76.93 → 로그 mIoU **44.54** (= 846.28/19).
> **Δ 도 같은 비율로 희석되므로 존재 클래스 평균으로 환산해야 한다** (fog_night 계수 19/11 ≈ 1.73).
> 환산하지 않으면 참 Δ +1.0 이 +0.58 로 읽혀 **방향을 잘못 폐기**한다.
>
> **2. fog_night 단독 프로브는 과소검정이다.** 위 8개 결석 클래스가 바로 MUSES 격차의
> 핵심(소형 things)이라, fog_night 25장으로는 그 축을 아예 관측할 수 없다.
> → `night`(val 100장) / `day`(val 150장) 쌍을 **추가로** 기동했다. G-P2 대조도
> clear_day(50장) 대신 통계력이 나은 `day` 를 쓴다.
>
> **3. 기동된 런** (전부 base = P39.1-rank 3모달 seed2 `epoch208_82.62_top1`, 1024², BS1, eff-batch 16 동일)
>
> | 런 | 조건 | train/val 장수 | ep | LR | GPU | 역할 |
> |---|---|---|---|---|---|---|
> | `base_fognight` | fog_night | 150/25 | 1 | **0.0** | 1 | 기준선 — **완료, mIoU 44.54 (존재11 76.93)** |
> | `fognight_lr1e4` | fog_night | 150/25 | 40 | 1e-4 | 6 | 전문가 |
> | `fognight_lr3e4` | fog_night | 150/25 | 40 | 3e-4 | 7 | 전문가 — **LR 민감도**(과소튜닝 거짓음성 방어) |
> | `base_night` / `night_lr1e4` | night | 600/100 | 1 / 20 | 0.0 / 1e-4 | 1 | 헤드라인 |
> | `base_day` / `day_lr1e4` | day | 900/150 | 1 / 13 | 0.0 / 1e-4 | 1 | G-P2 대조 (최적화 예산 정합: 12000 vs 11700 samples) |
>
> **기준선을 LR 0 런으로 잡은 이유**: 전문가와 **완전히 같은 평가 경로**로 재야 Δ 가 성립한다.
> 공개된 per-condition 수치(fog_night 69.610 등)는 다른 평가기 산출이라 직접 빼면 안 된다.
>
> **코드**: `DATASET.CASE`(조건 셀 제한) + `MODEL.FINETUNE_INIT`(가중치만, optimizer/scheduler/epoch 초기화)
> 를 `train_reliadino.py` 에 추가 — develop 8236773 / 4fb0979. FINETUNE_INIT×AUTO_RESUME 동시 사용은
> 하드 가드로 차단(epoch 0 되돌림 → 무한 재시작). 기동 검증: missing=0 unexpected=0, RANDOM INIT 0건.
>
> ⚠️ **미해결**: LR 1e-4/3e-4 는 임의 선택이다. 셋 다 Δ≈0 이면 "천장이 낮다"와 "미세조정이
> 부족하다"가 구분되지 않는다 — LR 민감도 런이 그 방어책이지만 완전하지는 않다.

# 조건×클래스 어댑터(CEA) 방향 — oracle 프로브 선행 제안 (2026-08-08)

> **한 줄**: "모달별 유용 정보는 환경조건×물체종류에 따라 달라진다"는 가설을 어댑터 **추출 단계의 조건부 전문화**로 구현하는 차기(가칭 P49) 방향. 단, 본학습 전에 **oracle 조건-전문가 상한 프로브**로 천장을 먼저 재고, 게이트 미달이면 방향 전체를 접는다.
>
> 배경 논의 = 2026-08-08 discussion 세션 (계보 공통 방향 분석). SOTA 진단 artifact("MemorySAM — SOTA까지 무엇으로 가는가", 2026-08-08)와 세트.

## 1. 동기 — 가설은 실측이 지지하고, 죽은 것은 구현 방식이다

계보 12세대의 공통 가설 "신뢰도/유용도에 따른 모달 가중"에서, **반증된 것은 가설이 아니라 특정 구현 계열**이다:

| 구현 계열 | 반증 근거 |
|---|---|
| 학습된 융합 게이트 (UAMM/SoftMoE/quality gating, P10~P27) | gate 상수수렴 — 전원 P9 미돌파 |
| attention logit 신뢰도 주입 (RBMA attn-bias, P28~P36) | P32 유의한 순손해(p=4.5e-22), DINOv3 계보 Δ≈0 |
| 추론 시 재가중 (gate/calib/veto) | fog_night·thin-class 유해 판정 (2026-07-20) |

반면 **조건×모달·클래스×모달 상호작용 자체는 실재**한다:
- drop-lidar dMIoU: day 0.64 vs fog_night 7.19~7.39 (P39.1, 비RGB 기여가 악조건에 집중)
- per-class router의 생존 기여가 RailTrack 한 클래스에 집중 (+11~16)
- DELIVER 붕괴는 클래스축(thin 정적 구조물), MUSES 붕괴는 조건축(fog_night 69.610) — 축이 다름

실패 원인 3가지(설계 제약으로 역변환):
1. **분화 강제 신호 부재** — CE만으로 게이트는 분화하지 않는다. 분화가 일어난 유일 사례는 강제 장치가 있었다(P39-V5 path-dropout 경쟁, router 직접 CE).
2. **얹으면 흡수** — dense/query 이중 중복 시스템에서 추론 경로 추가 모듈은 죽는다(P43/P44/P47). 생존자는 학습 전용 손실(P46-C3)뿐.
3. **평균 최적성 함정** — 조건 i.i.d. 혼합 학습에서는 정적 가중이 평균적으로 준최적이라 adaptive가 이길 여지가 작다. 조건 분리 감독이 없었다.

**선점 지형**: 조건축 단독 융합 변조는 CAFuser(condition token)가 선점. 미점유 코너 = **조건×클래스 2축 조건화를 융합 가중이 아니라 어댑터(추출) 단계에 넣고, 분화를 손실로 강제**하는 것. 주의 — P9~P11 SoftMoE는 무조건화+CE-only라 원인 1로 실패했고, **입력 조건화 버전(P12)은 설계만 되고 한 번도 학습되지 않았다**. 즉 이 코너는 반증이 아니라 미검증.

## 2. 프로브 — oracle 조건-전문가 상한 측정 (본학습 전 필수 게이트)

**질문**: 조건별로 어댑터가 완벽히 전문화된다면 최대 몇 점을 버는가? (이 상한이 낮으면 CEA 방향 전체의 천장이 낮다.)

**설계**:
- Base: P39.1-rank 3모달 seed2 ckpt (MUSES 최고, test 79.788)
- 조건 서브셋 {fog_night(최악), night_clear, day_clear(대조군)} 각각에 대해 **LoRA(+trunk)만 짧게 미세조정** (백본 frozen 유지, base와 동일 해상도·레시피, ~5-10 epoch)
- 평가: 각 조건 전용 val 서브셋에서 "전문가 ckpt vs 공용 ckpt" ΔmIoU = **oracle 상한**
- (선택 2차) DELIVER 동일 프로브: 조건 {night, fog} — 클래스축과의 분리 확인용

**GPU 비용**: 조건당 1런 × 빈 GPU 1~2장 × 수 시간(미세조정이라 짧음) = 총 반나절~1일급. **본학습(A100급 4장 × 수일)은 게이트 통과 후에만.** 기동은 experiments/plan.md 큐 뒤(현재 1024² 판정·seed3이 선순위), 서버는 기동 시점의 빈 슬롯(jarvis/yeon)에서 `remote_exp.sh status`로 선택.

**사전 등록 게이트**:
- **G-P1 (천장)**: oracle Δ(fog_night) **< +1.0 → 방향 폐기**(천장 부족, 재제안 금지 목록에 추가). ≥ +2.0 → P49 설계 착수. +1.0~2.0 사이 → DELIVER 프로브 결과와 합산 판단.
- **G-P2 (대조)**: day_clear의 oracle Δ가 fog_night와 비슷하게 크면, 이득의 원천이 "조건 전문화"가 아니라 "그냥 추가 학습"이라는 뜻 → 해석 기각. fog_night ≫ day_clear 여야 통과.

## 3. P49-CEA 스케치 (게이트 통과 시에만 구체화)

- **구조**: 모달별 어댑터를 소수(예: 4)의 LoRA expert 혼합으로 확장, 라우팅은 조건 신호 × 클래스 prototype 유사도의 2축 조건부. 같은 thermal 인코더도 "야간의 사람"과 "주간의 노면"에서 다른 expert 조합이 활성화.
- **강제 장치**(원인 1·3 처방, 전부 실증/구현된 부품 재사용): ① 조건 메타데이터(MUSES/DELIVER 보유)로 라우팅 직접 감독 ② C3 prototype(클래스축) 재사용 ③ P47-2 per-modal CE(modality laziness 차단, 구현 완료 미기동).
- **흡수 회피**(원인 2): expert 혼합은 추출 단계라 dense/query 중복의 하류가 아님. 추론 시 조건 신호는 CAFuser식 예측 토큰(경량)로 대체.
- **사전 등록 게이트(본학습)**: ① 기제 — expert 활성이 조건×클래스로 실분화(라우팅 엔트로피·조건별 활성 분포; 상수수렴이면 즉시 폐기) ② 성능 — 이득이 fog_night·night에 집중 + clear/day 무손실(P46-C3 MUSES 이식 실패 −0.765의 재림 방지) ③ 차별화 — 동일 파라미터 정적 LoRA 대비 + condition-token식 융합 변조 대비 우위.

## 4. 논문 연결 (CVPR/RA-L 트랙과의 관계)

- 이 제안은 **CVPR급 novelty 후보**(어댑터 구조 자체의 기여)다. 단순 per-modal LoRA는 진부하다는 판단(user, 2026-08-08)에 대한 응답.
- 현행 트랙(P46 @1024² 완주 판정 → DELIVER 돌파 여부 → CVPR/RA-L 분기)과 **독립적으로 병렬** 진행 가능 — 프로브는 현행 큐에 영향 없음.
- 실패해도 손실이 작다: 프로브 자체가 "조건 전문화 상한" 실측이라 논문 분석 절(modality-condition interaction)의 재료가 된다.

## 5. 실행 체크리스트

- [ ] 프로브 config 작성 (조건 필터 로더 — MUSES metadata 기반 서브셋; 신규 코드 최소)
- [ ] 현행 큐(1024² 판정, seed3) 종료 후 빈 GPU에 기동 — 학습 기동·조회는 sonnet 위임, 판정은 상위 모델
- [ ] G-P1/G-P2 판정 → 이 문서 상단 status 갱신 + experiments/registry.md 행 추가
- [ ] 통과 시 P49-CEA 상세 설계 문서 신설 (model-proposal 스킬 절차)

관련: [2026-08-05-p48-instance-supervision-proposal.md](2026-08-05-p48-instance-supervision-proposal.md)(폐기 판정에 시점 오류 논란 — 재판정 별도 필요) · [experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md](../experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md)(흡수 실증) · [[p34-reliadino]] · [[muses-dataset-setup]]
