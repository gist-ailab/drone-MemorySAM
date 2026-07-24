# P44-BMR + P45-FogStyle — 구현 기록 (2026-07-25)

> 설계 원본 = [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md) §3(P44)·§4(P45)·§7-b(coverage-aware).
> 상태: **구현 완료 (학습 대기)**. 학습 0건 — 아래 수치는 전부 합성 스모크 결과이고 성능 판정이 아니다.
> P42와 **직교**(P42 코드 경로 무수정, `P44.LOCAL_MASK.MODE: global`이 P42 의미를 재현) · m2f_head.py 무수정.

## 무엇을 만들었나

| ID | 변경 | 결선 위치 | config | 추론 영향 |
|----|------|-----------|--------|-----------|
| **B-1** | MMPareto gradient 통합 (2405.17730) — 주 CE와 per-modal aux CE의 gradient를 합산이 아니라 Pareto 방향 + **크기 복원**으로 통합 | `semseg/models/reliadino/mmpareto.py` (신규) + `train_reliadino.py` | `MODEL.P44.MMPARETO` | 없음 (학습 전용) |
| **B-2** | peer 상호증류 — per-modal aux logit 전 순서쌍 대칭 KL(teacher·stop-grad 없음) + 관계형 대응(토큰쌍 cos-sim) | `p44.py` 손실 + `fusion.py` aux dict | `P44.MUTUAL_KL` / `P44.REL_CORR` | 없음 |
| **B-3** | 커버리지 패턴 **국소** img 마스킹 (P42 전역 drop의 승격). MODE = global(=P42) / rect / **coverage**(같은 샘플의 lidar 리턴 패턴에서 blob 샘플링) | `p44.py` + `model.py::_p44_local_mask` | `P44.LOCAL_MASK` | 없음 (추론 full-modality) |
| **M-3** | hard-pixel aux — 마스킹 영역에서 fused가 틀린 픽셀에 생존 모달 aux CE 집중 | `model.py` | `P44.HARD_PIXEL_AUX` (기본 off) | 없음 |
| **V-1** | 결정론적 presence 재정규화 — 데이터가 **부재**한 픽셀에서 gate/router softmax를 present 모달 위로 재정규화. 학습 파라미터 0 | `p44.py` + `fusion.py::_gate`/`PerClassRouter` | `P44.VALIDITY_RENORM` (기본 off) | **있음** — P44에서 유일한 추론 경로 변경 |
| **P45** | feature-space fog style 섭동 + 일관성(img 브랜치 한정, 픽셀 증강 아님) | `p44.py::style_perturb` + `model.py` | `P45.FOGSTYLE` (기본 off) | 없음 |

전 항목 **키1 준수**: zero-init 잔차·attn-bias·학습형 추론 게이트를 하나도 추가하지 않았다(모듈/파라미터 신규 0개, 전부 손실·gradient·입력 레벨).
V-1은 "부재 판정"이지 "품질 추정"이 아니므로 반증된 3세대 게이트 계열과 구분된다(§7-b).

## B-1의 두 설계 결정 (리뷰 대상)

- **DDP = allreduce 이후 결합.** micro-step은 `model.no_sync()` 안에서 `autograd.grad` 2회(=DDP reducer 미개입), accumulation 경계에서 g_main·g_aux를 **각각** all_reduce 후 cos·α 계산. 전 rank가 동일 전역 gradient 위에서 동일 결합 → rank drift 0, 의미론이 단일 GPU 전역 배치와 일치. no_sync 없이 하면 다음 iteration에서 "Expected to have finished reduction"으로 죽는다.
- **AMP** = `scaler.scale()`로 미분한 뒤 누적 시 1/scale을 곱해 **cos 계산 전에 unscale**. bf16/AMP-off면 scale=1. fp16일 때는 GradScaler의 inf 기록이 없으므로 직접 유한성 검사 후 `new_scale`을 명시한다.
- 비용: backward 2회 → 스텝 시간 ~1.6~1.8× 예상. 탈출 밸브 `P44.MMPARETO.INTERVAL`(k step마다 적용).
- 그룹 = per-modal LoRA 슬라이스 3~4개 + shared trunk 1개. LoRA는 (M,·,·) 한 텐서지만 모달 m의 forward가 슬라이스 m만 건드리므로 합산 aux CE의 gradient가 이미 슬라이스 단위로 분리된다(추가 backward 불요).
- 진단 로그 `p44/cos_lora_<modal>` = 사전등록 게이트②("lidar-aux gradient와 주 gradient 내적 양수 전환")의 직접 측정치.

## 검증 (학습 0, 합성 스모크)

`/home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_p44.py` — CPU·초소형 텐서, **86 assert 전부 PASS** (3초).
핵심: all-off ⇒ fusion 출력·aux·1 optimizer step 파라미터가 baseline과 **bitwise 동일** · 충돌 시 투영 방향이 두 목표 모두에 하강(내적>0) + 크기 = ‖g_main‖+‖g_aux‖ · 비충돌 목표쌍은 단일 backward와 동일 업데이트(max|Δparam| 1.9e-9) · KL gradient가 **모든** 모달 브랜치 도달 · coverage 마스크 ⊆ lidar 발자국(팽창 마진 내) · V-1 부재 픽셀 가중 정확히 0 & 잔여 합 1 · config 3종 키 전달 + 미소비 키(오타) 0 · **tiny-ViT end-to-end**(전 토글 on) forward+분할 backward에서 aux gradient가 3개 모달 LoRA 그룹 전부에 도달·eval 결정론.

## 남은 확인 (학습 기동 시)

1. `train/p44_mask_rate` ep0 > 0 (B-3 무음 아님). DELIVER lidar 발자국이 조밀하면 coverage ≈ rect로 퇴화 — 이 로그로 판별.
2. `p44/cos_lora_lidar` 궤적 (게이트②).
3. 메모리/스텝시간 실측 후 BATCH_SIZE 재조정([[batch-sizing-policy]]).
4. ep30 사전등록 게이트: dMIoU(lidar) ≈0→>1 · CKA(img,lidar) ≥0.5 유지 · MUSES val ≥82.22.
