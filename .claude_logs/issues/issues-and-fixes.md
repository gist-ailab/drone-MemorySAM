---
legacy_id: 04
legacy_file: 04_issues_and_fixes.md
moved: 2026-07-08
---

# 이슈 및 해결 기록 (Issues & Fixes)

> 최종 업데이트: 2026-08-06
> 코딩 세션은 이 파일을 읽고 동일한 실수를 반복하지 말 것

---

## 🔎 이슈 상태 인덱스 (먼저 여기 보고 점프)

> **현재 액션 필요 여부는 이 표의 상태 컬럼으로 판단**할 것. `[해결]`된 ISSUE-021/020/019/018/016은 2026-06-24 "해결된 이슈" 섹션으로 물리 이동 완료(하단 "(이관됨)" 하위). 표의 ✅ 항목은 이력용이다.

| ID | 상태 | 한 줄 |
|----|------|-------|
| **ISSUE-032** | ✅ **수정**(2026-08-06) | `val.py` `evaluate()`(val 모드 함수)에 `@torch.no_grad()` 누락 — ViT-L 전체 autograd 그래프 유지로 **iteration 1에서 100% OOM**(ckpt 종류 무관). `run_test_inference()`(test 모드)는 정상 데코레이션돼 있어 test만 성공. 커밋 c0e413c로 1줄 수정. 상세: 하단 ISSUE-032 |
| **ISSUE-030** | ✅ **수정(2026-08-06)** | `train_reliadino.py` `last_checkpoint.pth`+topK best 저장이 임시파일+rename 없이 최종 경로에 직접 덮어써 **비원자적**이었음 — 저장 도중 사망(preempt/OOM/SIGKILL) 시 파일 손상으로 AUTO_RESUME 실패 위험. `_atomic_save`(tmp+os.replace) 헬퍼로 양쪽 다 수정, 스모크 3건 통과(커밋 0bc65f5). 상세: 하단 ISSUE-030 |
| **ISSUE-031** | 🟡 **프로세스 결함, 재발방지 적용(2026-08-04)** | hpca100 P47-1 `BATCH_SIZE:1`이 A100(40GB) 기준 재프로파일 없이 3090/4090용 값 그대로 사용됨 — 실측 rank당 24.6GB/40GB=60%(정책 목표 85~90% 미달). 이 런은 재기동 위험·1-변수 순수성 이유로 변경 안 함, **이후 A100 신규 기동 전 `torch.cuda.max_memory_allocated()` 프로파일 필수화**로 재발방지. 상세: 하단 ISSUE-031 |
| **ISSUE-028** | ✅ **수정(2026-07-29)** | **P46-CTR jarvis OOM은 누수가 아니라 warmup 계단이다.** C2_MCC/C3_PROTO `WARMUP_EP:5` + epoch 0-index → 로그상 **ep6(=epoch 5) iter0이 보조 branch·EMA teacher·proto 손실이 최초로 도는 지점**. ep1~5(=epoch 0~4)는 P39.1-base 그대로라 15.2GB였고 P46 비용은 **한 번도 측정된 적이 없다**. 부수적으로 상수 오버헤드 4건(보조 `_baux` 미도달 서브그래프·루프 지역변수·teacher `_last_*` 캐시·PrototypeBank full-copy)을 수정. **peak 2-그래프 구조 자체는 설계라 그대로** → BS1에서도 24GB로 부족. 상세: 하단 ISSUE-028 |
| **ISSUE-026** | ✅ **수정(2026-07-21)** | ColorAugSSD brightness가 uint8(0-255) 입력을 [0,1] 클램프 → 발화 샘플(p=0.5) RGB가 백색 상수로 붕괴(사실상 RGB-dropout 0.5). **07-16 이후 DGFUSION_AUG:true DELIVER 학습 전부 오염**(jarvis P37a-DELIVER/P37b(사망런), hpca100 P38-DELIVER 완주분·**P39-DPC resume 진행 중**, yeon 스모크). MUSES 전 계보 무영향. **P38-DELIVER/P39-DELIVER 게이트 판정 보류.** 상세: 하단 ISSUE-026 |
| **ISSUE-029** | ✅ **수정(2026-07-28)** | hpca100 HF 백본 이중고장 — `HF_HUB_OFFLINE=1`일 때 DINOv3+DINOv2 폴백 둘 다 local cache lookup 실패로 **RANDOM INIT**(경고 없이 조용히 진행됨), offline 미설정 시엔 반대로 HF Hub 온라인 조회 단계에서 **정체(hang)**. 최초 진단(P44-BMR 과균형)은 오진 — 실제 원인은 백본. `RELIADINO_LOCAL_BACKBONE` env로 로컬 safetensors 직접 로드하는 우회 코드로 해결(697a10a). 저조(mIoU 급락) 발생 시 **백본 로드 라인부터 먼저 확인할 것**. 상세: 하단 ISSUE-029 |
| **ISSUE-027** | ✅ **가드 추가(2026-07-21)** | GRADIENT_CHECKPOINT=true 시 timm non-reentrant 재계산이 stale active_modality로 비최종 모달 LoRA gradient 오염(무경고). encoder 강제 off 가드 + 체크인 configs 9종 false로 수정. 실피해는 bengio 사망런·yeon 스모크 등 한정적. 상세: 하단 ISSUE-027 |
| **ISSUE-025** | ✅ **해결(2026-07-21)** | MUSES radar 디코딩 3중 버그 — `_open_radar` 폴스루+디스패치 오배선+`RADAR_RANGE_MAX` 미정의로 100m 클립(실측 유효픽셀 2.76% 포화, 센서 캡=150.0m) + height 채널 0.25 상수 오염. develop에서 수정 완료. **영향은 4모달(radar 포함) 실험만, 3모달 전 계보 무영향.** 상세: 하단 ISSUE-025 |
| **ISSUE-024** | 🟡 **OPEN (조건부 — P37b kill-gate 생존 시 수정)** | P37b `classtoken.py`의 `mask_proj`(attn-mask 예측기)가 threshold 비교(비미분)로만 쓰여 gradient 미도달 → 영구 random init, masked attention이 사실상 random 마스킹. P38 `m2f_head.py`가 올바른 수정 패턴(`_attn_bias`) 보유. 전수조사에서 minor 다수 추가 확정·수정됨(2026-07-21 커밋 참조). 상세: 하단 ISSUE-024 |
| **ISSUE-023** | ✅ **완화 완료(2026-07-08)** / 🟡 근본해결 대기 | **/mnt/HDD2 ENOSPC = NTFS MFT 레코드 고갈** — 아카이브 27k+파일을 drone NAS로 전량 소산해 레코드 해방, **쓰기 정상화 검증 완료(2,000파일 연속 생성 OK)**. 단 대량 파일 쓰기(수만 개)는 재마운트/Windows 검증 전까지 자제. 상세: 하단 ISSUE-023 |
| **ISSUE-022** | ✅ **해결(2026-07-03)** | **P27.forward가 `_fuse_outputs` 훅 미호출 → P30 learned router 200ep 내내 미실행** (P31.2 훅 호출로 수정; P30 결과 = router 미참여로 재해석) |
| ISSUE-021 | ✅ 해결 | SAM3-RBMA sem_head BatchNorm→GroupNorm (train/eval 불일치) |
| ISSUE-020 | ✅ 해결 | SAM3-RBMA sam3.pt 백본 0개 로드(random) prefix remap |
| ISSUE-019 | ✅ 해결 | P26 entropy NaN (LogBackward gradient explosion) |
| ISSUE-018 | ✅ 해결 | P9/P22 UAMM 전후 피쳐 시각화 지원 |
| ISSUE-016 | ✅ 해결 | P26 DELIVER 런타임 에러 6건 |
| **ISSUE-013** | 🔴 **미해결(긴급)** | P24 Teacher signal sigmoid→CE 기반 수정 필요 |
| **ISSUE-017** | 🟠 **미수정** | val_multiaqua_detailed.py 시각화 버그 2건 |
| ISSUE-001 | 🟠 부분 | Val NIGHT_AUG 미적용 → 모델 선택 기준 |
| ISSUE-002 | 🟠 진행 | MoE Expert Collapse (E1 사망) |
| ISSUE-003 | 🟠 진행 | CrossModalFusionHead 상수 출력 (RBMA로 대응 중) |
| ISSUE-007 | 🟠 부분 | CRM/ZERO Overfitting (Night-Val↑ Test↓) |
| ISSUE-010 | 🟠 부분 | 로깅 시스템 개선 |
| ISSUE-006 | ⚪ 구현필요 | Aux Head Mask 시각화 (Energy 검증) |
| ISSUE-004 | ⚪ 예정 | Spatial-wise Confidence Weighting (P15) |
| ISSUE-005 | 💡 아이디어 | Diffusion 기반 Day→Night 합성 |
| ISSUE-008 | 📌 구조적 | Aux Head 품질 한계 (frozen backbone) |
| ISSUE-009 | 📌 결정 | Energy→Calibrated Entropy 교체 |
| ISSUE-011 | 📌 설계 | Fusion Head multi-scale 미활용 |
| ISSUE-012 | 🖥 하드웨어 | P23 A100 80GB OOM (deformable conv) |
| ISSUE-014 | ⚪ 개선 | RandomResizedCrop 패딩 위치 고정 |
| ISSUE-015 | ✅ 설계반영 | P25 구조적 문제 7가지 → P26 v5 |
| RESOLVED-001~004 | ✅ 해결 | 하단 "해결된 이슈" 섹션 참조 |

> ✅ 정리 완료(2026-06-24): `[해결]` ISSUE-021/020/019/018/016을 "해결된 이슈" 섹션으로 물리 이동함. 이제 "열린 이슈" 섹션은 ISSUE-001부터 시작(실제 미해결/진행 항목 위주).

---

### ISSUE-032: `val.py evaluate()`에 `@torch.no_grad()` 누락 — val 모드 100% OOM [수정, 2026-08-06]

**위치**: `val.py:1182` `def evaluate(...)` (val 모드에서 호출되는 함수) — 바로 위(1179)는 무관한 헬퍼 `_pad_rows_to_same_width`의 끝일 뿐, `evaluate()` 자체엔 `@torch.no_grad()`도 `with torch.no_grad():`도 없었다. `model.eval()`은 dropout/BN 동작만 바꿀 뿐 autograd 추적을 끄지 않는다.

**증상**: final-iter(`last_checkpoint.pth`) 11건 재평가 배치(`/tmp/finaliter_batch.sh`, jarvis)에서 **val 모드가 전부 PARSE_FAIL**. 원인은 파싱 실패가 아니라 진짜 크래시 — `CONTROL_valbest_ep62`(검증용 정상 val-best 체크포인트)조차 **iteration 1/2005**에서 `CUDA OutOfMemoryError`. ckpt 종류(final-iter/val-best) 무관하게 100% 재현.

**진단**: ViT-L/16 + LoRA + M2F 전체 forward에서 autograd 그래프가 유지된 채 backward 없이 매 iteration 쌓이므로 즉시 OOM. test 모드는 별도 함수 `run_test_inference()`(`val.py:1466`, 데코레이터 1465에 정상 존재)를 호출해 무사 — val만 죽는 이유가 이걸로 설명된다.

**수정**: `def evaluate` 바로 위에 `@torch.no_grad()` 1줄 추가. 다른 변경 없음. 커밋 `c0e413c`(develop push 완료).

**교훈**: 🔴 **OOM을 GPU 상주메모리·배치크기 문제로 진단하기 전에 `no_grad`/`inference_mode` 여부부터 확인할 것.** 특히 두 개의 유사 평가 경로(val/test) 중 하나만 죽으면, 배치·데이터 차이보다 먼저 두 함수의 grad-context 데코레이션 차이를 대조하라.

---

### ISSUE-031: A100 이전 시 배치 재프로파일 누락 — 상시규칙 미적용 [프로세스 결함, 재발방지 적용, 2026-08-04]

**상시규칙**(메모리 `batch-sizing-policy`): "배치는 GPU 85~90% 채우게, eff-batch 16은 accum으로 유지(LR 불변), 서버 이전 시 재프로파일 필수"

**위반 사례**: `configs/hpca100-muses_rgbelr_P47_d1_dgfproj_4modal.yaml`의 `BATCH_SIZE: 1`. 주석은 *"seed2와 동일(1-변수 비교 유지)"*로만 적혀 있고 **A100(40GB) 기준 프로파일 흔적이 없다.** 3090/4090(24GB)용 값을 그대로 가져왔다.

**실측(2026-08-04, 2GPU 재기동 후 정상 구간)**: **rank당 24.6GB / 40GB = 60%** — 정책 목표 85~90%에 크게 미달.
- GPU2 30,558MiB(우리 24,618 + 타 테넌트 5,922) / GPU3 24,683MiB(우리 24,670)
- ⚠️ 기동검증 당시 registry에 기록된 "34.1~34.3GiB/rank"는 PyTorch caching allocator의 **기회적 예약치**였고, 타 테넌트가 있으면 24.6GB로 줄어든다. **`nvidia-smi memory.used`는 예약량이지 실제 소요량이 아니다** — 프로파일에는 `torch.cuda.max_memory_allocated()`를 써야 한다.

**이 런은 변경하지 않기로 판정(2026-08-04)**: ①BS2는 GPU2의 가용 34GB(타 테넌트 5.9GB 점유)를 초과할 위험 ②ep181/300 진행 중인 **계보 최고 기록 런**(82.58@ep172)이라 재기동 위험 ③base가 BS1이라 1-변수 순수성 훼손

**재발방지**: 이후 **A100 신규 기동 전 메모리 프로파일을 필수 단계로** 둔다. 짧은 스모크로 `torch.cuda.max_memory_allocated()` 측정 → 85% 채우는 BS 채택 → eff-batch 16은 accum으로 유지(LR 불변). 적용 대상: P47-2 4모달 · MUSES RGB-L 2모달 · DELIVER RGB-D 2모달.

---

### ISSUE-030: `last_checkpoint.pth` 비원자적 저장 — 저장 도중 사망 시 재개 불가 [수정, 2026-08-06]

**위치**: `train_reliadino.py:485` — `torch.save(_ckpt(), save_dir/'last_checkpoint.pth')`가 **최종 경로에 직접 덮어쓴다**(임시파일+rename 없음).

**증상**: 매 epoch 같은 파일을 in-place로 다시 쓰므로, 그 사이에 프로세스가 죽으면(preempt·OOM·노드 장애·SIGKILL) **파일이 잘린 상태로 남아 AUTO_RESUME이 실패**한다.

**발견 경위(2026-08-04)**: P47-1을 4GPU→2GPU로 옮기며 `cp`로 백업했는데 **md5가 5분 내 3회 바뀌어** 백업본 무결성을 보장할 수 없었다. 원인은 학습이 백업보다 빠르게 같은 파일을 덮어쓴 것.

🔴 **hpca100은 공유 pod이라 preempt로 잡이 죽는 전례가 있는 서버**다. 실제 위험.

**완화(현재)**: `epochNN_<miou>_topK_checkpoint.pth`(epoch-태그)는 한 번 쓰고 다시 안 건드리는 **안정 파일**이라 최후 수단으로 쓸 수 있다. 이번에도 `epoch176_82.3_top5`가 정상 로드됨을 확인했다(epoch=176, model 722 entries, optimizer/scheduler/scaler 전부 존재).

**수정안**: `torch.save`를 `<path>.tmp`에 쓴 뒤 `os.replace(tmp, path)`로 원자적 교체. 같은 파일시스템이므로 rename은 원자적이다.

**상태**: ✅ 수정 완료(2026-08-06, 커밋 `0bc65f5`). `_atomic_save(obj, path)` 헬퍼 추가 — `<path>.tmp`에 `torch.save` 후 `os.replace(tmp, path)`. `last_checkpoint.pth`와 topK best-checkpoint 저장 양쪽에 적용. 스모크 3건(실제 torch, jarvis MMSS_SAM env): fresh save+load / 동일 파일명 반복 덮어쓰기 / tmp 파일이 중간에 남아도 기존 target 무결 — 전부 통과. 진행 중이던 학습런은 코드 미변경(다음 재기동부터 적용).

---

### ISSUE-028: P46-CTR "지연성 OOM"의 정체 = warmup 계단 (누수 아님) [수정, 2026-07-29]

**증상 보고**: `configs/jarvis-deliver_rgbdel_P46_ctr.yaml`(all-on C1+C2+C3, BS1) jarvis 4090×4.
ep1~5 정상(rank당 ~15.2GB, ep4 val 59.66 확보) → **ep6 iter0에서 4-rank 전부 OOM**
(23.47GiB 사용 중 20MiB 실패). "에폭이 갈수록 메모리가 서서히 증가하는 누수"로 접수됨.

**실제 원인 = 누수가 아니라 계단.** 근거 3개가 모두 한 지점을 가리킨다:

1. **타이밍이 정확히 warmup 경계다.** config `C2_MCC.WARMUP_EP: 5`, `C3_PROTO.WARMUP_EP: 5`.
   `train_reliadino.py`의 epoch은 0-index(`for epoch in range(start_epoch, epochs)`)이고
   로그는 `epoch+1`로 찍는다(`Epoch [{epoch+1}/{epochs}]`). 따라서 **로그상 ep6 = epoch 5**가
   `epoch >= 5`를 처음 만족하는 epoch이다. 그 iter0에서 **동시에 셋이 처음 켜진다**:
     - 보조 student branch (`model(_bx, True, gt_mask=lbl)`) — **주 forward 그래프와 동시에 살아 있는** 두 번째 full forward 그래프
     - EMA teacher forward (4×ViT-L 한 벌, no_grad지만 작업메모리는 실재)
     - 주 forward의 prototype 손실 (`model.py`: `self._current_epoch >= self.p46_proto_warmup_ep`)
2. **ep1~5는 P46 비용을 하나도 안 쓴다.** epoch 0~4에서는 위 세 게이트가 전부 False →
   그 구간의 15.2GB는 **P39.1-base의 수치**이고, P46의 실제 학습 비용은 이 런에서
   **한 번도 측정된 적이 없다**. "ep1~5 대비 증가"라는 비교 자체가 성립하지 않는다.
3. **iter0이라는 점.** 누수라면 에폭 중간에 터진다. 정확히 warmup epoch의 **첫 iteration**에서
   4-rank가 동시에 죽는 것은 스텝 누적이 아니라 구조적 peak 증가의 서명이다.
   (EVAL_INTERVAL:2라 eval은 ep4에서 끝났고 ep5는 학습만 했다 → eval 직후 파편화도 아니다.)

**계측으로 확인한 것 (CPU tiny 모델, gc live-tensor bytes)**: 수정 전 86.3MiB → 수정 후 69.5MiB로
**둘 다 스텝에 대해 완전히 평평**(단조증가 0%). 즉 P46에는 **단조 누수가 존재하지 않고**,
아래 4건은 전부 **상수 오버헤드**였다.

**부수 수정 4건 (상수 오버헤드 — 근본원인 아님, 그래도 실재)**
- `train_reliadino.py` 보조 branch `_baux`: total에 들어가는 건 `p46_proto` 하나뿐인데
  나머지(`m2f_loss`/`vicreg`/`aux_ce`/`router_*`)는 **backward가 도달하지 않는** 서브그래프라
  backward가 saved tensor를 해제하지 않는다 → `_bproto`만 꺼내고 즉시 `del _baux`.
- 루프 지역변수(`logits`/`m_feat`/`aux`/`total`/`loss`/`_blogits`/`_tlogits`)는 **다음
  iteration이 재대입할 때까지** 살아 있다 → 직전 스텝 잔재가 이번 스텝의 두 forward가
  peak를 찍는 내내 상주. iteration 끝에서 명시 `del`. (`_blogits`/`_tlogits` 각 56.2MiB @768²)
- `p46.EMATeacher.__call__`: teacher는 eval 경로라 `_last_per_modal_feats`/`_last_fused_*`/
  `_last_p43_out` 등 분석 탭(~41MiB)을 매 스텝 캐시하는데 **아무도 읽지 않는다** → `_clear_diag()`.
  (student `_core`의 탭은 val_*/tools/viz_*가 읽으므로 **건드리지 않는다**.)
- `p46.PrototypeBank._sample`: `feat.float()` → `permute().reshape()` → `f[keep]` 순서라
  (B,256,192,192) **fp32 전체 사본 3장(108MiB)**을 그래프에 올린 뒤 4096행만 썼다 →
  인덱스 먼저·캐스팅 나중(gather)으로 4.0MiB. 호출 2회 기준 **-208MiB**. 뽑는 행·난수 소비·
  수치 전부 동일(smoke H가 `max|diff|=0`으로 검증).
- eval 직후 `torch.cuda.empty_cache()` 추가(eval **전**에 있던 것의 짝 — 파편화 완화).

**🔴 남아 있는 것 = 근본 비용.** peak에 **student 그래프 2개가 동시에 산다**. 이건 설계다 —
`total = total + p46_cons + p46_xv` 후 backward 1회라는 구조가 DDP 계약에서 나온다:
`find_unused_parameters=True`는 **마지막 forward**로 unused 집합을 정하므로 두 forward의
파라미터 사용 집합이 같아야 하고(그래서 보조 branch에도 `gt_mask`를 넘겨 M2F/aux까지 전부
계산한다), backward를 둘로 쪼개면 보조 그래프에서만 grad를 받는 파라미터가 생겨 **reducer가
정지**한다(2026-07-16 NCCL 데드락과 같은 부류). **그래서 backward 분리는 하지 않았다.**
- `GRADIENT_CHECKPOINT: true`는 **쓸 수 없다** — ISSUE-027(멀티모달 강제 off 가드).
- 즉 24GB 4090에서 BS1 all-on은 여전히 빠듯하다. 실행 전 `P46_MEM_LOG=1`로 ep5→ep6 계단을
  먼저 실측할 것. 슬롯을 못 맞추면 (a) 더 큰 카드, (b) `IMAGE_SIZE` 축소,
  (c) C2/C3 중 하나만 켜기(보조 branch는 공유라 forward 수는 같지만 손실 그래프가 준다) 중 택1.

**회귀 방지**: `tools/smoke_p46.py`
- **I-a 스텝 간 참조 해제**(결정적, tolerance 없음): step0의 `_baux[*]`/`_blogits`/`_tlogits`에
  weakref를 걸고 **step1의 peak 지점**에서 생존 여부 판정. ⚠️ **측정 시점이 핵심** — 스텝이
  끝난 뒤 재면 수정 전에도 다 죽어 있어 아무것도 안 잡힌다(실측 확인). 수정 전 5/5 생존,
  수정 후 0/5로 검출력 확인함.
- **I-b 메모리 단조성**: N스텝 `cuda.memory_allocated`(CPU면 gc live-tensor bytes) 추이가
  step2 대비 +5% 이내. 진짜 누수용이며 **상수 오버헤드는 못 잡는다**(그래서 I-a가 따로 있다).
- **G** teacher `_last_*` 해제 + student 탭 보존, **H** PrototypeBank gather 등가성.

**교훈**: warmup이 걸린 모듈은 **warmup 이후 epoch을 최소 1회 통과**해야 자원 검증이 끝난 것이다.
`WARMUP_EP: 5`인 config를 스모크/초반 에폭만 보고 "OOM-safe"로 기록한 것이 이번 사고의 실체다
(config 주석의 "BS1, OOM-safe per smoke test"가 그 오기록). 자원 스모크는 **WARMUP_EP를 0으로
낮춰** 돌리거나, 최소한 `P46_MEM_LOG`로 계단 지점을 확인하라.

---

### ISSUE-029: hpca100 HF 백본 이중고장(offline=RANDOM INIT / online=hang) [해결, 2026-07-28]

**발견 경위**: hpca100에서 4-modal(+radar) 학습 2건(P44-BMR+radar, P39.1+radar seed2)이 연속으로 ep2 mIoU가 3모달·yeon 동일 레시피(~45~48) 대비 극단적으로 저조(11~22)하게 나와 원인 조사 중 발견. 최초 가설은 "P44-BMR이 radar에 과균형해 실패"였으나, **P44도 P39.1도 동일하게 저조**했던 점에서 재검토 → radar 데이터(md5 검증 완료, 정상) 무죄, 백본 로드 자체가 원인으로 확진.

**메커니즘**: 학습 시 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 걸었더니 timm이 DINOv3(`vit_large_patch16_dinov3.lvd1689m`)와 폴백 DINOv2(`vit_large_patch14_reg4_dinov2`) 둘 다 `LocalEntryNotFoundError`로 로드 실패 → encoder.py의 `_create`가 **경고만 남기고 RANDOM INIT으로 조용히 진행**("Do NOT train a real run like this" 경고는 있으나 학습은 계속 진행됨). 반대로 offline 플래그를 빼고 온라인으로 돌리면 HTTP HEAD 요청(302 Found)까지는 성공하나 이후 12분 이상 정체(hang)해 학습이 진행되지 않음. 캐시 자체(blob 1,212,347,640 bytes, md5 hpca100=yeon 완전 일치)는 정상이었고 config.json 부재도 원인이 아니었음(yeon도 동일 캐시 구조지만 always-online이라 문제가 드러나지 않았을 뿐) — 실제로는 timm의 HF offline 조회 로직이 `pytorch_model.bin`을 우선 HEAD 조회하다 오프라인에 막혀 로컬 `model.safetensors`로 폴백을 못 하는 것으로 관찰됨.

**수정**: `semseg/models/reliadino/encoder.py`의 `_create`에 `RELIADINO_LOCAL_BACKBONE` env가 설정되면 `timm.create_model(..., pretrained_cfg_overlay=dict(file=<local safetensors path>))`로 HF Hub 조회를 완전히 우회하고 로컬 파일에서 직접 로드하도록 수정(develop 697a10a). 격리 테스트(`LOAD_OK 303079424`)로 검증 후 배포. hpca100 P39.1+radar seed2 재기동 결과 RANDOM INIT 없이 ep2 mIoU 47.61로 정상 궤도 복귀 확인.

**영향 범위**: hpca100에서 `HF_HUB_OFFLINE=1`로 돌았던 4-modal 런 2건(P44-BMR+radar, P39.1+radar seed2 최초 시도)이 RANDOM INIT 상태로 학습됨 — 두 런의 ckpt/수치는 전부 무효(성능 인용 금지, 오염 ckpt는 `_contaminated_randominit/`로 격리). 3모달 hpca100 학습(P44-BMR MUSES 등)은 항상 온라인으로 돌아 무영향.

**재발 방지**: 저조(mIoU 급락, 특히 다수 클래스 IoU=0 패턴)가 나오면 **가장 먼저 백본 로드 라인**(`grep -iE 'dinov3|RANDOM INIT|falling back|Loading weights using safetensors'`)을 확인할 것. hpca100처럼 churn 노드에서 재기동 시 `RELIADINO_LOCAL_BACKBONE` env 누락하지 말 것 — AUTO_RESUME이 이전 오염 ckpt를 이어받지 않는지도 함께 확인.

---

### ISSUE-027: GRADIENT_CHECKPOINT=true 시 timm 재계산이 stale active_modality로 LoRA grad 오염 [가드 추가, 2026-07-21]

**발견 경위**: P37~현재 코드 전수조사(멀티에이전트 32기, 발견→반증검증)에서 발견. 팀이 bengio 학습에서 이미 실증적으로 마주쳐 jarvis/hpca100 configs에 "절대 true 금지" 주석을 달아뒀으나, 코드 자체에는 방지 가드가 없었고 체크인된 configs 9종에 `GRADIENT_CHECKPOINT: true`가 잔존해 있었음.

**메커니즘**: `GRADIENT_CHECKPOINT=true`일 때 timm의 non-reentrant checkpoint 재계산이 backward 시점에 forward를 다시 실행하는데, 이 재계산 시점의 `active_modality`(LoRA가 어느 모달 브랜치를 활성화할지 결정하는 전역/버퍼 상태)가 **forward 당시 값이 아니라 backward 시점에 마지막으로 설정된 값(= 마지막 모달)으로 stale**되어 있음. 그 결과 비최종 모달(예: img, lidar — thermal이 마지막이라면)의 backward 재계산이 실제로는 **마지막 모달의 LoRA 가중치로 재실행**되어, 비최종 모달의 gradient가 잘못된 LoRA 파라미터 경로로 흘러 들어감 — **에러나 경고 없이 조용히 오염**.

**수정**: encoder 안에 `GRADIENT_CHECKPOINT`와 멀티모달 LoRA 활성 조합을 감지해 **강제 off 가드**를 추가하고, 체크인된 configs 9종을 전부 `GRADIENT_CHECKPOINT: false`로 정정.

**영향 범위**: 실피해는 한정적으로 판단됨 — 실제 본학습(hpca100/jarvis)은 이미 `false`였고, 이 값이 `true`인 채 돌았던 것은 bengio 사망런(어차피 노드 HW 고장으로 조기 종료)·yeon 스모크(참고용, 헤드라인 미사용)·hinton P34 config(실사용 여부 미상) 정도. 팀이 이미 주석으로 "금지" 표시해뒀던 덕에 실질 본학습 경로는 회피됨.

**재발 방지**: config 주석만으로 위험 파라미터 조합을 막지 말 것 — 코드 레벨 가드(assert/강제 override)가 원칙.

---

### ISSUE-026: ColorAugSSD brightness가 uint8 입력을 [0,1]로 클램프 → 발화 샘플 RGB가 백색 상수로 붕괴 [수정됨, 2026-07-21]

**발견 경위**: P37~현재 코드 전수조사(멀티에이전트 32기, 발견→반증검증)에서 발견.

**메커니즘**: `ColorAugSSD`(07-16 커밋)의 brightness 조정 로직이 입력을 [0,1] 정규화 float로 가정하고 클램프를 적용하는데, 실제 입력은 **uint8 0-255 스케일**로 들어옴. 이 스케일 불일치 때문에 brightness 증강이 발화(activate)되는 샘플(적용 확률 p=0.5)에서 RGB 값이 사실상 전부 클램프 상한(백색)으로 포화 — **결과적으로 RGB 채널이 의도치 않은 상수(백색)로 붕괴**하며, 이는 확률 0.5로 RGB 정보를 완전히 지워버리는 **사실상의 RGB-dropout 0.5**로 작동한 것과 동일한 효과를 냄.

**영향 범위**: ColorAugSSD 커밋(07-16) 이후 `DGFUSION_AUG: true`가 켜진 **DELIVER 학습 전부**에 해당.

| 실험 | 영향 |
|---|---|
| jarvis P37a-DELIVER | 오염 |
| bengio P37a/b (사망런) | 오염(단 조기 사망이라 영향 미미) |
| **hpca100 P38-DELIVER 200ep 완주분** | 오염 — **P38 게이트 미달 판정에 사용된 그 런** |
| **hpca100 P39-DPC resume (진행 중)** | **오염 상태로 현재도 학습 중** |
| yeon 스모크 | 오염(참고용, 헤드라인 미사용) |
| **MUSES 전 계보** | **무영향** — `DGFUSION_AUG` 키 자체가 MUSES config에 없음 |

⚠️ **재해석**: P36 fair 게이트(val 67.74/test 55.62)는 07-16 이전 학습이라 정상 RGB로 진행됨 — 즉 **P37+/P38/P39 DELIVER와 P36의 비교는 불공정 비교였음**. 이에 따라 **P38-DELIVER "게이트 미달 −1.63" 판정과 P39-DELIVER "−1.63 thin-class 퇴행" 판정 모두 보류**(RGB 파괴가 교란변수로 개입) — 픽스 후 재검증 전까지 확정 판정으로 인용 금지.

**수정**: brightness 조정 로직에 uint8→float [0,1] 정규화를 정합시켜 클램프 스케일을 맞춤. **P39.1부터는 픽스 적용 클린 학습**(develop 반영, 대기열 [experiments/plan.md](../experiments/plan.md) #1).

**후속**: hpca100 P39-DPC resume(진행 중)은 오염 상태로 계속 학습 중 — 지속/중단은 user 판단 필요([experiments/plan.md](../experiments/plan.md) 실행 중 표 참조). 픽스 후 재검증이 필요한 항목은 P38-DELIVER/P39-DELIVER 게이트 재판정.

---

### ISSUE-025: MUSES radar 디코딩 3중 버그 — RADAR_RANGE_MAX 미정의로 100m 클립 + height 채널 오염 [해결, 2026-07-21]

**발견 경위**: jarvis에서 진행 중인 P39 4모달(rgbelr) radar 기여 재검토를 위해 radar 디코더 경로를 실측 검증하던 중 발견. jarvis radar 75파일 실측(2026-07-21)으로 확정.

**메커니즘 (3중)**:
1. `_open_radar`가 자체 구현 없이 `_open_lidar`로 **폴스루** — radar 전용 처리(range 스케일 등)가 애초에 존재하지 않았음.
2. 데이터셋 `__getitem__` 디스패치가 radar 모달을 `_open_radar`가 아니라 `_open_lidar`로 **직접 라우팅** — ①의 폴스루 여부와 무관하게 `_open_radar` 자체가 죽은 코드였음.
3. 결과적으로 radar range가 (lidar 기준) `LIDAR_RANGE_MAX=100m`에 **클립**됨 — 실측 결과 유효 픽셀의 **2.76%가 이 클립에서 포화**. radar 센서 실제 캡은 **정확히 150.0m**로 확인. 추가로 lidar 파이프라인의 height 채널(radar는 전 픽셀 0)이 정규화 후 **0.25 상수 평면**으로 오염되어 3번째 채널이 정보 없이 채워짐.

**수정**: `RADAR_RANGE_MAX=150.0`으로 radar 전용 클립 상수 도입, ch3(height)를 radar에서는 **occupancy 마스크**로 대체, `__getitem__` 디스패치에서 radar를 `_open_radar`로 정상 라우팅. develop에 반영 완료(merge 80d65a0 계보).

**참고**: lecun 세션에서도 동일 시기 미검증 픽스(브랜치 `lecun-wip-20260721`)가 있었으나 방향만 같고 별개로 독립 실측 검증 후 재작성함 — 두 세션이 동일 버그를 독립 발견.

**영향 범위**:
| 실험 | 영향 |
|---|---|
| P34 4모달(rgbelr) MUSES test 78.256 | **오염** — "radar 유해 −0.72" 판정은 broken decoder 상태에서 나온 결론이라 **보류**. 픽스 후 재측정 필요 |
| diag_D zeroradar 계열 | 오염(radar 포함 조건) |
| P39 4모달(jarvis 진행 중, rgbelr) | 오염 — 진행 중 "+0.86" 등 수치는 **broken-radar 하한**으로 취급, 픽스 후 상향 여지 있음. 완주는 그대로 시켜 broken-radar 기준선으로 보존 |
| **3모달 전 계보** (P34-3모달 78.979 / P37a / P38-m2f 79.025 / P39-3모달 78.881 등) | **무영향** — radar 미사용이므로 오염 없음. lidar/event/camera 디코더는 원래 정상이었음 |

**후속**: 픽스 적용 4모달 재실험을 대기열 후보로 등록(`experiments/plan.md`) — ISSUE-025 픽스 후 radar 기여를 재측정해야 P34의 "radar −0.72" 판정을 확정/철회할 수 있음.

---

### ISSUE-024: P37b `classtoken.py`의 `mask_proj`가 gradient를 받지 못함 (random attn mask) [OPEN, 조건부, 2026-07-17]

**발견 경위**: P38 MaskQueryLite(`m2f_head.py`) masked cross-attention 구현 중, P37b `classtoken.py`의 attn-mask 생성 경로를 재사용하려다 발견.

**메커니즘**: P37b `ClassToken-lite-Learned`의 `mask_proj`는 다음 layer의 cross-attn 마스크를 만들기 위한 예측기이지만, 실제로는 그 출력을 **threshold 비교**(`mask_proj(x) > 0` 류의 비미분 연산)로만 소비해 boolean mask를 만든다. threshold 비교는 미분 불가능하므로 `mask_proj`의 파라미터로 역전파되는 gradient가 전혀 없다 → **`mask_proj`가 영구 random init 상태로 남고, 결과적으로 masked attention이 사실상 random 마스킹**으로 동작한다. layer1이 unmasked(첫 레이어는 마스크 없이 attend)이고 NaN guard가 있어 학습이 발산하지는 않지만(치명적이지 않음), 의도한 "이전 예측 기반 점진적 마스크 정제"라는 Mask2Former 관행이 실질적으로 작동하지 않는다.

**P38 수정 패턴 (참고용, 이미 적용됨)**: `semseg/models/reliadino/m2f_head.py`의 attn mask 생성은 **공유 cls/mask-embed head의 예측을 직접 stride4→16으로 리사이즈**해 사용(`_attn_bias`) — threshold-only 비미분 지름길을 거치지 않으므로 head 예측에 gradient가 정상적으로 흐른다. P38은 처음부터 이 방식으로 구현되어 동일 문제가 없다.

**영향 범위**: P37b 단독(kill-gate 결과 대기 중, bengio 생존 미확인). ClassToken 헤드 자체의 다른 경로(class-token cross-attn 본체)는 정상 학습되므로 P37b 결과가 완전히 무효는 아니나, masked attention이 설계 의도대로 기여하지 못하고 있어 P37b 수치를 "ClassToken 어블레이션"으로 해석할 때 이 결함을 감안해야 한다.

**조치 필요**: P37b가 kill-gate에서 생존해 후속 실험 대상이 되면, `m2f_head.py`의 `_attn_bias` 패턴(공유-head 예측 리사이즈)으로 `mask_proj` 경로를 동일하게 수정할 것.

---

### ISSUE-023: /mnt/HDD2 전체 쓰기 불능 (ENOSPC, ntfs-3g) [미해결, 2026-07-08]

**발견 경위**: 재구조화 세션에서 데드 실험 outputs(~170G)를 HDD2 아카이브로 이동 중, P10~P17(68G)까지 정상 이동 후 P18부터 rsync mkstemp가 전부 `No space left on device (28)` 실패. 이후 `touch`/`dd` 등 모든 신규 쓰기가 즉시 ENOSPC.

**증상/증거**: `df -h` = 17T 중 1.3T 사용(16T 여유), `df -i` = inode 1% 사용 — 그런데도 ENOSPC. 파일시스템은 `fuseblk`(ntfs-3g). 읽기는 정상.

**영향**: ① 공유 eval/분석 산출물 정규 위치 `/mnt/HDD2/src/logs/`에 새 결과 저장 불가 (모든 세션 영향) ② 데드 outputs 이동 잔여분 ~105G 보류 (`outputs/ARCHIVE_MANIFEST.md` 참조 — 원본은 그대로 보존됨).

**시도한 것**: 3회 재시도(즉시 실패 재현), lsof 확인 — nautilus + 타 프로젝트(tactile) Claude 세션들이 HDD2 사용 중이라 **재마운트는 하지 않음**(타 세션 파괴).

**조치 필요(사용자)**: HDD2 사용 세션 정리 후 `sudo umount /mnt/HDD2 && sudo mount ...` 재마운트, 그래도 재발 시 Windows에서 chkdsk (ntfs-3g $Bitmap 불일치 의심). 복구 후 이동 재개: `bash /home/jemo/.claude/jobs/ac8fdb6e/tmp/move_dead_outputs.sh` (이동완료분 자동 SKIP).

**2026-07-08 오후 갱신 — 원인 확정 + 완화**:
- **근본 원인 = NTFS MFT 레코드 고갈**: 판별 실험으로 확정 — 기존 파일 append/truncate 정상, 새 파일 생성만 ENOSPC, **파일 1개 삭제 → 정확히 1개 생성 가능(1:1)**, 대량 생성 즉시 실패. `ntfsinfo`(-f, docker root nsenter, 읽기전용): 클러스터 92.3% 여유·MFT zone 여유 → 용량 문제 아님, MFT 확장 실패(장기 마운트된 ntfs-3g 할당자 상태 또는 ntfs-3g 한계).
- 오늘 오전 아카이브 push(P10~P17, 19k 파일)가 잔여 레코드를 소진시킨 트리거로 추정.
- `mount -o remount`는 ntfs-3g 미지원("umount 후 재마운트 필요"). 완전 재마운트는 활성 홀더(nautilus + tactile 프로젝트 Claude 세션들) 때문에 보류 — lazy umount는 이중 데몬 위험이라 금지.
- **완화(진행)**: HDD2의 임시 아카이브 68G/19k 파일을 drone NAS(`/drone_nas/home/jemo_archive/`)로 소산 → HDD2 MFT 레코드 ~19k 해방 = eval 로그 쓰기 정상화. 잔여 데드 105G도 HDD1→NAS 직행 (HDD2 경유 안 함).
- **잔여 리스크**: 해방된 레코드를 다 쓰면 재발. 근본 해결 = ① 전 홀더 종료 후 재마운트(ntfs-3g가 fresh mount에서 MFT 확장 성공하는지 확인) ② 안 되면 Windows에 연결해 파일 생성(Windows 드라이버는 MFT 확장 가능) 또는 백업 후 재포맷. HDD2에 대량 파일 쓰기(수천 개 단위)는 재마운트 검증 전까지 금지.
- 참고: 진단 과정에서 `.Trash-1000/files/SMPLX_FEMALE.npz`(사용자 휴지통) 1건을 판별용으로 삭제함.

---

### ISSUE-022: P27.forward가 `_fuse_outputs` 훅을 호출하지 않음 → P30 learned router 미실행 [해결, 2026-07-03]

**발견 경위**: P31 첫 학습(B200)에서 `[P31] router w̄` 로그 라인이 출력되지 않아 추적 → `p31_routerw_rows`가 빈 상태 = `ReliabilityAnchoredRouter.forward` 미호출.

**원인**: `_fuse_outputs` 훅(P30 router의 진입점)은 **P26.forward에만** 호출부가 있고, P27이 forward 전체를 오버라이드하면서 융합을 **인라인**(`m_output = Σ amf_norm·output; m_feat = Σ q_uamm·feat`)으로 다시 구현함. P28/P29/P30/P31은 P27.forward를 상속 → **P30의 `_fuse_outputs` override(router)는 도달 불가능 코드**였음. DDP `find_unused_parameters=True`가 죽은 router 파라미터를 은폐(에러 0).

**영향**:
- **P30 B200 200ep 결과(Val 49.76/Test 44.10)는 "CTD+SDC only, router 미참여"로 재해석**해야 함 — 붕괴 범인 후보가 class-token decoder(+SDC)로 좁혀짐. router는 무죄(참여 자체 없음).
- doc 16 §2의 P30 Mode C/D "메커니즘 ✅" 판정은 코드 정적 분석 기준 — **런타임 도달성 검증 누락**이 교훈.

**수정 (P31.2)**: P27.forward Phase 4의 인라인 융합 2줄을 `self._fuse_outputs(...)` 호출로 교체. P26 기본 훅 본문이 인라인 수식과 byte-identical → P27/P28/P29 행동 불변, P30/P31 router가 비로소 활성화.

**재발 방지**: 훅 패턴 도입 시 **모든 서브클래스 forward의 호출부 존재를 grep으로 확인**할 것 (`grep -n "_fuse_outputs("` → 정의 수 vs 호출 수).

---

## 열린 이슈 (Open Issues)

### ISSUE-001: Val에 NIGHT_AUG 미적용 → 모델 선택 기준 부적합 [심각]

**상태**: ✅ 해결됨 (2026-02-25)
**영향**: 모든 P 버전 (P8~P13)

**문제**:
- `get_val_augmentation()` (`semseg/augmentations_mm.py` line 593)에 NIGHT_AUG 없음
- Val = 주간 이미지 그대로 평가 → val mIoU = 주간 성능만 반영
- Test = 야간 이미지 → val best checkpoint ≠ test best checkpoint
- 실제로 모든 모델의 val mIoU가 93~94%로 거의 동일하여 모델 구분 불가
- 하지만 test mIoU는 35~70%로 편차 매우 큼

**해결**:
- `get_nightval_augmentation()` 함수 신규 추가 (`semseg/augmentations_mm.py` line 608~)
  - 기존 `get_val_augmentation()`은 수정하지 않음 (호환성 유지)
  - NightSim p=1.0 (항상 적용, dice-roll 제거)
  - CRM / Zero-out은 config 확률 그대로 (더 realistic)
  - 기하학적 증강(Flip, Crop) 없음
- `train_sam2_lora_paper.py` 변경:
  - `get_nightval_augmentation` import 추가
  - `nightvalset` / `night_valloader` 생성 (NIGHT_AUG.ENABLE 시에만)
  - `best_night_mIoU` / `best_night_epoch` 상태 변수 추가
  - Night-Val 평가 블록 추가 (`val_night/mIoU` TensorBoard 로깅)
  - `night_epoch{N}_{mIoU}_checkpoint.pth` 별도 저장
  - resume checkpoint에서 `best_night_miou` / `best_night_epoch` 복원
  - 최종 summary 테이블에 "Best Night-Val mIoU" 행 추가

**체크포인트 구분**:
- `epoch{N}_{mIoU}_checkpoint.pth`       → Day-Val best (주간 성능)
- `night_epoch{N}_{mIoU}_checkpoint.pth` → Night-Val best (야간 시뮬 성능)

**미해결 (후속 과제)**:
- hardaug4의 brightness 분포 vs 실제 test(lj4) brightness 분포 비교 → Night-Val이 test를 얼마나 잘 proxy하는지 검증 필요
- P13 학습 후 day-best vs night-best의 test mIoU 비교 필요

**관련 파일**:
- `semseg/augmentations_mm.py`: `get_nightval_augmentation()` (line 608~)
- `train_sam2_lora_paper.py`: night-val 평가 블록 (line 602~632)

---

### ISSUE-002: MoE Expert Collapse — Block 6-20에서 E1 사망 [중요]

**상태**: ❌ P13에서 수정 시도했으나 **해결 실패** (2026-02-26 검증 완료)
**영향**: P8, P9, P10, P11, P12, P13 (전 버전)

**문제**:
- `SoftMoE_LoRA_Layer.reset_parameters()` (`sam_lola_utils.py` line 562~575)
- `experts_b` zero-init → 모든 expert 출력 = 0 → gate gradient = 0
- Rich-get-richer 현상 → Block 6-20 (15개, stage 3)에서 E1 사용률 < 3%
- 3-expert MoE가 실질적으로 2-expert로 동작, 용량 1/3 낭비

**진단 데이터** (val_pred_P9/uamm_amf_moe_log.json):
```
Block9_Q argmax_fraction:
  img:     E0=43~51%, E1=0~10%, E2=40~54%
  lidar:   E0=16~17%, E1=0~0.5%, E2=83~84%
  thermal: E0=84%,    E1=0.3~0.9%, E2=15~16%
  → E1 거의 미사용
```

**P13 수정 시도**:
- `LoRA_Sam_P13.__init__`에서 experts_b를 `kaiming_uniform_ * 0.01`로 재초기화
- `sam_lola_utils.py`는 수정하지 않음 (P9 체크포인트 호환성 유지)

**P13 검증 결과 (2026-02-26)**:
- Collapse rate: P13 val 17.4% vs P12 val 16.0% → **개선 없음 (오히려 소폭 악화)**
- LiDAR collapse: ~27% (P12와 동일)
- Q blocks: 23-25% collapse, V blocks: 10-11% (P12와 동일 패턴)
- Stage별: S1(44-55%) > S4(30%) > S2(20%) > S3(9-13%)
- 실패 원인:
  1. Resume 학습 → 이전 gate weights가 로드되면서 init 효과 무력화
  2. kaiming * 0.01 (~0.005 수준)은 zero-init과 실질적 차이 미미
  3. 근본 원인이 init이 아니라 soft-MoE softmax의 winner-take-all 특성

**미해결**: 근본적인 해결책 필요 (load balancing loss, top-k routing, expert dropout 등)

---

### ISSUE-003: CrossModalFusionHead 상수 출력 [중요]

**상태**: ⚠️ P13에서 수정됨 — **부분 성공** (2026-02-26 검증 완료)
**영향**: P9 (P10/P11은 HeadV2 사용하지만 유사 문제)

**문제**:
- `CrossModalFusionHead` (P9)의 UAMM/AMF 출력이 모든 이미지에서 동일:
  - UAMM: img=0.745, lidar=0.961, thermal=1.0
  - AMF: img=0.275, lidar=0.355, thermal=0.370
- 원인: GAP(65536 토큰 평균) + LayerNorm 정규화 → 입력 무관하게 같은 벡터
- 결과: adaptive fusion이 아닌 fixed fusion. 밤에 RGB가 어두워도 27.5% 가중치

**P13 수정 방법**:
- CrossModalFusionHead 제거 → ConfidenceAuxHead + Energy Score로 교체
- Energy Score는 aux head의 raw logit에서 계산 (학습 파라미터 없음)
- 학습/추론 동일 메커니즘 (P10의 train≠test 문제 없음)

**P13 검증 결과 (2026-02-26)**:
- UAMM CV (이미지별 변동성): img val 0.112 (P12: 0.005, **22x 증가**) → **상수 수렴 문제 해결**
- Test에서도 img CV=0.073 (P12: 0.014, 5x 증가)
- **단, test LiDAR UAMM = 1.0 고정 (CV=0.000)** — LiDAR aux head가 항상 가장 높은 energy 출력
- Dynamic IoU +5.55pp 개선 → energy fusion이 모달리티 가중치를 유의미하게 변경
- Val mIoU -0.87pp → adaptive weight의 정확도가 P9의 안정적 상수 비율보다 val에서 불리

**결론**: 상수 출력 문제 자체는 해결됨. 하지만 adaptive weight의 **정확도**가 새로운 병목.

---

### ISSUE-006: Aux Head Mask 시각화 — Energy Score 검증 [구현 필요]

**상태**: 🔲 미구현
**우선순위**: 높음 — P13 Energy Score fusion의 실제 동작 검증 및 P14 방향 설정에 필수
**영향**: P13 평가/분석

**목적**:
- Aux head가 모달리티별로 무엇을 보고 있는지 시각적 확인
- Energy Score가 "잘못된 confidence"를 주는 케이스 식별 (특히 test LiDAR UAMM=1.0 문제)
- Thermal aux mask가 야간에 RGB보다 Dynamic/Sky를 더 잘 잡는 프레임 확인
- P14 설계에 필요한 실증 데이터 수집

**구현 사양**:

1. **저장 대상** (각 이미지에 대해):
   - RGB / LiDAR / Thermal 입력 이미지 (3장)
   - Aux head prediction mask — 모달리티별 argmax 컬러맵 (3장)
     - `aux_logits_img`, `aux_logits_lidar`, `aux_logits_thermal` 각각에 argmax
     - 컬러맵: Static=빨강, Dynamic=초록, Water=파랑, Sky=노랑, ignore=회색
   - Final prediction mask (1장)
   - Ground truth (val에서만, test는 없음)
   - Energy confidence weights: img/lidar/thermal 각 스칼라 값 (이미지 파일명에 포함 또는 별도 JSON)
   - UAMM/AMF weights: 이미지당 값

2. **출력 형식**:
   - 한 이미지당 1개의 panorama 이미지 (가로 배치):
     ```
     [RGB] [LiDAR] [Thermal] | [Aux_RGB] [Aux_LiDAR] [Aux_Thermal] | [Pred] [GT]
     ```
   - 각 aux mask 위에 energy confidence 값 표시 (e.g., "img: 0.28, E=3.42")
   - 파일명: `{image_id}_auxmask.png`
   - 별도 JSON: `{image_id}_energy.json` (수치 데이터)

3. **실행 위치**:
   - `val_multiaqua.py`에 추가 (`val_multiaqua_P9.py`는 삭제된 상태)
   - `--save-auxmask` 플래그로 활성화
   - 출력 디렉토리: `{save_dir}/auxmask/`

4. **코드 변경 필요 사항**:
   - `LoRA_Sam_P13.forward()`에서 `aux_logits`를 반환하도록 수정 (현재는 loss 계산에만 사용)
   - 또는 eval 모드에서 별도로 aux head forward를 호출
   - `compute_energy_confidence()`의 중간 값 (per-modality energy)도 저장

5. **분석 포인트** (시각화 결과 확인 시):
   - [ ] LiDAR aux mask가 Water/Dynamic을 못 잡는데 UAMM=1.0인 프레임 확인
   - [ ] Thermal aux mask가 Dynamic을 RGB보다 잘 잡는 프레임 비율
   - [ ] RGB aux mask가 야간에 완전히 깨지는 프레임에서 Energy Score가 RGB weight를 낮추는지
   - [ ] Sky를 잘못 예측하는 모달리티가 어느 것인지 (Sky -1.42pp 하락 원인)

**관련 파일**:
- `val_multiaqua.py`: 평가 스크립트 (시각화 추가 대상)
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: LoRA_Sam_P13, ConfidenceAuxHead
- `val_pred_P13/`: 기존 P13 평가 결과 디렉토리

---

### ISSUE-007: CRM/ZERO Overfitting — Night-Val↑ Test↓ 역전 현상 [부분 해결]

**상태**: 🟡 hardaug5에서 CRM/ZERO 제거 완료 (2026-02-27). 하지만 Sky collapse 여전 → 부분 원인.
**영향**: P13 (epoch39에서 발견), 잠재적으로 모든 P 버전
**우선순위**: 최고 — P13 epoch39에서 test mIoU -19.5pp 폭락 유발

**문제**:

CRM (`RandomRGBComplementaryMasking`, p=0.35)과 ZERO (`RandomRGBZeroOut`, p=0.09)가 RGB에 **exact zero** 값을 삽입하여 train-test 분포 불일치를 유발.

1. **Exact zero는 실제 센서 데이터에 없음**: 야간 RGB는 noise가 있는 near-zero (0.001~0.01), 절대 exact 0이 아님
2. **Normalize 후 고유한 feature vector 생성**: `(0-mean)/std = (-2.118, -2.036, -1.804)` — 자연 이미지에서 나타나지 않는 극단값
3. **Shortcut 학습**: "exact zero 감지 → RGB 무시, thermal/LiDAR 의존" — train/night-val에서는 유효, test에서는 무효
4. **Night-val 오염**: `get_nightval_augmentation()`에도 CRM/ZERO가 동일 확률로 적용 → night-val이 shortcut을 보상 → checkpoint 선택 오염

**정량적 증거** (P13):

| 지표 | Epoch17 | Epoch39 | Δ |
| --- | --- | --- | --- |
| Night-val | 87.71 | **89.53** (+1.82) | ✅ shortcut 학습 강화 |
| Test mIoU | 69.98 | **50.48** (-19.50) | ❌ shortcut 무효 |
| Test Sky | 75.12 | **23.36** (-51.76) | Sky가 가장 취약 |
| Test Sky=0 프레임 | 5/200 | **80/200** | 16배 증가 |

Sky가 가장 심각한 이유: 야간 하늘은 near-zero RGB → CRM/ZERO의 exact zero와 가장 유사 → shortcut이 가장 활성화되는 영역

**권장 조치**:

1. **Night-val에서 CRM/ZERO 제거** (`get_nightval_augmentation()`에서 CRM/ZERO 비활성화)
   - NightSim만 적용 → 실제 test 조건에 더 가까운 proxy
2. **학습 시 CRM/ZERO 확률 축소**: CRM_P 0.35→0.10, ZERO_P 0.09→0.03
3. **Exact zero → Noisy near-zero 대체**: `img[mask] = torch.randn_like(...) * 0.01`
4. **Early stopping 기준 개선**: night-val (CRM/ZERO 제거 버전)을 checkpoint 선택 기준으로 사용

**관련 코드**:

- `semseg/augmentations_mm.py`: `RandomRGBComplementaryMasking` (line 142), `RandomRGBZeroOut` (line 168), `get_nightval_augmentation` (line 609)
- Train config의 `NIGHT_AUG.CRM_P`, `NIGHT_AUG.ZERO_P`

---

### ISSUE-010: 로깅 시스템 전면 개선 — 모듈별 동작 모니터링 부재 [부분 해결]

**상태**: 🟡 Training script 부분 해결 (2026-02-27). Eval script 미해결.
**우선순위**: 중간 — Training 로깅은 trackio로 대폭 개선됨
**영향**: train_sam2_lora_paper.py, val_multiaqua.py, 전 P 버전

**문제 요약**:

모델이 매 forward에서 `_last_uamm_scores`, `_last_amf_weights`, `_last_moe_gates`, `_last_aux_logits` 등을 내부 버퍼에 저장하지만, **학습 스크립트가 이 버퍼를 한 번도 읽지 않음**. 평가 스크립트도 일부만 사용. 결과적으로 fusion, MoE routing, expert collapse, aux head 품질을 학습 중에 전혀 모니터링할 수 없음.

---

#### A. Training Script (train_sam2_lora_paper.py) 빈틈

**TensorBoard에 기록 안 되는 것들:**

| 누락 항목 | 심각도 | 현재 상태 | 추가할 TB key |
|-----------|--------|-----------|--------------|
| Gate loss | HIGH | tqdm에만 표시, 학습 후 소실 | `train/gate_loss` |
| MI loss | HIGH | tqdm에만 표시, 학습 후 소실 | `train/mi_loss` |
| UAMM per modality | HIGH | 학습 중 미수집 | `train/uamm_img`, `_lidar`, `_thermal` |
| AMF per modality | HIGH | 학습 중 미수집 | `train/amf_img`, `_lidar`, `_thermal` |
| Aux loss per modality | HIGH | 3 모달리티 합산 후 기록 | `train/aux_loss_img`, `_lidar`, `_thermal` |
| Per-class IoU (매 eval) | MEDIUM | new best일 때만 텍스트 | `val/iou_static`, `_dynamic`, `_water`, `_sky` |
| Night per-class IoU | MEDIUM | new best일 때만 텍스트 | `val_night/iou_static`, `_dynamic`, `_water`, `_sky` |
| MoE routing entropy | MEDIUM | 미수집 | `train/moe_entropy_mean` |
| Expert collapse count | MEDIUM | 미수집 | `train/expert_collapse_count` |

**이전 TensorBoard 기록은 6개 스칼라만**: `train/loss`, `train/proto_loss`, `train/aux_loss`, `train/lr`, `val/mIoU`, `val_night/mIoU`

**2026-02-27 개선 (trackio 전환)**:

- TensorBoard → trackio 전환 (TensorBoard fallback 유지)
- Training: total_loss, seg_loss, proto_loss, aux_loss, gate_loss, mi_loss, lr, warmup_ramp
- Day-Val: mIoU, pixel_acc, mean_f1, per-class IoU/acc/f1, best_mIoU
- Night-Val: 동일한 포괄적 메트릭 세트 (`val_night/` prefix)
- tqdm: 0값 loss 숨김, P16 warmup 상태 표시

**구현 방법**:
1. 매 eval 주기에 모델 버퍼에서 `_last_uamm_scores`, `_last_amf_weights` 읽어 TB에 기록
2. Aux loss 계산 루프에서 per-modality loss를 리스트로 따로 저장 후 개별 기록
3. Gate loss / MI loss를 epoch 평균으로 TB에 기록 (현재 tqdm 표시 코드 바로 옆에 추가)
4. `print_iou()`의 per-class 결과를 매 eval마다 TB에 기록 (new best 조건 제거)

---

#### B. Evaluation Script (val_multiaqua.py) 빈틈

| 누락 항목 | 심각도 | 현재 상태 | 추가할 형식 |
|-----------|--------|-----------|------------|
| Per-block MoE gate weights | HIGH | 24블록 mean으로 축소 → 블록별 정보 소실 | JSON dict (block별) |
| Expert utilization / entropy per block | HIGH | 미수집 | JSON |
| Energy Score raw values per modality | HIGH | softmax 후 weight만 저장, 원시 energy 폐기 | JSON per-image |
| Aux head predictions (ISSUE-006) | HIGH | `_last_aux_logits` 저장되지만 미사용 | PNG + JSON |
| Confusion matrix | MEDIUM | `metrics.hist` 계산되지만 미저장 | PNG heatmap |
| Per-image IoU | MEDIUM | aggregate만 | CSV |

**Per-block MoE gate 수정 방법**:
현재 코드가 `np.stack(moe_gate_collector, axis=0).mean(axis=0)`로 즉시 축소.
→ mean 대신 블록별 dict로 저장: `{"block0_Q": [e0, e1, e2], "block0_V": [e0, e1, e2], ...}`

**Energy Score raw values 수정 방법**:
`compute_energy_confidence()` 내부에서 중간값(per-modality raw energy, softmax 전 값)을 반환하도록 수정.
→ return `(weights, raw_energies)` 형태로 변경, `_last_energy_raw` 버퍼 추가.

---

#### C. 삭제된 스크립트

`val_multiaqua_P9.py`가 삭제된 상태. CLAUDE.md에서 참조 중이나 실제 파일 없음.
- per-block MoE routing 분석 기능이 사라짐
- **권장**: `val_multiaqua.py`에 해당 기능을 통합하거나, 별도 진단 스크립트로 분리

---

#### D. 구현 우선순위

1. **즉시 (ISSUE-006과 함께)**: Aux mask 시각화 + Energy raw values 저장
2. **단기**: Per-block MoE gate를 JSON에 블록별로 기록 (mean 축소 제거)
3. **단기**: Aux loss per-modality 분리 기록 (TB)
4. **단기**: Gate loss / MI loss TB 기록 (tqdm 옆에 1줄 추가)
5. **중기**: Per-class IoU 매 eval TB 기록, Confusion matrix 저장
6. **중기**: Expert collapse 자동 감지 + 경고 (training 중)

**관련 파일**:
- `train_sam2_lora_paper.py`: Training TB logging 추가 대상
- `val_multiaqua.py`: Eval logging 확장 대상
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: 모델 버퍼 접근 (`_last_*`)
- `semseg/models/sam2/sam2/sam_lola_utils.py`: `SoftMoE_LoRA_Layer._gate_callback`

---

### ISSUE-012: P23 (MoE DeBA-BB) A100 80GB OOM — Deformable Conv Activation 메모리 [하드웨어]

**상태**: 🔴 미해결 (2026-03-10)
**영향**: P23 (MoE_DeBA_BB), 잠재적으로 conv 기반 adapter 전체
**우선순위**: 높음 — P23 학습 자체가 불가능

**문제**:

P23 (MoE DeBA-BB)이 A100 80GB에서 batch=1로도 OOM 발생.
- 초기: 4 expert, scale [0.5, 1, 2, 4] → OOM
- 축소: 2 expert, scale [1, 2] → **여전히 OOM**
- P9 (SoftMoE-LoRA, Linear 기반)는 동일 환경에서 정상 학습

**원인 분석**:

1. **Conv activation이 Linear보다 훨씬 큰 메모리 차지**:
   - LoRA: `Linear(C→r)` → activation shape `(B, H, W, r)`, r=4로 매우 작음
   - DeBA-BB: `DCM 3×3 conv` → activation shape `(B, bottleneck_dim, H, W)` + deformable offset 저장
   - 24개 block × expert 수 × 모달리티 3개 → conv activation 총량 막대

2. **Frozen encoder라도 backward를 위해 전체 activation 보관**:
   - Encoder가 frozen이어도 inject된 trainable adapter의 gradient 계산을 위해 모든 중간 activation이 메모리에 유지됨
   - 24개 Hiera block의 self-attention + FFN + injected DeBA-BB adapter의 conv activation 전부

3. **Gradient checkpointing 미적용**:
   - 현재 encoder에 gradient checkpointing 없음
   - SAM head 부분에만 일부 적용 (`semseg/models/sam2/training/model/sam2.py:495`)
   - encoder activation이 VRAM의 가장 큰 비중

4. **3× forward (3 modalities)**:
   - 각 모달리티마다 encoder forward → activation 3세트 동시 보관
   - P9에서도 동일하지만, Linear vs Conv 차이로 P23에서 터짐

**해결 방안 (우선순위 순)**:

1. **Encoder gradient checkpointing** (가장 효과적, ~40-50% VRAM 절약):
   - Hiera trunk의 각 block에 `torch.utils.checkpoint.checkpoint()` 적용
   - Activation 저장 대신 backward 시 재계산 → 학습 시간 ~30% 증가
   - 적용 위치: `sam2/modeling/backbones/hieradet.py` 또는 `image_encoder.py`

2. **Frozen encoder를 torch.no_grad()로 분리** (주의 필요):
   - Frozen 부분만 no_grad로 forward → autograd graph 축소
   - 단, DeBA-BB가 encoder 내부에 inject되어 있어 단순 적용 불가
   - Frozen block output → detach → trainable adapter 입력 방식으로 재구성 필요

3. **Expert 수 / bottleneck dim 추가 축소**:
   - 1 expert로 축소 (MoE 의미 소실)
   - bottleneck_dim 축소 (표현력 감소)
   - → 근본 해결이 아닌 우회

**Inference vs Training VRAM 차이 (참고)**:
- Inference (RTX Titan 24GB): forward만 → 이전 layer activation 즉시 해제
- Training (A100 80GB): forward 전체 activation 보관 + gradient + Adam state(×2) → ~3-4배

**관련 파일**:
- `train_sam2_lora_paper.py:532` — `autocast` (AMP 이미 적용됨)
- `semseg/models/sam2/training/model/sam2.py:495` — SAM head gradient checkpointing (encoder 아님)
- `semseg/models/sam2/sam2/sam_lola_utils.py` — `MoE_DeBA_BB`, `_MoE_DeBA_BB_qkv`
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P23`

---

### ISSUE-013: P24 Teacher Signal이 Sigmoid Confidence로 구현 — CE 기반으로 수정 필요 [긴급]

**상태**: 🔴 수정 필요 (2026-03-13)
**영향**: P24 (LoRA_Sam_P24) — 학습 중 gating network supervision 무효화
**우선순위**: 최고 — 현재 학습이 의미 없는 target으로 진행 중

**문제**:

P24의 teacher quality target이 원래 계획(per-pixel CE loss)과 다르게 **sigmoid confidence**로 구현됨:

```python
# 현재 구현 (sam_lora_image_encoder_seg.py:6073-6074)
mask_prob = torch.sigmoid(teacher_logits)         # [0, 1]
confidence = torch.abs(mask_prob - 0.5) * 2       # [0, 1]
```

**관찰된 증상** (quality_vis/ 시각화):
- Epoch 1~20: Target Q에 spatial 패턴 보임 (불확실 영역 어둡게, 확실 영역 밝게)
- **Epoch 40**: Target Q가 **전부 흰색 (≈1.0)** — decoder가 학습되면서 모든 위치에서 자신감 상승 → signal 소멸

**실패 원인 분석**:
1. **GT를 사용하지 않음**: `gt_mask`가 forward에 전달되지만 teacher target 계산에 미사용
2. **Decoder 학습과 함께 포화**: decoder가 잘 학습될수록 sigmoid→0 또는 1 → `|p-0.5|*2` → 전부 1.0
3. **"Confidently wrong" 무시**: decoder가 틀린 예측을 확신해도 confidence=1.0 → 실제 quality 미반영
4. **P16 Calibrated Entropy와 동일한 실패 패턴**: 자체 confidence ≠ 실제 quality

**원래 계획 (CE 기반)**:
```python
# 수정해야 할 코드
ce_map = F.cross_entropy(teacher_logits, gt_mask, reduction='none')  # (B, H, W)
quality_target = torch.exp(-ce_map)  # GT 대비 잘 맞추면 1.0, 못 맞추면 ~0
quality_target = F.interpolate(quality_target.unsqueeze(1), size=(fpn_h, fpn_w), ...)
```

**CE 기반이 해결하는 것**:
- Decoder가 수렴해도 **모달리티별 구조적 약점은 남음**:
  - LiDAR → 수면/하늘 예측 실패 → CE 높음 → quality 낮음 유지
  - 야간 RGB → 어두운 객체 영역 CE 높음 → quality 낮음 유지
  - Thermal → 하늘/수면 경계 CE 높음 → quality 낮음 유지
- Signal이 epoch에 걸쳐 소멸하지 않음 (GT 대비 절대적 오차)
- "Confidently wrong" 문제 없음 (CE는 GT 대비 실제 오류 측정)

**추가 문제: BCE vs 4-class CE — Teacher Decoder 출력이 Binary (2026-03-13 추가)**:

현재 코딩봇 수정에서 BCE 기반으로 변경:
```python
# 현재 수정 (코딩봇)
Teacher: BCE(decoder_logit, gt_binary) → per-pixel BCE map → target = exp(-BCE)
Student: quality_gating(feat) → raw logits
Loss: BCE_with_logits(logits, target)
```

**구조적 한계**: `_teacher_decode_single()`이 SAM2 원본 binary decoder(`_forward_sam_heads`)를 사용 → 출력이 `high_res_masks: (B, 1, H, W)` — **1채널 binary logit** (object vs background).

- Binary decoder → BCE는 "여기에 뭔가 있는지 맞췄나"만 측정
- 4-class CE → "여기가 Static/Dynamic/Water/Sky 중 **뭔지** 맞췄나" — semantic 정보 포함

| | BCE (binary, 현재) | CE (4-class, 원래 계획) |
|---|---|---|
| LiDAR 수면 | "object 없음" → 맞음 → quality 높음 ❌ | Water를 Sky로 혼동 → quality 낮음 ✅ |
| RGB 야간 하늘 | "object 없음" → 맞음 → quality 높음 ❌ | Sky를 Water로 오분류 → quality 낮음 ✅ |
| 정보량 | foreground/background 2가지 | 4클래스 구분 → 풍부한 quality signal |

**권장 수정 방향**: `_teacher_decode_single()`이 4-class logits를 출력하도록 변경하거나, main decoder의 중간 출력(4-class segmentation head)을 teacher signal로 활용. 그래야 원래 계획대로 `F.cross_entropy(teacher_logits_4class, gt_mask, reduction='none')`가 가능.

**KD Loss 선택 (BCE_with_logits vs MSE)**:

Student가 teacher의 soft target을 추종하는 KD에서는 BCE_with_logits와 MSE 모두 유효:
- **BCE_with_logits**: gradient 안정적 (sigmoid 포화 시에도), probability space에서 학습
- **MSE**: 원래 계획, continuous regression으로 단순 명쾌
- 핵심은 KD loss 선택이 아니라 **teacher signal의 quality (binary vs 4-class)**

**수정 위치**:
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py:6067-6081`
- `_teacher_decode_single()`: binary decoder 대신 4-class segmentation logits를 출력하도록 수정
- Quality target 생성: `ce_map = F.cross_entropy(logits_4class, gt_mask, reduction='none', ignore_index=255)` → `quality_target = exp(-ce_map)`

**수정 시 주의사항**:
1. `_teacher_decode_single()` 출력이 현재 `(B, 1, H, W)` binary → `(B, 4, H, W)` 4-class로 변경 필요
2. 4-class logits를 얻으려면 main decoder의 segmentation head를 공유하거나, teacher용 4-class head를 별도 추가
3. `gt_mask` downsample 방식 — CE 계산 시 teacher_logits 해상도에 맞춰야 함 (nearest interpolation, ignore_index=255 처리)
4. `quality_target` 범위 — `exp(-CE)` ∈ (0, 1], CE=0이면 quality=1.0 (완벽 예측)
5. 시각화 함수 `save_p24_quality_vis`는 수정 불필요 (target은 여전히 (B, 1, H, W) ∈ [0, 1])

**추가 요청 — Quality Map 로깅 (val_multiaqua_detailed.py)** ⬜ 미구현:

- 모델이 `self._last_quality_maps`에 per-modality quality map을 이미 저장 중 (`sam_lora_image_encoder_seg.py:6176`)
  - 리스트: `[q_rgb, q_lidar, q_thermal]`, 각각 numpy array
- `val_multiaqua_detailed.py:925` 근처 (UAMM/AMF 로깅 직후)에 아래 코드 추가:

```python
# P24 Quality Gating map statistics
quality_maps = getattr(core, '_last_quality_maps', None)
if quality_maps is not None:
    img_log['quality_gating'] = {}
    for m_idx, m_name in enumerate(modals):
        qm = quality_maps[m_idx]
        img_log['quality_gating'][m_name] = {
            'mean': round(float(qm.mean()), 4),
            'std': round(float(qm.std()), 4),
            'min': round(float(qm.min()), 4),
            'max': round(float(qm.max()), 4),
        }
```

- `MISC/analyze_detailed_log.py`의 `analyze_quality_gating()` 함수가 `quality` 키워드 포함 데이터를 자동 분석

**관련 파일**:

- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: `LoRA_Sam_P24`, `_teacher_decode_single()`
- `semseg/models/sam2/sam2/modeling/sam2_base.py:257-300`: `_forward_sam_heads()` — binary mask 출력 구조
- `train_sam2_lora_paper.py:706-711`: P24 quality loss 계산
- `train_sam2_lora_paper.py:178-243`: `save_p24_quality_vis()` (수정 불필요)
- `val_multiaqua_detailed.py`: quality map 로깅 추가 대상
- `MISC/analyze_detailed_log.py`: quality map 분석 함수 이미 구현됨
- `outputs/MMSamP24/.../quality_vis/`: 현재 학습의 시각화 결과

---

### ISSUE-017: val_multiaqua_detailed.py 시각화 버그 2건 [미수정]

**상태**: 🟡 미수정 (2026-03-24 확인)
**영향**: `val_multiaqua_detailed.py` 시각화 출력
**우선순위**: 중간

**버그 1 — Row 3 `build_stats_row` 블록 선택 오류**:
- **현상**: Row 3에 Block 0, 1, 2만 표시됨 (초기 low-level 블록들)
- **원인**: `build_stats_row()` (line 699~713)에서 `REPRESENTATIVE_LAYERS`를 사용하지 않고 `sorted(Q.keys())[:3]`으로 앞 3개만 선택
- **비교**: Row 4는 `REPRESENTATIVE_LAYERS`의 중간값(Block 9)을 정상적으로 사용
- **수정**: `blocks[i]` → `REPRESENTATIVE_LAYERS` 중 데이터 있는 블록 사용
```python
# 기존 (line 703-707):
n_charts = min(len(blocks), 3)
for i in range(n_charts):
    block_idx = blocks[i]

# 수정:
target_blocks = [b for b in REPRESENTATIVE_LAYERS if b in capture.routing_data['Q']]
if not target_blocks:
    target_blocks = blocks[:3]
n_charts = len(target_blocks)
for block_idx in target_blocks:
```

**버그 2 — Expert Selection 범례 색상 구분 불량**:
- **현상**: 우하단 범례에서 E0, E1, E2가 모두 같은 색(파란색)으로 보임
- **원인**: `ax.barh([], [])` 빈 범례 아이콘이 저해상도에서 너무 작아 색 구분 불가. 또한 routing이 33/33/33 균등이라 stacked bar 자체도 구분 어려움
- **EXPERT_COLORS 정의** (line 278-283): Red/Green/Blue로 코드상 올바름
- **수정**: 범례 아이콘 크기 확대 + 폰트 키우기
```python
# 기존 (line 473):
ax.legend(fontsize=11, loc='lower right')

# 수정:
ax.legend(fontsize=14, loc='lower right',
          handlelength=2.0, handleheight=1.5,
          framealpha=0.9, edgecolor='black')
```

**관련 파일**:
- `val_multiaqua_detailed.py:278-283`: EXPERT_COLORS 정의
- `val_multiaqua_detailed.py:426-481`: `get_stats_bar_chart()` — Row 3 차트 생성
- `val_multiaqua_detailed.py:697-714`: `build_stats_row()` — Row 3 빌더

---

### ISSUE-015: P25 구조적 문제 7가지 — P26 설계 v5로 해결 [설계]

**상태**: 🟡 P26 설계 v5 완료, 구현 일부 완료 (2026-03-23)
**영향**: P25 → P26
**우선순위**: 높음

**문제 (P25 비판적 분석 7가지)**:

1. **SQG 가중치 공유**: 하나의 SQG(12.5K)로 RGB/THR/LID 3개 모달리티 동시 처리 → multi-task 충돌
2. **Triple-duty**: quality map이 UAMM/AMF/Memory 3곳에 공유 → optimization conflict (UAMM 최적 ≠ AMF 최적 ≠ Memory 최적)
3. **Teacher target 분포 편향**: `exp(-CE)` 대부분 ~1.0 → 유의미한 variation이 경계 일부에만 존재
4. **Pixel-wise max-norm 불연속**: 인접 픽셀에서 max modality 전환 시 정규화 기준 불연속 → checkerboard artifact 가능
5. **Memory modulation 이중 페널티**: UAMM에서 이미 조절된 feature의 maskmem을 다시 깎음
6. **Shared Decoder 충돌**: SAM2 decoder 1개가 3개 모달리티의 다른 feature 분포를 처리 → SQG와 동일한 multi-task 충돌
7. **MoE LoRA gate 상수 수렴**: gate가 입력/모달리티와 무관하게 고정 비율 → expert 특화 불가, 사실상 단일 LoRA

**해결 — P26 설계 v5 (7가지 변경)**:

| # | 변경 | 해결하는 문제 | 상태 |
|---|------|-------------|------|
| ① | **Per-Modality SQG** (3개 독립, +Multi-Scale fpn[0,1,2] 입력) | 문제 1 | ①-SQG분리: 구현완료, ①-MultiScale: 미구현 |
| ② | **UAMM softmax** (max-norm → softmax) | 문제 4 | 구현 완료 |
| ③ | **Relative Quality Teacher** (`softmax(-CE/tau)` + KL loss) | 문제 3 | 구현 완료 |
| ④ | **AMF output entropy 기반** (SQG와 분리) | 문제 2 | 구현 완료 |
| ⑤ | **Memory Modulation 제거** | 문제 5 | 구현 완료 |
| ⑥ | **Per-Modality Decoder 역할 분리** (auxiliary decoder ×m + shared decoder ×1) | 문제 6 | 🟡 v5 구현됨 (dual-use), **v6 설계 수정 대기** (역할 분리) |
| ⑦ | **Modality-Conditioned MoE LoRA Gate** (modality embedding → gate bias) | 문제 7 | 미구현 |
| + | min_quality 0.1→0.3 | 연쇄 약화 방지 | 구현 완료 |
| + | DeBA-FP (config on/off) | ablation용 | 미구현 |

**⑦ 관련 연구**:
- **MoE-Adapters4CL** (NeurIPS'24): LoRA-level MoE + task/domain identity embedding — 우리 설계와 가장 유사
- **VLMo/BEiT-3** (NeurIPS'22, CVPR'23): Mixture-of-Modality-Experts (hard routing)
- **Mod-Squad** (CVPR'23): Modality-aware sparse MoE + aux loss

**관련 파일**:
- `semseg/models/sam2/sam2/sam_lola_utils.py:630-730`: `SoftMoE_LoRA_Layer` (cond_dim 인프라 포함)
- `semseg/models/sam2/sam2/sam_lola_utils.py:1339-1400`: `SpatialQualityGating` 모듈
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: `LoRA_Sam_P25`/`P26` 클래스
- `train_sam2_lora_paper.py:444-457`: Decoder/Memory Attention requires_grad 설정
- `.claude_logs/02_model_arch.md`: P26 설계 v5 상세 (변경 ①~⑦, Forward 흐름, Config, 관련 연구, 리스크)

---

### ISSUE-014: RandomResizedCrop 패딩 위치 좌상단 고정 — 랜덤 배치 필요 [개선]

**상태**: 🟡 미수정 (2026-03-23 확인)
**영향**: 모든 P 버전 (P8~P25)의 학습 augmentation
**우선순위**: 중간 — 치명적이지는 않지만 학습 다양성 개선 가능

**문제**:
`RandomResizedCrop` (`semseg/augmentations_mm.py:869-915`)에서 scale < 1.0이면 이미지가 crop 크기(1024×1024)보다 작아짐.
이때 padding이 **항상 우측/하단에만** 적용됨 (line 908):
```python
padding = [0, 0, tW - W, tH - H]  # left=0, top=0, right=부족분, bottom=부족분
```

결과:
- scale < 1.0 (전체 케이스의 약 50%)에서 **이미지가 항상 좌상단 고정, 우하단 패딩**
- `margin_h = margin_w = 0`이라 random crop 위치도 (0, 0) 고정 (line 897-900)
- Val/Test는 `ResizeWidthPadToSquare`에서 **상하 균등 (중앙 정렬)** 패딩 → 학습-추론 패딩 위치 불일치
- 모델이 "좌상단 = 이미지, 우하단 = 패딩" bias를 학습할 수 있음

**수정 방안**:
line 908의 padding 로직을 랜덤 배치로 변경:
```python
# 현재:
padding = [0, 0, tW - W, tH - H]

# 수정:
pad_h = tH - H
pad_w = tW - W
pad_top = random.randint(0, pad_h)
pad_left = random.randint(0, pad_w)
padding = [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top]
```

**기대 효과**:
- 패딩 위치 랜덤화 → augmentation 다양성 증가
- Val/Test 중앙 정렬 패딩이 학습 분포에 자연스럽게 포함됨
- 위치 bias 제거

**관련 파일**:
- `semseg/augmentations_mm.py:906-913`: `RandomResizedCrop` 패딩 로직

---

### ISSUE-004: Spatial-wise Confidence Weighting → P15 구현 예정

**상태**: **P15로 구현 예정** (설계 완료, 구현 대기)
**영향**: P15
**상세 설계**: `02_model_arch.md` P15 섹션 참조

**아이디어**:
- P13/P14: confidence를 spatial mean → 이미지당 스칼라 1개 `(B, m)`
- P15: mean 없이 `(B, m, H_feat, W_feat)` 유지 → **위치마다 다른 모달리티 가중치**
- 예: 가로등 근처 RGB 토큰 → 높은 가중치, 어두운 영역 RGB 토큰 → 낮은 가중치

**P15에서 ISSUE-004와 함께 수정되는 문제 (ISSUE-009 통합)**:

1. **Spatial-wise**: `(B, m)` → `(B, m, H, W)` (본 이슈)
2. **Energy → Calibrated Entropy**: "confident but wrong" 문제 해결 (ISSUE-009)
3. **Gradient 격리**: `.detach()` 적용 (ISSUE-008 gradient 경로 문제)
4. **Aux Warmup**: 초기 N epoch uniform weight → aux head 충분히 학습 후 활성화

**기대 효과**:

- Sky 영역: LiDAR 억제 (LiDAR는 상공 포인트 없음) → Sky IoU 하락 방지
- Water 영역: RGB 억제 (야간 수면 암전) → LiDAR/Thermal 활용
- Dynamic 영역: 위치별 최적 모달리티 선택 → Dynamic IoU 개선

**전제 조건 (P14 결과에서 확인된 위험)**:

- Spatial confidence map의 정확도는 aux mask 품질에 의존
- P14에서 aux mask가 여전히 GT 대비 부정확 → spatial confidence도 부정확할 가능성
- 하지만 entropy 기반은 energy 기반보다 "confident but wrong" 케이스에 강건 (ISSUE-009 참조)

---

### ISSUE-008: Aux Head 품질 한계 — Frozen Backbone Feature 정보량 부족 [구조적]

**상태**: 🔴 확인됨 (2026-02-27). P13/P14 공통 근본 문제.
**영향**: Energy Score fusion 방식 전체 (P13, P14, P15)
**우선순위**: 높음 — aux mask 품질이 Energy Score의 전제조건

**문제**:

P13(공유 aux head)과 P14(독립 aux head) 모두에서 aux mask 품질이 GT 대비 매우 부정확. 모달리티 간 "어느 것이 낫다" 비교 자체가 불가능한 수준.

**근본 원인**: Frozen SAM2 Hiera backbone feature의 정보량 한계

1. SAM2는 자연 이미지(SA-1B)로 pretrained → 야간 수상 환경, LiDAR 점군, Thermal gradient의 모달리티별 특성이 feature에 잘 인코딩되지 않음
2. Backbone이 frozen → 새로운 도메인에 적응 불가. LoRA만으로는 feature 자체의 품질을 근본적으로 바꿀 수 없음
3. Aux decoder(Conv 2-3 layer)가 아무리 커져도 입력 feature의 정보 부족을 보상할 수 없음

**Aux decoder 크기 실험 (P13 vs P14)**:

| | P13 (공유 1개) | P14 (독립 3개) | 차이 |
|---|---|---|---|
| Aux Head | ConfidenceAuxHead (1×1 conv) | ModalAuxDecoder (3×3 conv) | 독립화 + 확대 |
| Aux mask 품질 | GT와 큰 괴리 | **소폭 개선, 여전히 부족** | 유의미한 개선 없음 |
| LiDAR UAMM | 1.0 고정 (test) | 1.0 고정 (test) | **동일** |

**구조적 한계 분석**:

| 속성 | Main Decoder (SAM2 track_step) | Aux Decoder |
|---|---|---|
| 입력 | UAMM 이후 vision_feats + **cross-modal memory** | 단일 모달리티 backbone_fpn[0] |
| 구조 | Transformer decoder + memory attention + upsampling | Conv 2-3 layer |
| 정보 | **3개 모달리티 상호 참조** | 해당 모달리티만 |
| 목적 | 최종 segmentation | 모달리티별 품질 추정 |

Aux decoder는 구조적으로 main decoder와 같아질 수 없음:
- cross-modal 정보를 쓰면 "개별 모달리티 품질 측정"이라는 목적에 부합하지 않음
- 단일 모달리티 feature만으로는 정확한 segmentation이 어려움 (특히 야간)

**Energy Score 신뢰성 조건**:

```
Aux mask 부정확 → Energy Score 무의미 (현재 상태)
Aux mask 정확하되 overconfident → Energy Score 오도됨
Aux mask 정확하고 well-calibrated → Energy Score 유효 ✓
```

Energy Score가 올바르게 작동하려면 aux mask의 **정확도 + calibration** 모두 필요.

**검토된 대안들**:

1. **Aux decoder 확대** (4-5 layer + skip connection): 소폭 개선 가능하나 frozen feature 병목 해결 불가
2. **Prototype-based aux**: gradient 오염 없음 (`.data` EMA). 하지만 선형 분류기 수준 → aux mask 품질 더 떨어질 수 있음
3. **Backbone 일부 unfreeze**: 가장 직접적이나 overfitting 위험 + SAM2 pretrained knowledge 손실 가능
4. **Label smoothing / Focal loss**: calibration 개선에 도움. 하지만 mask 정확도 자체는 안 올림

**현재 결론**: frozen backbone feature 위에서 aux mask 품질을 근본적으로 올리는 것은 구조적으로 어려움. Energy Score fusion 방식 자체의 재검토가 필요할 수 있음.

**gradient 경로 주의사항 (2026-02-27 확인)**:

현재 P14에서 energy score 계산에 `.detach()` 없음 → main loss gradient가 aux heads + LoRA에 역전파. LoRA가 두 가지 목표를 동시에 최적화:
1. 좋은 segmentation feature (main loss)
2. "적절한" energy score를 만드는 feature (간접 gradient)

→ 두 목표 충돌 가능. `compute_energy_confidence([z.detach() for z in aux_logits_list])` 로 gradient 차단 권장.

---

### ISSUE-009: Energy Score "Confident but Wrong" — Calibrated Entropy로 교체

**상태**: **P15에서 수정 예정** (설계 완료)
**영향**: P13, P14 (Energy Score 사용하는 모든 버전)
**우선순위**: 높음 — ISSUE-008과 함께 Energy Score fusion 실패의 직접 원인

**문제**:

Energy Score `E(x) = -T * logsumexp(z/T)` 는 **logit magnitude** 기반 confidence.
모달리티가 "자신있게 틀리는" 경우 오히려 높은 점수를 부여:

```
LiDAR aux head → Sky 영역에서 Water로 확신있게 오예측
→ logit: [Static=1, Dynamic=0, Water=8, Sky=0]
→ Energy Score 높음 (logsumexp ≈ 8)
→ UAMM이 LiDAR에 높은 가중치 → Sky IoU 붕괴
```

**정량적 증거**:

| 버전 | Test LiDAR UAMM | Test Sky IoU | 비고 |
| --- | --- | --- | --- |
| P9 | 0.961 (상수) | 76.54 | 고정 비율로 안정 |
| P13 | **1.000 (고정)** | 75.12 | Energy가 LiDAR 맹신 |
| P14 | **1.000 (고정)** | **36.47** | 더 심각한 맹신 |

P13/P14 모두 test에서 LiDAR UAMM=1.000, stdev=0.0000 (200장 전부 동일).
Energy Score가 LiDAR를 항상 "가장 confident"로 판정.

**P15 해결책: Calibrated Entropy**

```python
# Energy (문제): logit 크기 → confident but wrong에 취약
energy = -T * logsumexp(z / T, dim=1)
conf = -energy  # 높은 logit = 높은 confidence (위험)

# Entropy (해결): 확률 분포 균등도 → 불확실성 직접 측정
probs = softmax(z / T, dim=1)
entropy = -(probs * log(probs)).sum(dim=1)
confidence = 1 - entropy / log(num_classes)  # 0~1 정규화
```

Entropy의 장점:

- 4클래스에 골고루 분산된 예측 = 높은 entropy = 낮은 confidence
- 단일 클래스에 집중된 예측 = 낮은 entropy = 높은 confidence
- **aux head가 부정확하면** (Sky에서 모든 클래스에 비슷한 확률) → entropy 높음 → 자동 억제
- Temperature `T`로 calibration 가능 (val에서 grid search)

**한계**: aux head가 "하나의 틀린 클래스에 확신"하면 entropy도 낮음 → 여전히 실패 가능.
하지만 Energy보다는 robust: Energy는 logit magnitude만 보지만, Entropy는 분포 형태를 봄.

**관련**: ISSUE-004 (spatial-wise), ISSUE-008 (aux head 품질), `02_model_arch.md` P15 섹션

---

### ISSUE-005: 야간 합성 데이터 생성 — Diffusion 기반 Day→Night [아이디어]

**상태**: **M=85 달성을 위한 최유력 접근** (Night Aug 포화 확인됨)
**영향**: 전체 학습 파이프라인
**우선순위**: 높음 — Night Aug만으로는 +1~2pp가 한계, +7.4pp 필요

**배경**:
- Val(주간) 93% vs Test(야간) 70% 갭이 핵심 병목
- NIGHT_AUG(프로그래밍 방식)로 no-aug 대비 +33.7 개선 (35.93→69.62)
- 하지만 NIGHT_AUG는 global brightness/contrast 조절 수준 → 실제 야간과 괴리
  - 가로등 조명, 수면 반사, 불균일 조명 패턴 등 미반영
- 실제 야간 데이터 추가 수집 불가 (드론, 수상 환경)

**접근 방법**: Flux/SDXL img2img + ControlNet
```
입력: 주간 RGB (145장) + segmentation GT (ControlNet 조건)
       ↓ Flux img2img (prompt: "nighttime, dark, drone aerial view, water")
       ↓ ControlNet(segmentation map) → 구조 보존
출력: 야간 합성 RGB (145장)

LiDAR: 원본 그대로 사용 (능동 센서, 주야 무관)
Thermal: 원본 그대로 사용 (열 기반, 구조 유지)
Label: 원본 GT 그대로 사용 (ControlNet이 구조 보존)
```

**장점**:
- Flux pretrained 사용 → 별도 학습 불필요 (inference only)
- ControlNet(seg map)으로 구조 보존 → label 일관성 높음
- 다양한 야간 조건 생성 가능 (달빛, 가로등, 완전 암흑 등)
- 학습 데이터 2배 (주간 145 + 야간합성 145)

**리스크**:
- Diffusion이 수상 환경/드론 시점에 최적화되어 있지 않을 수 있음
- 생성된 야간 RGB와 원본 LiDAR/Thermal 간 consistency 검증 필요
- 생성 품질에 따라 오히려 학습에 노이즈가 될 수 있음

**실행 계획 (P13 결과 확인 후)**:
1. Flux img2img + ControlNet(seg) 파이프라인 구축
2. 10장 샘플 생성 → 품질/구조보존 검증
3. 검증 통과 시 전체 145장 생성
4. 주간+야간합성 혼합 학습 실험
5. Night-Val + Challenge 제출로 효과 측정

**관련 도구**:
- Flux.1-dev / SDXL (Hugging Face diffusers)
- ControlNet (segmentation condition)
- 별도 GPU 필요 (학습과 병렬 실행 가능)

---

### ISSUE-011: Fusion Head가 backbone_fpn[0]만 사용 — Multi-Scale 정보 미활용 [설계]

**상태**: 🔵 설계 단계 (2026-03-01)
**영향**: P9 (CrossModalFusionHead), 향후 모든 fusion head
**우선순위**: 높음 — P9의 fpn[0]-only 한계를 돌파하는 핵심 개선

**현재 상황 — 버전별 Fusion Weight 생성 방식**:

| 버전 | Fusion Weight 모듈 | 학습 가능? | FPN 사용 | 출력 형태 |
|------|-------------------|-----------|---------|----------|
| P8 | ConfidenceHeadV2 (독립 sigmoid) | O | fpn[0]만 | (B, 1) × m |
| **P9** | **CrossModalFusionHead (상대 softmax)** | **O** | **fpn[0]만** | **(B, m)** |
| P13~P16 | Entropy (규칙 기반, aux decoder 출력) | X (aux만 학습) | fpn[0]만 (aux 입력) | (B, m) 또는 spatial |
| P17 | Entropy (규칙 기반, multi-scale aux) | X (aux만 학습) | fpn 3개 (aux 입력) | (B, m, H, W) |
| P18-A | CrossModalFusionHead (P9 동일) | O | fpn[0]만 | (B, m) |

**모듈 정의 (혼동 주의)**:
- `ConfidenceHeadV2` (`sam_lola_utils.py:91`): **모달리티별 독립** 점수. Conv→GAP→MLP→sigmoid (B,1). P8 전용.
- `CrossModalFusionHead` (`sam_lola_utils.py:119`): **모달리티 간 상대 비교**. GAP→Linear→concat→MLP→softmax (B,m). P9/P18-A.
- 둘은 **완전히 다른 모듈**이며, 역할만 유사 (fusion weight 생성).

**문제**:

1. **P9의 CrossModalFusionHead**: backbone_fpn[0] (32ch, 256×256)만 GAP → 256×256 공간 정보가 단일 벡터(32-dim)로 압축. 이 벡터만으로 모달리티 품질을 판단.
2. **fpn[1] (64ch, 128×128)과 fpn[2] (256ch, 64×64)의 mid-level/semantic 정보 완전 폐기**
3. fpn[0]은 high-res spatial detail에 특화 → 모달리티별 "전반적 품질"(semantic confusion, noise level) 판단에는 fpn[2]의 deep feature가 더 적합할 수 있음
4. P17이 aux decoder에 fpn 3개를 넣어서 정보량 11배 증가시킨 것처럼, fusion head에도 동일 전략 적용 가능

**선행 실험 — P10 CrossModalFusionHeadV2 (GAP+GMP+Std) 실패 기록**:

P10에서 동일한 fpn[0]-only 구조에 multi-pool(GAP+GMP+Std)을 시도한 적 있음:
- M-score: 79.27 (P9: 81.47, **-2.2 하락**)
- 취소 사유: "Multi-pool의 Std feature가 야간에서 부정확한 quality estimation"
- 단, Oracle KL loss + ModalAuxHead가 동시 투입되어 multi-pool 단독 영향 분리 불가
- **교훈**: 같은 fpn[0] 위에서 pooling 방식만 바꿔서는 근본적 해결 불가

**근본 문제 재정의 — Scalar Fusion의 구조적 한계**:

기존 P8/P9의 GAP 기반 scalar weight (B, m)은 **이미지 전체를 하나의 스칼라로 요약**:
- LiDAR: sparse point projection → 포인트 있는 곳은 확실한 obstacle 증거, 없는 곳은 빈 공간. GAP은 이 sparse/dense 차이를 평균으로 뭉개버림
- Thermal/IR: 중앙 영역에만 실제 열 데이터, 나머지는 padding(zero). GAP에 padding 영역도 포함 → thermal 품질 과소평가
- RGB: 야간에 가로등 근처는 밝고 나머지는 암전. 밝은 영역에서만 RGB 신뢰 → GAP은 이 공간적 차이 무시

**핵심 요구사항**: 위치(h, w)마다 다른 모달리티 가중치를 학습

```
예시 (256×256 feature map 위의 한 프레임):
┌──────────────────────────────────┐
│ Sky 영역:    RGB↑  LiDAR↓        │ ← LiDAR는 상공 포인트 없음
│                                  │
│ 가로등 근처: RGB↑  Thermal↑      │ ← 두 모달리티 모두 유효
│                                  │
│ 암전 수면:   RGB↓  LiDAR↑        │ ← LiDAR 포인트가 수면 장애물 감지
│                                  │
│ Thermal 패딩 영역: Thermal↓      │ ← 빈 데이터, 기여 0
│ Thermal 중앙:      Thermal↑      │ ← 실제 열 데이터만 활용
└──────────────────────────────────┘
출력: (B, m, H, W) — 위치별 모달리티 가중치 맵
```

---

**제안: Spatial Multi-Scale Cross-Modal Fusion Head (가칭 `SpatialCrossModalFusionHead`)**:

P17의 entropy 기반 spatial weight를 **학습 가능한 모듈**로 대체.
GAP 제거, spatial 차원 유지, 1×1 Conv로 위치별 cross-modal 비교.

```
Phase A — Multi-Scale Spatial Feature 생성 (모달리티별, 가중치 공유):

  fpn[0] (32ch, 256×256) → Conv1×1(32→D)  → BN → ReLU ──────→ (B, D, 256, 256)
  fpn[1] (64ch, 128×128) → Conv1×1(64→D)  → BN → ReLU → ×2 upsample → (B, D, 256, 256)
  fpn[2] (256ch, 64×64)  → Conv1×1(256→D) → BN → ReLU → ×4 upsample → (B, D, 256, 256)
                                                    ↓
                                            concat → (B, 3D, 256, 256) per modality

  D = proj_dim (e.g., 16 or 32)
  Conv1×1 사용: spatial 차원 보존, 채널만 압축
  upsample: bilinear interpolate to fpn[0] resolution

Phase B — Spatial Cross-Modal Compare (1×1 Conv):

  3개 모달리티의 spatial feature를 채널 축으로 concat:
    modal_0: (B, 3D, H, W)
    modal_1: (B, 3D, H, W)   → concat → (B, m×3D, H, W)
    modal_2: (B, 3D, H, W)

  Spatial Compare Network:
    Conv1×1(m×3D → hidden_dim) → BN → ReLU
    Conv1×1(hidden_dim → m)                   # zero-init
    → softmax(dim=1) → (B, m, H, W)

  핵심: 1×1 Conv는 각 위치(h,w)에서 독립적으로 cross-modal 비교 수행
    - LiDAR 포인트 있는 위치 → LiDAR feature 강함 → 높은 가중치 학습
    - Thermal 패딩 위치 → Thermal feature ≈ 0 → 낮은 가중치 학습
    - RGB 암전 위치 → RGB feature 약함 → 낮은 가중치 학습
```

**P17 Entropy vs 제안 SpatialCrossModalFusionHead 비교**:

| | P17 Entropy (현재) | SpatialCrossModalFusionHead (제안) |
|---|---|---|
| 방식 | aux logits의 entropy 규칙 계산 | **학습된 Conv로 직접 weight 예측** |
| 학습 | aux decoder만 (간접) | **fusion head 자체가 학습** |
| 입력 | aux decoder 출력 (간접 feature) | **backbone FPN 3개 레벨 직접** |
| 문제 | aux mask 부정확 → entropy 부정확 (ISSUE-008) | backbone feature 직접 사용 → aux 품질 의존 없음 |
| Gradient | .detach() 필요 (gradient 격리) | main loss에서 직접 학습 가능 |
| 출력 | (B, m, H, W) | (B, m, H, W) |

**P9 대비 구조 변경 요약**:

```
P9 (현재):
  fpn[0] × 3 modalities → GAP → Linear → concat → MLP → softmax
  출력: (B, m) scalar → UAMM max-norm + AMF

SpatialCrossModalFusionHead (제안):
  fpn[0]+[1]+[2] × 3 modalities → Conv1×1 proj → upsample → concat
  → 1×1 Conv compare → softmax
  출력: (B, m, H, W) spatial → UAMM max-norm(spatial) + AMF(spatial)
```

**파라미터 추정 (D=16, hidden=32, m=3)**:

| 컴포넌트 | 계산 | 파라미터 |
|----------|------|---------|
| proj_layers (3개) | Conv1×1: 32×16 + 64×16 + 256×16 + BN×3 | ~5.7K |
| compare_net | Conv1×1: 3×48→32 + Conv1×1: 32→3 + BN | ~4.8K |
| **총계** | | **~10.5K** |

P9의 CrossModalFusionHead (~15K)보다 오히려 작음. 1×1 Conv만 사용하여 경량.

**UAMM/AMF 적용 변경**:

P9는 scalar (B, m):
```python
uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)
score = uamm_scores[:, i].unsqueeze(1)  # (B, 1)
score_expanded = score.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)
modulated = [feat * score_expanded for feat in vision_feats]
```

Spatial (B, m, H, W):
```python
uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m, H, W)
spatial_score = uamm_scores[:, i]  # (B, H, W)
# P17과 동일한 패턴: interpolate to each vision_feat resolution
for level, feat in enumerate(vision_feats[frame_idx]):
    h, w = feat_sizes[level]
    score_resized = F.interpolate(spatial_score.unsqueeze(1), size=(h, w), ...)
    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (HW, B, 1)
    modulated.append(feat * score_flat)
```
→ P17의 spatial UAMM/AMF 코드를 그대로 재사용 가능

**초기화 전략**:
- proj_layers: kaiming init (일반적)
- compare_net 마지막 Conv1×1: **zero-init** (weight=0, bias=0)
  → 초기 출력 = 0 → softmax(0,0,0) = (1/3, 1/3, 1/3) 균등
  → 기존 P9와 동일한 시작점에서 점진적 학습

**설계 결정 포인트 (향후 확정 필요)**:

1. **proj_dim D**: 16 vs 32 — 16이면 ~10K params, 32이면 ~30K params
2. **target resolution**: fpn[0] (256×256) vs 축소 (128×128) — 메모리/속도 트레이드오프
3. **새 P 버전 번호**: P19 또는 P9-V2로 명명
4. **P17/P18과의 통합**: P18에도 적용 가능 (use_entropy_fusion=False 경로 교체)

**관련 파일**:
- `semseg/models/sam2/sam2/sam_lola_utils.py`: CrossModalFusionHead (line 119-187), ConfidenceHeadV2 (line 91-116) — 신규 클래스 추가 대상
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: P9 (line 1356), P17 (line 3880) — spatial UAMM/AMF 참조
- `.claude_logs/02_model_arch.md`: P10 CrossModalFusionHeadV2 실패 기록 (line 158-218)

---

## 해결된 이슈 (Resolved Issues)

### RESOLVED-001: MoE Gate "Uniform" 분포 — 측정 Artifact

**해결일**: 2026-02-25
**영향**: P9 분석/진단

**문제**: `_gate_callback` (`sam_lola_utils.py` line 546-548)이 gate_weights의 spatial mean을 계산 → 65536개 토큰 평균이 CLT에 의해 항상 ~1/3으로 수렴 → "gate가 uniform"으로 잘못 해석

**해결**: per-token 분석 (entropy_ratio, argmax_fraction) 수행 → gate는 실제로 결정적 routing 수행 중 (Block9: entropy_ratio=0.22~0.25, max_weight=0.87)

**교훈**: 공간 평균은 per-token 다양성을 숨김. 항상 per-token 통계로 분석할 것.

---

### RESOLVED-002: P10/P11 Test 성능 하락

**해결일**: 2026-02-25 (원인 규명, P10/P11 취소)

**문제**: P10 M=79.27, P11 M=77.09 → P9(81.47) 대비 하락

**원인**:
- P10: Oracle KL loss가 주간(val) GT에 과적합 → 학습 시 oracle 있음, test 시 없음 → 메커니즘 불일치
- P11: MI loss가 이미 정상 작동하는 gate에 불필요한 제약 → 학습 방해
- 공통: 복잡도 추가 → overfitting 가속

**교훈**:
1. 학습 시와 추론 시 동일한 메커니즘을 사용해야 함
2. 이미 작동하는 컴포넌트에 추가 loss를 넣지 말 것
3. 진단(분석) 없이 loss/모듈을 추가하지 말 것

---

### RESOLVED-003: val_multiaqua_P9.py SyntaxError

**해결일**: 2026-02-25

**문제**: `from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *`가 함수 내부에 위치 → `SyntaxError: import * only allowed at module level`

**해결**: wildcard import를 `raise ValueError(f"Unknown LORA_MODEL: {lora_model_name}")` 으로 교체

---

### RESOLVED-004: Title Bar 흰색 마진 (val_multiaqua_P9.py 시각화)

**해결일**: 2026-02-25

**문제**: `_add_title_to_image()`에서 `plt.subplots()` + `tight_layout(pad=0)` 사용 → 흰색 padding 잔류

**해결**: `fig.add_axes([0, 0, 1, 1])` + `fig.patch.set_facecolor('#1a1a2e')` 로 전체 figure를 채움

---

### RESOLVED-005: val_multiaqua_detailed.py — SoftMoE_LoRA_Layer_V2 gate 호환 에러

**해결일**: 2026-03-10
**영향**: P20 (SoftMoE_LoRA_Layer_V2 사용 모델)의 detailed evaluation

**문제**:
- `val_multiaqua_detailed.py:326` hook에서 `module.gate(x)` 호출
- V1 (`SoftMoE_LoRA_Layer`): `self.gate = nn.Linear(...)` → 접근 가능
- V2 (`SoftMoE_LoRA_Layer_V2`, P20): gate 자체 미보유, 외부 `_shared_gate` 참조 → `AttributeError: 'SoftMoE_LoRA_Layer_V2' object has no attribute 'gate'`

**해결**:
```python
# Before (V1 only):
gate_logits = module.gate(x)

# After (V1/V2 compatible):
gate_fn = getattr(module, 'gate', None) or getattr(module, '_shared_gate', None)
gate_logits = gate_fn(x)
```

**주의**: hook은 forward 결과를 관찰만 하므로 (return 없음), 추론 결과에 영향 없음.

---

## 코딩 시 주의사항 (Common Pitfalls)

### 1. Checkpoint 포맷 차이
- `.pth` = raw state_dict (`torch.load()` → dict of tensors)
- `_checkpoint.pth` = `{'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}`
- `val_multiaqua.py`는 `_checkpoint.pth` 기대, `val_multiaqua_P9.py`는 `.pth` 직접 로드
- 새 스크립트 작성 시 양쪽 포맷 모두 처리할 것

### 2. LoRA 모델 import
- P8~P13이 모두 `sam_lora_image_encoder_seg.py`에 있음
- config의 `LORA_MODEL` 값으로 동적 선택: `LoRA_Sam_P8`, `LoRA_Sam_P9`, ..., `LoRA_Sam_P13`
- wildcard import (`from ... import *`)를 함수 내부에서 사용하면 SyntaxError

### 3. MULTIAQUA 데이터셋 특수사항
- 클래스: Static(0), Dynamic(1), Water(2), Sky(3), ignore(255)
- Val = 주간 145장 (정답 있음), Test = 야간 200장 (정답 없음, challenge server 평가)
- Recording Boat 영역 = ignore(255) → 평가에서 제외, 시각화 시 회색 처리
- 이미지 크기가 다양 → ResizeWidthPadToSquare로 전처리

### 4. DDP 학습 관련
- `TRAIN.DDP: True` 시 `torchrun` 또는 `torch.distributed.launch` 사용
- 단일 GPU: `train_sam2_lora_paper_singlegpu.py` 또는 DDP=False
- LoRA parameter만 학습 → backbone은 freeze

### 5. SAM2 Memory Attention 순서
- 모달리티 처리 순서: img → lidar → thermal (config의 MODALS 순서)
- 각 모달리티가 이전 모달리티의 memory를 참조
- 순서 변경 시 성능이 달라질 수 있음 (미실험)

### 6. experts_b init 수정 위치 (P13)
- `sam_lola_utils.py`의 `reset_parameters()`를 직접 수정하면 P9 등 기존 모델에 영향
- P13에서는 `__init__`에서 LoRA 설치 후 experts_b만 재초기화하는 방식으로 구현
- 기존 모델 호환성 유지

### 7. 평가 출력 디렉토리 네이밍 (2026-02-28 변경)

- **변경 전**: `val_pred/`, `test_pred/`, `eval_macvi/` (체크포인트 구분 불가, 덮어쓰기 위험)
- **변경 후**: 체크포인트 이름이 prefix로 붙음
  - `val_multiaqua.py`: `{ckpt_prefix}_val_pred/`, `{ckpt_prefix}_test_pred/`, `{ckpt_prefix}_eval_macvi/`
  - `val_multiaqua_detailed.py`: `{ckpt_prefix}_val_pred_{P버전}/`, `{ckpt_prefix}_test_pred_{P버전}/`
  - 결과 txt: `eval_{split}_{ckpt_prefix}_{timestamp}.txt`
- `ckpt_prefix` = checkpoint 파일명에서 `_checkpoint` 제거 (예: `epoch28_93.77_top1`)
- `--save_dir` 직접 지정 시 prefix 미적용 (기존 동작 유지)

### 8. P16/P17 평가 시 `_current_epoch` 설정 (2026-02-28 변경)

- P16/P17은 warmup schedule 사용 (`_current_epoch < 10` → uniform weights)
- 체크포인트 로드 시 `_current_epoch`은 저장되지 않음 → 기본값 0
- **`_current_epoch=0`이면 entropy fusion이 비활성화** (uniform 1/m으로 동작)
- `val_multiaqua.py`, `val_multiaqua_detailed.py` 모두 로드 후 `model._current_epoch = 9999` 설정
- P15 이하 모델은 `_current_epoch` 속성 없음 → `hasattr` 체크로 호환성 유지

### (이관됨) 해결된 ISSUE 블록
> 아래는 원래 "열린 이슈" 섹션에 있었으나 `[해결]` 완료되어 이 섹션으로 물리 이동된 항목이다 (2026-06-24). ID는 원본 유지.

### ISSUE-021: SAM3-RBMA train loss 정상인데 val mIoU ~2 — sem_head BatchNorm train/eval 불일치 [해결]

**상태**: ✅ 해결됨 (2026-06-19)
**영향**: `LoRA_Sam3_RBMA` (`SemanticHead`/`MultiScaleSemanticHead` 둘 다)
**우선순위**: Blocker — 평가가 무의미했음

**증상**: 멀티스케일 head 적용 + 백본 로드 정상(504/504) 후에도 **train loss main 1.9~2.2(정상 학습)인데 val mIoU=2.56**. 비교: 동일 DELIVER 25cls에서 SAM2 P28(`LoRA_Sam_P28`)은 ep2부터 Day-Val 42.67/Test 40.14 → ep10 55.26/49.41.

**근본 원인**: head의 `nn.BatchNorm2d`. head가 한 forward에서 **분포가 다른 입력으로 여러 번** 호출됨 — ① reliability용 standalone backbone feat, ② 출력용 memory-conditioned feat, ③ ×4 모달리티(img/depth/event/lidar). 공유 BN의 running_mean/var가 이 8가지 분포의 무의미한 평균이 됨 → **train(batch 통계)은 정상, eval(running 통계)은 어느 분포에도 안 맞아 붕괴**. train-good/eval-bad의 교과서적 패턴.

**수정 1차 (GroupNorm) — 실패/되돌림**: 두 head `BatchNorm2d`→`GroupNorm`. train==eval은 맞았으나 **GN이 최적화를 저해** — train loss가 main~3.2(ln25≈3.22, 거의 random)에서 안 내려감(BN 땐 ~2.0), val 0.13. GN 기각.

**수정 2차 (확정)**: `BatchNorm2d(track_running_stats=False)`. running stats 자체를 없애 **train·eval 모두 batch 통계 사용 → train==eval**(오염 원천 제거), 각 호출(standalone/mem ×4모달)이 자기 입력으로 self-normalize, **BN의 per-channel 정규화(좋은 최적화) 유지**. head spatial 288²라 eval batch 1도 안정. 검증: train/eval 출력 차이 0.0, batch1 정상.

**수정 3차 (최종 확정) — GroupNorm**: track_running_stats=False도 결국 **eval batch 통계**라 eval이 batch 의존 → **같은 ckpt(ep45)인데 trainer val=8.5 vs 단독 진단(diag_sam3_eval.py) val=1.28**. batch 독립·train==eval인 **GroupNorm**만이 정답. 2차에서 GN을 "최적화 저해"로 기각한 건 **오판**(ep0~3 warmup만 보고 판단, BN-no-rs도 그때 3.2였다가 ep38 1.4 도달). 검증: GN train==eval diff 0.0, **batch4 sample0 == batch1 단독 = 0.0**(batch 독립).
**판정 주의**: GN warmup(ep0~10) 구간 train loss는 ~3대로 보일 수 있음 → **ep20~40까지 보고** val 판정(이제 val 숫자 신뢰 가능).
**미해결(다음)**: GN 재학습 후에도 val 낮으면 = norm이 아닌 모델/구조 한계 → LoRA rank↑ 등. (diag_sam3_eval.py로 val/train-as-val per-class 재측정)

**교훈**: train-good/eval-bad → 정규화 의심. BatchNorm은 다중분포·다중호출 head에서 **양쪽 다 깨짐**(running=오염, batch-stat=batch의존). batch 독립 norm(**GroupNorm/LayerNorm**)이 정답이고, **warmup 지나서 판정**할 것.

**주의**: norm 키 변경 → 기존 체크포인트 비호환. **fresh 재학습 필수**(output 폴더 이동, AUTO_RESUME이 옛 last.pth 잡지 않게).

---

### ISSUE-020: SAM3-RBMA val mIoU ~2% — sam3.pt 가중치가 0개 로드됨 (백본 random) [해결]

**상태**: ✅ 해결됨 (2026-06-17)
**영향**: `LoRA_Sam3_RBMA` (B200 DELIVER/MULTIAQUA SAM3-RBMA 학습 전부)
**우선순위**: Blocker — 학습이 무의미했음

**증상**: B200 `b200-deliver_rgbdel_SAM3RBMA_physaug.yaml` 학습에서 val mIoU=2.05 (25클래스 random ≈4%보다 낮음), train loss `main≈27~57` (ln25≈3.2 대비 비정상적으로 높음, garbage feature 위 confident-wrong).

**근본 원인**: full `sam3.pt`(3.45GB)는 가중치를 `detector.*`(1156) / `tracker.*`(309) 네임스페이스에 저장하는데, `build_tracker()`로 만든 standalone tracker의 키는 **접두사가 없음**(`backbone.*`, `transformer.*` 등 773개). `build_sam3_tracker()`가 `tracker.load_state_dict(sd, strict=False)`를 그냥 호출 → **identity 매칭 0/773** → ViT 백본이 random인 채 frozen → LoRA(rank4)로 복구 불가 → val 붕괴.

**진단**: `diag_sam3_ckpt.py` (신규, 루트). 파일 존재/크기, ckpt vs tracker 키 접두사 히스토그램, prefix-strip/suffix remap 매칭 카운트, backbone random fraction, 바로 쓸 remap 출력. 결과: identity 0/773, `detector.` strip 520, `tracker.` strip 309, suffix-remap 717/773(backbone 504/504=random 0%).

**수정** (`semseg/models/sam3/sam3_lora_rbma.py` `build_sam3_tracker`):
- 우선순위 prefix remap — tracker 모듈은 `tracker.*`, 백본은 `detector.backbone.*`(백본은 tracker.*에 없음). shape 체크 후 매핑.
- 로그에 `loaded=N/773 (tracker.=.. detector.=..) backbone=504/504` 출력.
- **가드**: backbone 로드 <90%면 `RuntimeError` → 다신 random으로 학습 안 됨 (의도적 random은 `CHECKPOINT_PATH=''`).

**주의**: ep46까지 학습된 LoRA/sem_head는 random 백본 기준이라 폐기. `RESUME_ENABLE: false`로 처음부터 재학습 필요.

---

### ISSUE-019: P26 entropy 계산 NaN — LogBackward0 gradient explosion [해결]

**상태**: ✅ 해결됨 (2026-03-24)
**영향**: P26 학습 (Epoch 10, Iter 83에서 crash)
**우선순위**: 긴급 — 학습 불가

**에러**:
```
RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.
```

**발생 위치**: `sam_lora_image_encoder_seg.py:7038`
```python
entropy = -(prob * prob.log().clamp(min=-100)).sum(dim=1, keepdim=True)
```

**원인**:
- `prob`(softmax 출력)에 0 또는 극소값이 포함됨
- `.clamp(min=-100)`은 **forward 값**만 보호 — `log(0) → -inf → clamp → -100` (OK)
- 그러나 **backward**에서 `LogBackward0`의 gradient = `1/prob` → `1/0` = **NaN**
- Epoch 10까지는 prob이 0에 도달하지 않다가, 학습 진행으로 softmax가 한 클래스에 극단적으로 몰리면서 발생

**수정**:
```python
# 기존 (line 7038):
entropy = -(prob * prob.log().clamp(min=-100)).sum(dim=1, keepdim=True)

# 수정:
entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
```
- `prob + 1e-8`로 log 입력 자체를 0이 아니게 만듦 → forward/backward 모두 안전
- 값 변화: `log(1e-8) ≈ -18.4` → entropy 값에 미미한 영향

**관련 파일**:
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py:7038`: P26 forward 내 entropy 계산
- 동일 패턴이 다른 라인에도 있는지 확인 필요 (`prob.log()` 검색)

---

### ISSUE-018: P9/P22 UAMM 전/후 피쳐 시각화 미지원 [해결]

**상태**: ✅ 해결됨 (2026-03-24 구현 완료)
**영향**: P9, P22 (및 기타 UAMM 사용 모델)
**우선순위**: 중간 — 디버깅/분석 목적

**문제**:
- `val_multiaqua_detailed.py`는 UAMM/AMF **scalar 가중치**만 시각화 (`_last_uamm_scores`)
- UAMM 전/후의 **vision_feats 텐서** 자체는 저장/시각화 안 됨
- `val_mm_samP_detailed.py`에 AnalysisWrapper 기반 피쳐 시각화가 있지만 **P1~P7 전용** (P9/P22 미지원)

**설계 — Option 1 (모델 내부 버퍼 방식)**:

Option 2 (외부 AnalysisWrapper 확장) 대비 장점:
- P9/P22의 UAMM은 인라인 연산이라 hook으로 전/후를 잡을 수 없음
- 기존 `_last_uamm_scores` 패턴과 일관
- 모든 eval 스크립트에서 범용 접근 가능

**P9** (`LoRA_Sam_P9`, line 1365~):

| 위치 | 내용 |
|------|------|
| line ~1483 | `vision_feats[frame_idx]` — UAMM 이전 |
| line 1485 | `modulated_vision_feats = [feat * score_expanded ...]` — UAMM 적용 |
| line 1519-1527 | 기존 `_last_*` 버퍼 저장 블록 |

**P22** (`LoRA_Sam_P22`, line 5367~):

| 위치 | 내용 |
|------|------|
| line ~5513 | `vision_feats[frame_idx]` — UAMM 이전 (DeBA-FP 이후) |
| line 5515 | `modulated_vision_feats = [feat * score_expanded ...]` — UAMM 적용 |
| line 5549-5557 | 기존 `_last_*` 버퍼 저장 블록 |

**구현 내용 (P9/P22 각각 동일 패턴)**:
```python
# 1) forward 내 UAMM 루프 안에서 수집
feats_before_uamm = []
feats_after_uamm = []

for frame_idx in range(m):
    feats_before_uamm.append(vision_feats[frame_idx][0].detach().cpu())    # 추가
    modulated_vision_feats = [feat * score_expanded for feat in vision_feats[frame_idx]]
    feats_after_uamm.append(modulated_vision_feats[0].detach().cpu())      # 추가

# 2) 기존 버퍼 저장 블록에 추가
self._last_feats_before_uamm = feats_before_uamm  # list of m tensors
self._last_feats_after_uamm = feats_after_uamm    # list of m tensors
```

**시각화 코드 수정** (`val_multiaqua_detailed.py`):
- `build_uamm_amf_row()` 또는 신규 Row에서 `_last_feats_before_uamm` / `_last_feats_after_uamm` 읽기
- 채널 평균 → 2D heatmap (viridis) + diff map

**주의사항**:
- `detach().cpu()` 필수 (학습 그래프 영향 방지)
- `vision_feats[frame_idx]`는 multi-scale 리스트 → `[0]`만 저장 (FPN 최저해상도, 메모리 절약)
- 학습 시 불필요하면 `if not self.training:` 가드 추가

**수정 대상 파일**:
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: P9 forward (~4줄), P22 forward (~4줄)
- `val_multiaqua_detailed.py`: 피쳐 시각화 행 추가 + JSON 피쳐 통계 기록

---

### ISSUE-016: P26 DELIVER 학습 시 런타임 에러 6건 — 전수 해결 [해결]

**상태**: ✅ 해결됨 (2026-03-24)
**영향**: P26 (LoRA_Sam_P26) + DELIVER 4모달 학습
**우선순위**: Blocker

**에러 1~4**: (이전 세션에서 해결)
- Conv2d weight mismatch, _swap_decoder KeyError, multi_scale_sqg FPN indexing, scalp parameter 미반영

**에러 5: CheckpointError — 1932 vs 61 tensors saved**:
- **원인**: `forward()`의 `finally` 블록에서 `trunk.gradient_checkpointing = True` 복원 → backward recomputation 시 per-block checkpointing이 활성화되어 tensor count 불일치 (forward: 1932, recomputation: 61)
- **수정**: `_encode_single_modality()` 안에서 `trunk.gradient_checkpointing = False` 설정 → recomputation도 동일 설정

**에러 6: CUDA OOM (23.5GB / 24GB)**:
- **원인**: 에러 5 수정이 per-block checkpointing을 완전 비활성화 → 4모달 full activation이 메모리에 유지
- **최종 수정**: nested checkpointing 도입
  - **Outer**: `torch.utils.checkpoint`로 `_encode_single_modality()` 전체를 감싸 (per-modality)
  - **Inner**: `HieraDet.forward()`의 기존 per-block checkpointing 유지
  - `set_condition()`이 `_encode_single_modality()` 안에서 호출되므로, outer/inner recomputation 모두 올바른 `_condition` 상태 보장
  - `finally` 블록에서 gradient_checkpointing 복원 코드 제거 (불필요)
- **하위 호환성**: `trunk.gradient_checkpointing = False`일 때 (다른 P 버전) nested 아닌 단일 outer checkpoint만 동작 → 기존 코드 영향 없음

**관련 파일**:
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: `LoRA_Sam_P26._encode_single_modality()`, `forward()` finally 블록
- `semseg/models/sam2/sam2/modeling/backbones/hieradet.py:269,295-296`: per-block checkpointing

---


## 2026-07-02: E1.1 YOLO baseline 셋업 중 발견한 함정 3건

**함정 1: `/ailab_mat2` 원본 어노테이션이 jarvis split 생성 이후 변경됨**:
- `build_det_splits.py`를 지금 재실행하면 jarvis 정본과 다름: train 11,017 vs 10,535 (+500장 신규), 라벨 diff 759건 (test도 2건)
- **결론**: v2 split 비교 실험은 반드시 **jarvis `/SSDd/jemo_maeng/dset/poongsan_v2/_det_splits/det_{train,test}_v2.json` 정본** 사용. 사본: `objdet/yolo11m-rgb/splits/`
- 로더가 실제 keep한 프레임 목록도 고정: `kept_{train,test}_v2.txt` (5,862/1,772; box 18,020/5,078 교차 검증 완료). RGB 픽셀은 원본과 md5 동일 확인

**함정 2: hinton CUDA 디바이스 열거 순서 ≠ nvidia-smi 순서**:
- `CUDA_VISIBLE_DEVICES=1`이 물리 GPU0으로 매핑됨 → `CUDA_DEVICE_ORDER=PCI_BUS_ID` 설정 필요

**함정 3: ultralytics `device=N`은 절대 GPU 번호 (CVD 무시)**:
- ultralytics는 런타임에 `CUDA_VISIBLE_DEVICES`를 `device=` 값으로 덮어씀 → CVD로 GPU를 고르면 무시됨. `device=1` 직접 지정할 것
- 이 조합 때문에 학습이 공유 중인 GPU0에 올라가 OOM → ultralytics가 batch 16→4 **무음 자동 축소** (로그에 optimizer 라인 3회 반복이 흔적). iter 수(5862/batch)로 실배치 검증 권장

## 2026-07-03: poongsan lidar 결손 원인 규명 (복사 문제 아님 — 센서 주기 불일치)

- 증상: 라벨 프레임 15,153장 중 depth_map_lidar 존재는 8,538장(63%)뿐 → REQUIRE_ALL_MODALITIES 필터로 대량 탈락
- 검증: (1) drone_nas raw==Labeled==9,615장 (복사 정상), (2) /ailab_mat2도 어노테이션 참조 파일 전부 보유(누락 1,052장은 라벨셋 밖 프레임), (3) 탈락 6,615장 전부 modalities에 lidar 키 없음 + 같은 이름 lidar 파일이 원본에도 0장
- 원인: **RGB ~15Hz vs LiDAR ~10Hz 주기 차이** + 1:1 배정 정렬. capture_115624가 결정적 증거(정확히 15 vs 10fps, 결손 전부 1프레임 × 232개, 1,2,1,2 교대). 전 캡처 커버리지 50~67%, 블록 결손 없음(최장 gap ~1초)
- 복구 방법: 정렬기를 최근접-스캔 재사용(±60ms)으로 변경 시 ~100% 가능 (인접 프레임 depth 공유 trade-off). 업스트림 결정 필요
