---
legacy_id: 20
legacy_file: 20_train_eval_optimization_audit.md
moved: 2026-07-08
---

# 학습/평가 코드 최적화 감사 (2026-06-25, hierarchy 리팩토링 Phase D)

> 대상: `train_sam2_lora_paper.py`, `train_sam2_lora_paper_singlegpu.py`, `val_multiaqua.py`,
> `val_multiaqua_detailed.py`, `semseg/augmentations_mm.py`, `semseg/datasets/multiaqua.py`, `val_mm_sam.py`.
> 원칙: **저위험·기계적 수정만 적용**, 수치/DDP 의미를 바꾸는 항목은 **미적용(권고만)** — 학습 실행 검증 후 도입.

## ✅ 이번에 적용한 저위험 수정
| # | 위치 | 변경 | 효과 |
|---|------|------|------|
| 1 | `train_sam2_lora_paper.py` epoch 말미 | `torch.cuda.empty_cache()`를 **매 epoch → eval 직전에만** 호출 | caching allocator 리셋/재-cudaMalloc 제거 (비-eval epoch) |
| 6 | 학습 hot loop | 매 iter `param_group['lr']=float(...)` **no-op 루프 제거** | iter당 Python 오버헤드 제거 |
| 7 | loader 구성 | eval loader용 `_eval_loader_kwargs` 분리(worker≤4, **persistent_workers 미사용**) | (val+night+test)×8 persistent worker 상주 → RAM/shm/fd 압박·train loader starvation 완화 |
| 2 | `val_multiaqua.py` eval loader | `pin_memory=True`+`persistent_workers`+`prefetch_factor=4`, transfer `non_blocking=True` | pageable H2D→pinned, 전송/연산 overlap |

## ⏸ 미적용 — 저위험이나 손이 큰(후속) 항목
- **#3 (강력추천)** `train_sam2_lora_paper.py` hot loop의 `.item()` 6~8회/iter → device 동기화 다발. 손실을 detach된 GPU 스칼라로 누적하고 `.item()`은 epoch당(또는 N iter마다 tqdm 갱신 시) 1회만. **학습 최대 체감 개선**. 로깅 값 동일성만 확인 후 도입.
- **#5** `augmentations_mm.py` `RandomFDA`가 샘플마다 타깃 이미지 재-decode+FFT → `__init__`에서 풀 pre-decode/pre-FFT 캐시. (FDA 켤 때만)
- **#9** `train_sam2_lora_paper_singlegpu.py` epoch마다 DataLoader 재생성 + pin_memory 없음 → 루프 밖 1회 생성, 멀티GPU 트레이너의 `_loader_kwargs` 미러링.
- **#8** `val_multiaqua_detailed.py` batch마다 `MoERoutingCapture` 재생성/hook 재등록 → 루프 밖 1회.

## ⚠️ 미적용 — 중/고위험(수치·DDP 의미 변경, 반드시 학습검증 후)
- **#13** eval forward가 fp32(autocast 밖). bf16 autocast로 감싸면 1.5~2× 빠르나 mIoU 미세 변동 가능 → delta가 noise 내인지 검증.
- **#14** DDP `find_unused_parameters=True`(+`static_graph` 없음). 참여 파라미터가 고정이면 `False`+`static_graph=True`로 backward마다 그래프 순회 제거. 단 P10/P11/P24 등 조건부 aux 출력 변형에서 hang/error 위험 → 변형별 테스트.
- **#4** eval interval마다 val set을 **두 번** 순회(mIoU 1회 + val-loss 1회, 후자는 fp32). val-loss를 evaluate 단일 패스에 접거나 최소한 autocast로 감쌀 것.
- **#10/#11/#12** per-sample Python 메트릭 루프 / PhysAug per-channel conv / PrototypeSegmentation per-class 루프 → 벡터화 가능하나 출력 동일성 검증 필요.

## 이미 최적(변경 불필요)
학습 `_loader_kwargs`의 pin_memory/persistent/prefetch/non_blocking, AMP config-gating(bf16 시 GradScaler off), per-iter `torch.cuda.synchronize()` 제거, 학습중 evaluate는 rank0 한정+`@torch.no_grad()`, 모든 eval 진입점 `@torch.no_grad()`.
