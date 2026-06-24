# 환경 · 인프라 로그 (Environment & Infrastructure)

> 최종 업데이트: 2026-06-24
> 카테고리: **환경세팅 로그**. 학습/평가 실행 환경, 데이터/가중치 경로, 멀티GPU·B200 튜닝, 체크포인트 포맷 등 "재현·실행"에 필요한 모든 인프라 사실을 모은다.
> 명령어 원본(canonical)은 [CLAUDE.md](../CLAUDE.md) "환경 설정" 섹션. 여기는 그 위에 쌓인 인프라 변경 이력까지 포함.

---

## 1. 실행 환경

| 항목 | 값 |
|------|-----|
| Conda env | `MMSS_SAM` (`conda activate MMSS_SAM`) |
| Python | `/home/jemo/anaconda3/envs/MMSS_SAM/bin/python` |
| 학습 (SAM2) | `python train_sam2_lora_paper.py --cfg configs/<config>.yaml` |
| 학습 (단일 GPU) | `train_sam2_lora_paper_singlegpu.py` |
| 학습 (SAM3-RBMA) | `PYTHONPATH=semseg/models/sam3 [torchrun --nproc_per_node=N] python train_sam3_rbma.py --cfg <cfg>` (또는 `run_sam3_train.sh`) |
| 평가 (val/test) | `python val_multiaqua.py --cfg configs/eval_config/<cfg>.yaml --mode {val,test} --model_path <ckpt> [--macvi]` |
| P9 전용 평가 | `python val_multiaqua_P9.py --cfg <cfg> --mode {val,test}` |

## 2. 데이터셋 · 가중치 경로

| 자원 | 경로 |
|------|------|
| MULTIAQUA (야간 챌린지) | `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night` |
| DELIVER (25cls, SAM3 디버깅용) | 서버/B200 마운트 (config ROOT 참조) |
| SAM2 pretrained | `semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt` (Hiera-B+) |
| SAM2 백본 config-driven | `SAM2_CHECKPOINT` / `SAM2_CONFIG` (P28 Hiera-Large 지원) |
| SAM3 pretrained | `sam3.pt` (Meta gated — 승인 필요, 로컬 GPU 메모리 부족 → 실학습은 B200) |

⚠️ **P28 multiaqua B200 config의 ROOT/PRETRAINED 경로는 DELIVER 템플릿 기반 placeholder** — 실제 B200 마운트 확인 필요.

## 3. 체크포인트 포맷 (혼동 주의)

- `.pth` = raw `state_dict`. `val_multiaqua_P9.py`가 직접 로드.
- `_checkpoint.pth` = `{'model_state_dict', 'optimizer_state_dict', ...}` dict. `val_multiaqua.py`가 기대.
- arch 변경(BN→GN 등 키 변경) 시 옛 ckpt와 **비호환 → fresh 재학습 필수**. AUTO_RESUME이 옛 ckpt 잡지 않게 폴더 이동.

## 4. 멀티GPU · DDP

- `TRAIN.DDP: True`로 멀티GPU. 단일 GPU는 `*_singlegpu.py`.
- SAM3 트레이너: `static_graph=True` (RBMA bias hook의 'marked ready twice' 회피).
- SAM3 rank0 eval(val+test ~4k장) 시 NCCL timeout 방지 패치 적용됨.

## 5. B200 학습 파이프라인 튜닝 (2026-04-15)

단일 GPU util 10~80% 진동(데이터 파이프라인 병목) 대응 4건:

| 항목 | 위치 | 변경 |
|------|------|------|
| (a) AMP on | b200 config | `AMP: false→true` (bf16 Tensor Core) |
| (b) DataLoader | `train_sam2_lora_paper.py:687-697` | `pin_memory`, `persistent_workers`, `prefetch_factor=4` (num_workers>0 조건부) |
| (c) synchronize 제거 | `train_sam2_lora_paper.py:863` | `torch.cuda.synchronize()` 제거 (DDP all-reduce+scaler.step이 이미 동기화) |
| (d) non_blocking | `train_sam2_lora_paper.py:782,1010` | `.to(device, non_blocking=True)` train/val |

- 호환성: AMP는 config별 독립 → P9~P26 영향 없음. DataLoader 변경은 `num_workers>0` 분기로 단일 GPU 보호.
- 주의: P27/P28+AMP는 1 iter dry-run으로 `lambda_bias` grad 흐름 / float bias SDPA 수치 확인. util 여전히 불안정하면 PhysAug `P:0.40→0.20` 또는 Fourier GPU 이동.

## 6. 단일 GPU VRAM 프로브 (P26, 2026-04-10)

- `tmp/p26_amp_gc_probe/probe_p26_memory.py` — 학습 경로 따라 `max_memory_allocated/reserved`·step time 측정.
- 결과: RTX TITAN 3모달 baseline(AMP=false,GC=true) peak≈16.5GB 통과, 동일조건 AMP=true+GC=false는 ~16.9GB에서 OOM (TITAN 기준 AMP 절감 < checkpoint 해제 증가분). 3090은 FlashAttention 경로 별도 실측 필요.

---

> 관련: 실험 결과 → [03_experiment_log.md](03_experiment_log.md), 이슈 → [04_issues_and_fixes.md](04_issues_and_fixes.md), 인덱스 → [00_INDEX.md](00_INDEX.md).
