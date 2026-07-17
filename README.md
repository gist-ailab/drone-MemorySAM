# drone-MemorySAM — 멀티모달 세그멘테이션 & 검출 연구 리포

SAM2/SAM3의 시간축 memory attention을 **모달리티 축 Cross-Modal Fusion**으로 전용하고, 그 위에
**RBMA (Reliability-Biased Memory Attention)** — training-free reliability를 memory-attention logit에
additive bias로 가산 — 를 얹는 멀티모달 인식 연구 코드베이스다.

> 이 리포는 upstream [MemorySAM (HKUST, arXiv 2503.06700)](https://arxiv.org/abs/2503.06700)에서 출발해
> 전면 개작한 연구 fork다. 원본 README는 [`_archive/upstream_MemorySAM_README.md`](_archive/upstream_MemorySAM_README.md) 참조.

## 연구 트랙

| 트랙 | 목표 | 벤치마크 |
|------|------|----------|
| **Seg — 챌린지** | MACVi MULTIAQUA Challenge (야간 수상 드론, RGB+LiDAR+Thermal) | M-score (최고 82.10, P9/P22) |
| **Seg — 논문** | RBMA 논문 publish | DELIVER val ≥66.51 / test ≥56.71, MUSES SOTA 급 |
| **Det — 국가 R&D** | poongsan indoor 멀티모달 검출 | mAP50 0.85 (egofill lidar로 달성) |

## 시작하기 (AI 세션 / 사람 공통)

1. **[`CLAUDE.md`](CLAUDE.md)** — 세션 규칙 + 프로젝트 개요 + 명령어 canonical. (에이전트 공통 지침은 [`AGENTS.md`](AGENTS.md))
2. **[`.claude_logs/00_INDEX.md`](.claude_logs/00_INDEX.md)** — 연구 로그 front door (폴더 구조 + 구번호 매핑표).
3. **[`.claude_logs/status/current.md`](.claude_logs/status/current.md)** — 현재 상태 스냅샷 (단일 출처).

## 리포 구조 요약

```
drone-MemorySAM/
├── CLAUDE.md / AGENTS.md         # 세션·에이전트 지침 (canonical)
├── .claude_logs/                 # 연구 로그 — status/ models/ experiments/ det/
│                                 #   datasets/ research/ decisions/ infra/ issues/ meta/ archive/
├── train_sam2_lora_paper.py      # 메인 학습 (DDP)
├── val_multiaqua.py              # 평가 + MACVi 제출
├── configs/                      # 학습/평가 config (det은 configs/det/)
├── semseg/models/sam2/           # SAM2 + LoRA_Sam_P8~P31 모델 정의
├── objdet/                       # detection 보조 (YOLO 기준점 등)
├── tools/                        # eval_per_domain / viz_features 등 재사용 분석 도구
├── scripts/                      # remote_exp.sh(원격 실행) · servers.conf · pick_free_gpus.sh
└── outputs/                      # 실험 산출물 (MMSamP*/, det*/)
```

## 핵심 명령

```bash
conda activate MMSS_SAM

# 학습 (로컬, 빈 GPU 자동 선택)
NGPU=4 bash run_sam.sh

# 학습 (원격 서버, tmux 세션 'jemo')
bash scripts/remote_exp.sh status bengio
bash scripts/remote_exp.sh run bengio configs/<config>.yaml auto:4

# 평가 (val / test+MACVi 제출)
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode val --model_path <ckpt>
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode test --model_path <ckpt> --macvi
```

⚠️ 어떤 학습이든 실행 전 **빈 GPU 확인 필수** — 규칙은 `CLAUDE.md` 주의사항 6번.

## 데이터셋

- **MULTIAQUA**: RGB/LiDAR/Thermal, 4클래스(Static/Dynamic/Water/Sky), 야간 test — 경로는 `CLAUDE.md`.
- **DELIVER**: img/depth/event/lidar, 논문 트랙 벤치마크.
- **poongsan indoor (det)**: COCO 포맷 멀티모달 검출, lidar egofill 파생셋은 [`.claude_logs/datasets/lidar-egofill.md`](.claude_logs/datasets/lidar-egofill.md).
