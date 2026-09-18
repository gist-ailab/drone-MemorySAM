---
created: 2026-09-18
status: 🗄 동결 — CLAUDE.md 에서 이동한 MULTIAQUA/P9 시대 주의사항 원문
---

# CLAUDE.md 구 주의사항 원문 (2026-09-18 이동)

## 주의사항

1. **Checkpoint 포맷 차이**: `.pth` = raw state_dict, `_checkpoint.pth` = `{'model_state_dict': ..., 'optimizer_state_dict': ..., ...}` 형태. `val_multiaqua.py`는 `_checkpoint.pth`를 기대하고, `val_multiaqua_P9.py`는 `.pth`를 직접 로드.
2. **Val vs Test 갭**: Val mIoU ~93-94% (주간) vs Test mIoU 58-70% (야간). 모든 모델이 이 갭을 보임.
3. **MoE Gate "Uniform" 문제**: 공간 평균(`_gate_callback`) 결과 uniform으로 보이지만, per-token 분석 시 실제로는 분화되어 있음 (entropy_ratio=0.55, max_weight=0.72). 측정 artifact임.
4. **NIGHT_AUG**: 야간 시뮬레이션 증강. hardaug4가 최종 튜닝 버전. `BRIGHTNESS_SAMPLING: dark_biased`로 극저조도 편향.
5. **DDP 학습**: `TRAIN.DDP: True`로 멀티GPU 학습. 단일 GPU 시 `train_sam2_lora_paper_singlegpu.py` 사용.
