# SAM3 — vendored 업스트림 + 우리 코드 (동결 경계)

Meta SAM3 업스트림 사본 위에 우리 RBMA 포팅 코드가 얹혀 있다.
**업스트림 부분은 수정 금지**(frozen). 우리 코드만 수정 대상.

## ✅ 우리 코드 (수정 대상)
- `sam3_lora_rbma.py` — LoRA_Sam3_RBMA (Reliability-Biased Memory Attention 포팅)

## ⛔ 업스트림 frozen (수정 금지)
- `sam3/model/` · `sam3/train/` · `sam3/eval/` · `sam3/agent/` · `sam3/perflib/` — Meta SAM3
- `sam3/model_builder.py` · `sam3/io_utils.py`
- `scripts/` — 업스트림 평가/도구

> 제거됨(2026-06 정리): `examples/*.ipynb`, `assets/*.gif` — 업스트림 데모/예제.
> 필요 시 원본 SAM3 저장소에서 받을 것.
