# SAM2 — vendored 업스트림 + 우리 코드 (동결 경계)

이 트리는 Meta SAM2 업스트림 사본 위에 우리 모델 코드가 얹혀 있다.
리팩토링/업데이트 시 **업스트림 부분은 건드리지 말 것**(frozen). 우리 코드만 수정 대상.

## ✅ 우리 코드 (수정 대상)
- `sam2/sam_lora_image_encoder_seg.py` — LoRA_Sam_P1..P28 모델 zoo (메인)
- `sam2/sam_lola_utils.py` — LoRA/MoE/SQG/fusion 유틸
- `sam2/sam_lora_image_encoder_seg_bkup.py` — 베이스 `LoRA_Sam` 클래스(위 파일이 상속)

## ⛔ 업스트림 frozen (수정 금지)
- `sam2/modeling/` · `sam2/configs/` · `sam2/utils/` — Meta SAM2 인코더/디코더/메모리어텐션
- `sam2/build_sam.py` · `sam2/sam2_image_predictor.py` · `sam2/sam2_video_predictor.py`
- `training/` · `tools/` · `sav_dataset/` — 업스트림 학습/도구(우리 파이프라인 미사용)
- `checkpoints/` — 사전학습 가중치(`*.pt`, git 미추적)

> 제거됨(2026-06 정리): `demo/`(웹UI), `notebooks/*.ipynb`, `*.gif` — 업스트림 데모/예제.
> 필요 시 원본 SAM2 저장소에서 받을 것.
