# MemorySAM — 에이전트 지침 (Codex / Cursor / Claude 공통)

이 파일은 Codex가 우선 참고하는 **프로젝트 루트 지침**이다. 상세 스펙·모델 표·주의사항 전체는 [`CLAUDE.md`](CLAUDE.md)를 본다.

## 역할

- MACVi MULTIAQUA 멀티모달 세그멘테이션(SAM2 memory attention 기반) 연구 보조 및 엔지니어로 동작한다.
- 사용자가 한국어로 말하면 한국어로 답한다.

## 세션 시작 시

코드 수정 전에 다음을 읽고 맥락을 맞춘다.

1. **역할**: `.claude_logs/meta/bot-roles.md` — 메시지에 "코드분석봇", "코딩봇", "실험분석봇", "그림봇"이 있으면 해당 역할을 적용한다.
2. **상태**: `.claude_logs/00_INDEX.md`(front door, 구번호→새경로 매핑표) → `status/current.md`(현재 스냅샷), `models/arch-evolution.md`, `experiments/log.md`, `issues/issues-and-fixes.md` (코드 작성 전 이슈 확인).
3. **전체 지침**: [`CLAUDE.md`](CLAUDE.md) 요약·구조·실험 명령.

## 실험·코드 변경 시

- 아키텍처나 실험 config를 바꾸면 사용자에게 묻지 말고 `.claude_logs/models/arch-evolution.md` 또는 `experiments/log.md`에 [버전], [변경], [사유], [경로]를 남긴다 (새 실험은 `experiments/registry.md` 행도 갱신).
- 의미 있는 작업 완료 시 `status/current.md`를 갱신하고 이력은 `status/history-2026H2.md`에 append한다. 아키텍처 변경이 있으면 `models/arch-evolution.md`도 함께.
- 논문·보고에는 로그에 적힌 수치·팩트만 사용한다.

## 환경·도구 (이 레포는 Node/pnpm 없음)

- Conda: `conda activate MMSS_SAM` (또는 `/home/jemo/anaconda3/envs/MMSS_SAM/bin/python`).
- 학습: `python train_sam2_lora_paper.py --cfg configs/<config>.yaml`
- 단일 GPU: `train_sam2_lora_paper_singlegpu.py` 사용 가능.
- 평가(val): `python val_multiaqua.py --cfg configs/eval_config/<config>.yaml --mode val --model_path <checkpoint>`
- 평가(test / MACVi): 위에 `--mode test --macvi` 추가.
- P9 시각화·MoE: `val_multiaqua_P9.py` + 해당 eval_config.

검증이 필요하면 변경 범위에 맞는 최소 단위(해당 스크립트 dry-run, 소규모 eval, 기존 테스트)를 실행한다. 실행 못 하면 그 사실을 명시한다.

## 변경 정책

- 요청 범위 밖 파일은 건드리지 않는다. 작고 되돌리기 쉬운 변경을 선호한다.
- 동작이 바뀌면 `README.md` 또는 `docs/`·관련 config 설명을 함께 맞춘다.
- 넓게 고치기 전에 의도와 영향을 짧게 요약한다.

## 빠른 참고

- **최선 체크포인트(요약)**: P9 hardaug8_physaug ep131 — 경로는 `CLAUDE.md` 표 참고.
- **체크포인트**: `val_multiaqua.py`는 보통 `*_checkpoint.pth` 형태를 기대하고, `val_multiaqua_P9.py`는 raw `.pth` 로드 차이가 있을 수 있음 — `CLAUDE.md` 주의사항 절 참고.
- **Codex 설정**: [`.codex/config.toml`](.codex/config.toml)의 `project_doc_fallback_filenames`에 `CLAUDE.md`가 있으므로, 루트에 `AGENTS.md`가 없을 때는 `CLAUDE.md`가 폴백으로 쓰일 수 있다. **현재는 루트 `AGENTS.md`가 우선**이다.

## Skills

- 작업이 스킬 설명과 맞으면 해당 스킬을 쓰고, 범용 장문 스킬보다 목적이 분명한 스킬을 우선한다.
