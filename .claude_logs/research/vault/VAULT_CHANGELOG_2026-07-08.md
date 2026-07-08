# Vault Changelog — 2026-07-08

> 링크 무결성 수리 + MOC 정비 (Claude Code, 사용자 승인). 내용 삭제 없음 — 이동/rename/frontmatter/MOC 갱신만.

## 1. 번호 충돌 해소 (relatedworks/)

- RENAME: `relatedworks/46_benchmark_protocol_split_resolution.md` → `relatedworks/93_benchmark_protocol_split_resolution.md` (46_ 충돌 해소; 예시로 제안된 49_는 `49_corb_novelty_defense`가 이미 점유 → 빈 번호 93_ 선택)
- RENAME: `relatedworks/90_jepa_predictive_representations_for_multimodal_seg.md` → `relatedworks/91_...` (90_ 충돌 해소; `90_clustered_relatedwork_synthesis`는 참조 5건으로 유지, jepa 쪽만 이동)
- LINK UPDATE: `relatedworks/09_benchmark_tables_deliver_muses_mcubes.md` L319 — `[[relatedworks/46_benchmark_protocol_split_resolution]]` → `[[relatedworks/93_...]]`
- LINK UPDATE: `relatedworks/00_relatedworks_index.md` L139 — 46_ → 93_ 링크 + rename 주석 추가
- LINK UPDATE: `relatedworks/00_relatedworks_index.md` L164 — `[[90_jepa_...]]` → `[[91_jepa_...]]` + rename 주석 추가
- ANNOTATE: `sources/07_parallel_research_prompts_2026-07-02.md` L301 아래 — "2026-07-08 해소" 블록 추가 (표 안의 구 번호 언급은 당시 기록으로 보존)

## 2. bare 개념 wikilink 해소 (aliases)

- ALIAS ADD (frontmatter `aliases:` 삽입, 총 14개 노트):
  - `relatedworks/01_memorysam_relatedwork.md` ← [MemorySAM]
  - `relatedworks/02_dgfusion_relatedwork.md` ← [DGFusion]
  - `relatedworks/03_unimodal_bias_entropy_relatedwork.md` ← [Reducing Unimodal Bias, Reducing Unimodal Bias in Multi-Modal Semantic Segmentation]
  - `relatedworks/04_stitchfusion_relatedwork.md` ← [StitchFusion]
  - `relatedworks/05_anyseg_relatedwork.md` ← [AnySeg]
  - `relatedworks/06_deliver_muses_mcubes_dataset_note.md` ← [DELIVER, DeLiVER, MUSES, MCubeS]
  - `relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines.md` ← [CMX, TokenFusion, MAGIC, MAGIC++, CAFuser]
  - `relatedworks/20_lora_adapter_relatedwork.md` ← [LoRA]
  - `relatedworks/30_segformer_relatedwork.md` ← [SegFormer]
  - `relatedworks/31_mask2former_relatedwork.md` ← [Mask2Former]
  - `relatedworks/32_oneformer_relatedwork.md` ← [OneFormer]
  - `relatedworks/40_uncertainty_reliability_fusion_relatedwork.md` ← [TMC]
  - `relatedworks/42_attention_logit_bias_novelty_defense.md` ← [RBMA, Reliability-Biased Memory Attention]
  - `relatedworks/44_hyperdum_uncertainty_fusion_relatedwork.md` ← [HyperDUM]
- MOVE (삭제 대체): 볼트 루트 0바이트 스텁 `AnySeg.md`, `MCubeS.md`, `MUSES.md` → `.trash/` (Obsidian vault trash 규약). **이 sshfs 마운트는 unlink(삭제)가 ENOENT로 거부됨** (rename/쓰기는 허용) — 완전 삭제 불가하여 Obsidian이 인덱싱하지 않는 `.trash/`로 이동. 3파일 모두 0바이트 확인 → 내용 유실 없음. 로컬에서 마운트 원본 접근 시 `.trash/` 비우면 됨.
- UNRESOLVED (대응 노트 없음 — alias 불가, 기록만): [[UTFNet]], [[ReliFusion]], [[READ]], [[SETR]], [[DPT]], [[Expedit]], [[DToP]], [[ToMe]], [[PiToMe]], [[Token Transforming]], [[SAM]], [[SAM2]], [[ViT]] (일반 개념/미작성 노트), [[26_MultimodalSeg]] (프로젝트 자체 참조 — 아래 MOC alias로 해소), [[sources]]/[[relatedworks]]/[[material]] (폴더 링크 — 노트 아님)

## 3. 승격완료 스텁 아카이브 (sources/)

- CREATE: `sources/archive/` 폴더
- MOVE + 배너 추가 (상단 "→ 승격: [[relatedworks/NN_...]]" 1줄, 총 6건):
  - `sources/20260702_rsgmamba_*.md` → `sources/archive/` (→ relatedworks/61)
  - `sources/20260702_robust_multimodal_*balanced_modality*.md` (EQUISeg) → `sources/archive/` (→ relatedworks/62)
  - `sources/20260702_geomprompt_*.md` → `sources/archive/` (→ relatedworks/63)
  - `sources/20260702_multiaqua_*.md` → `sources/archive/` (→ relatedworks/64)
  - `sources/20260702_omnisegmentor_*.md` → `sources/archive/` (→ relatedworks/54)
  - `sources/20260702_m4_sam_*.md` → `sources/archive/` (→ relatedworks/48)
- BANNER ADD (미승격 스텁 4건, 이동 없음): `20260702_fs_sam2_*`, `20260702_clustvit_*`, `20260702_modalpatch_*`, `20260702_aw_moe_*` — 상단 "⚠️ abstract-only 미검증" 배너 추가

## 4. MOC 정비

- CREATE: `sources/00_MOC_sources.md` — sources 156파일 유형별 목차 (소스맵·인덱스 / discovery DB(db/·raw/·pdfs/) / 스윕로그 / 미승격 스텁 / archive)
- UPDATE: `00_MOC_26_MultimodalSeg.md`
  - frontmatter: `aliases: [26_MultimodalSeg]` 추가 (볼트 내 [[26_MultimodalSeg]] 링크 18건 resolve), `updated: 2026-07-08`
  - Core project folders 표: [[sources]] → [[sources/00_MOC_sources|sources]], [[relatedworks]] → [[relatedworks/00_relatedworks_index|relatedworks]] (폴더 링크 → MOC 링크)
  - "Current notes" 섹션: 2026-06-24 → 2026-07-08 구성으로 재편 (Status/our-method · Related work · Sources 3분류; P32_CoRB index, 90 synthesis, 09 canonical 벤치표, 93 protocol, material ko/en, sources MOC, 07 병렬리서치, 08 threat watch 추가). 기존 항목 삭제 없음 — 전부 유지/재배치
  - Next work queue: 벤치표(→09)·비교 matrix(→08)·related-work 문단(→90)·study material PDF(→material/01) 4항목 [ ]→[x] + 근거 링크
- UPDATE: `relatedworks/00_relatedworks_index.md` — "Index completeness additions — 2026-07-08" 섹션 추가: 누락이던 [[relatedworks/49_corb_novelty_defense]], [[relatedworks/90_clustered_relatedwork_synthesis]] 등재 (그 외 76개 노트는 기존 등재 확인) + rename 기록
- CONFIRM: `P32_CoRB/00_P32_CoRB_index.md` 링크가 00_MOC Core folders 표 + Current notes에 존재
