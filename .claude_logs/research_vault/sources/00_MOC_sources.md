---
title: sources/ — Map of Content
aliases: [sources MOC]
tags: [moc, sources, multimodal-segmentation]
created: 2026-07-08
status: active
---

# sources/ — Map of Content

sources 폴더(총 156파일)의 유형별 목차. 개별 논문 synthesis는 [[relatedworks/00_relatedworks_index]] 참조 — **인용은 검증된 relatedworks 노트에서, discovery DB에서 직접 인용 금지.**

## 1. 소스맵 / 인덱스 노트

- [[sources/00_imported_claude_related_work_2026-06-24]] — 사용자 Claude/NotebookLM 메모 import (프로젝트 출발점)
- [[sources/01_source_index_multimodal_segmentation]] — 1차 Semantic Scholar 소스 인덱스
- [[sources/02_source_map_multimodal_semantic_segmentation]] — 카테고리 소스맵: 멀티모달 semantic seg
- [[sources/02_source_map_multimodal_object_detection]] — 카테고리 소스맵: 멀티모달 detection
- [[sources/02_source_map_adapter_lora_foundation_seg_det]] — 카테고리 소스맵: adapter/LoRA/foundation
- [[sources/02_source_map_segmentation_detection_heads]] — 카테고리 소스맵: seg/det heads
- [[sources/03_seed_paper_verification_candidates]] — seed 논문 metadata 매칭 후보

## 2. Discovery 데이터베이스 (노이즈 많음 — 인용 전 per-paper 검증 필수)

- [[sources/02_openalex_top_venue_literature_database]] — OpenAlex 3,010건 discovery DB (top-venue 408건 플래그)
- [[sources/02_top_venue_literature_database]] — top-venue 문헌 DB (Semantic Scholar 계열)
- `sources/db/` — 위 DB의 기계가독 원본 8파일 (csv/json/jsonl/sqlite × openalex/top_venue)
- `sources/raw/` — API 원시 응답: json 5파일 + `openalex_top_venue_expansion_2026-06-24/`(66) + `top_venue_expansion_2026-06-24/`(29) + `priority_a_arxiv_2026-06-24/`(3)
- `sources/pdfs/priority_a/` — Priority A 논문 PDF 원문 10 + 추출 텍스트 `text/` 10

## 3. 스윕 로그 / 리서치 실행 기록

- [[sources/04_weekly_source_sweep_log]] — 주간 소스 스윕 로그
- [[sources/05_x_trend_watch_queries]] — X 트렌드 워치 쿼리 스캐폴드
- [[sources/06_linkedin_trend_watch_log]] — LinkedIn 트렌드 워치 로그
- [[sources/07_parallel_research_prompts_2026-07-02]] — **병렬 deep-research 8트랙 프롬프트 + 완료 기록** (Track별 결과 위치 표)
- [[sources/08_threat_watch_2026H2]] — 2026H2 위협(scoop) 감시 triage 표
- [[sources/09_gap_fill_deep_research_run_2026-07-02]] — gap-fill deep-research 실행 로그

## 4. 미승격 스텁 (⚠️ abstract-only 미검증 — 정량 인용 금지)

- [[sources/20260702_fs_sam2_adapting_segment_anything_model_2_for_few_shot_semantic_segmentation_via]] — FS-SAM2
- [[sources/20260702_clustvit_clustering_based_token_merging_for_semantic_segmentation]] — ClustViT
- [[sources/20260702_modalpatch_a_plug_and_play_module_for_robust_multi_modal_3d_object_detection_und]] — ModalPatch (검증 수치는 [[relatedworks/46_attention_reweighting_detection_nearmisses]] 참조)
- [[sources/20260702_aw_moe_all_weather_mixture_of_experts_for_robust_multi_modal_3d_object_detection]] — AW-MoE (검증 수치는 [[relatedworks/50_moe_lora_condition_routing]] 참조)

## 5. 아카이브 (relatedworks로 승격 완료된 스텁 — 2026-07-08 이동)

`sources/archive/` — 승격 노트를 인용할 것 (각 파일 상단에 승격 링크 있음):

| 스텁 | 승격 노트 |
|---|---|
| RSGMamba | [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]] |
| EQUISeg (balanced modality contributions) | [[relatedworks/62_equiseg_balanced_modality_contributions]] |
| GeomPrompt | [[relatedworks/63_geomprompt_missing_degraded_depth]] |
| MULTIAQUA | [[relatedworks/64_multiaqua_maritime_robust_training]] |
| OmniSegmentor | [[relatedworks/54_omnisegmentor_relatedwork]] |
| M⁴-SAM | [[relatedworks/48_m4sam_moe_lora_sam2_threat]] |
