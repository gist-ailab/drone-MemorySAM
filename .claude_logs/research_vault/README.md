# research_vault/ — 옵시디언 리서치 볼트 사본 (front-door)

> **출처**: 사용자 Obsidian 볼트 `/nas_jm/Research/26_MultimodalSeg/` — **2026-07-02 동기화 사본**.
> 활용 요약본은 `../18_research_digest.md` (아이디어 회의/구현용 다이제스트).

## 포함 / 제외

- **포함**: `relatedworks/` (논문별 synthesis + novelty-defense 노트, 2-skeptic adversarial 검증 포함), `sources/` (스텁·병렬 리서치 프롬프트·주간 스윕 로그), `material/` (클러스터 related-work 초안 ko/en), `00_MOC_26_MultimodalSeg.md`.
- **제외 (NAS에만 존재)**: OpenAlex DB (3,010건 — 노이즈 많음, 인용 전 per-paper 검증 필수), source map 노트 (`sources/01`, `sources/02`), PDF 원문 (`sources/pdfs/`), 트렌드 워치 스캐폴드 (`sources/05`, `sources/06`).
- `sources/08_threat_watch_2026H2.md`(위협 triage 표)와 `sources/09_gap_fill_deep_research_run_2026-07-02.md`(gap-fill 실행 로그)는 **동기화됨** (2026-07-02 추가).

## Canonical 규칙 (충돌 시 우선순위)

1. **벤치마크 숫자** → `relatedworks/09_benchmark_tables_deliver_muses_mcubes.md`가 canonical (특히 2026-07-02 §U1–U9 확정판, [val]/[test] 태그 필수).
2. **RBMA/P28~P30 포지셔닝·프로젝트 결정** → `../12_novelty_and_related_work.md`가 canonical으로 유지.
3. **충돌 시**: 프로젝트 결정은 doc 12 우선, 외부 논문 수치는 볼트 `09` 우선.

## 파일 맵 (클러스터별)

| 클러스터 | 파일 |
|---|---|
| 인덱스/종합 | `relatedworks/00`(인덱스), `90_clustered_relatedwork_synthesis`(6클러스터+문단 후보), `52`(VFM 지형), `material/01_*` |
| 직접 baseline (seg) | `01`(MemorySAM), `02`(DGFusion), `03`(unimodal bias), `04`(StitchFusion), `05`(AnySeg), `07`(CMX/TokenFusion/MAGIC/CAFuser), `08`(비교 matrix) |
| 벤치마크/프로토콜 | `06`(dataset note), **`09`(canonical 숫자표)**, `46_benchmark_protocol_split_resolution` |
| 검출 (fusion 공유) | `10`–`14`, `15`(condition-adaptive det), `75`–`86`(2026 gap-fill det) |
| Adapter/LoRA/SAM 적응 | `20`–`23`, `53`(MM-SAM-adapter), `56`, `88`(MLE-SAM) |
| Seg/Det heads | `30`–`34`, `92`(RF-DETR) |
| Uncertainty/novelty 방어 | `40`(reliability fusion), `41`(unimodal bias), **`42`(RBMA logit-bias 방어 + fenced claim)**, `43`(A-신호 kill-check), `44`(HyperDUM), `45`(SAE near-miss), `46_attention_reweighting_*` |
| P29/P30 방어 | **`50`(MoE-LoRA condition routing)**, **`51`(class-token decoder)** |
| 2026 위협 노트 | `47`, `48`(M⁴-SAM), `54`(OmniSegmentor), `55`, `57`–`60`(SAE/BiXFormer/PRIMED), `61`–`69`(RSGMamba/EQUISeg/GeomPrompt/MULTIAQUA 등), `70`–`74`(구현 supplement) |
| SAM2 memory / JEPA | `55`(memory-attn 점유자), `90_jepa_*` |
| sources | `00`(2026-06-24 import), `03`(seed 검증 후보), `04`(주간 스윕), **`07`(병렬 리서치 8트랙 + 실행 전 필터 + 완료 기록)**, `20260702_*`(스텁 10건) |

## ⚠️ 경고

- `sources/20260702_*.md` 스텁 10건은 **abstract-only 미검증** (arXiv metadata만). 단, 이 중 RSGMamba/EQUISeg/GeomPrompt/MULTIAQUA/OmniSegmentor/M⁴-SAM은 이후 `relatedworks/61/62/63/64/54/48`로 gap-fill 검증됨 — 스텁 대신 해당 노트를 인용할 것. FS-SAM2/ClustViT/ModalPatch/AW-MoE 스텁은 원문 정독 전 정량 인용 금지 (ModalPatch/AW-MoE의 검증 수치는 `46_attention_reweighting_*`/`50`에 있음).
- 옵시디언 wiki-link `[[...]]`는 이 사본에서 링크로 동작하지 않을 수 있음 — 상대 경로로 해석할 것.
- "first/전례 없음" 류 universal negative는 전부 "to our knowledge" 헤지 + near-miss 선제 인용이 강제됨 (`42`/`43`/`50`/`51`의 fenced claim 참조).
