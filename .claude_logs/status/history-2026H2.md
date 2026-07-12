---
split_from: 01_project_status.md
created: 2026-07-08
period: 2026-07-01 ~ 2026-12-31
---

> **역할**: 2026 하반기(2026-07-01~) 역시간순 진행 로그. **새 진행 엔트리는 이 파일 최상단(이 안내 블록 바로 아래)에 추가**한다.
> 현재 상태는 [current.md](current.md), 2026-06-30 이전 이력은 [history-2026H1.md](history-2026H1.md) 참조.

## 역시간순 진행 로그 (History — 2026H2)

### P29·P31·P32·P34 표준분석 종합 완료 (동일 프로토콜 4모델) — 2026-07-12

lecun GPU0-3에서 표준 분석항목 1-4 풀 파이프라인 실행 완료. **종합 = [experiments/analysis/2026-07-12-p29-p34-standard-analysis.md](../experiments/analysis/2026-07-12-p29-p34-standard-analysis.md)**, 산출물 = NAS `/drone_nas/drone/analysis_logs/` (⚠️ HDD2 ISSUE-023 MFT 고갈 **재발** — 쓰기 불가, NAS 대체). 헤드라인: ① **P34(ep40 스냅샷) 전 도메인 1위**(mean 53.96, +1.75 vs P29) + Water 0→12 부활 = ISSUE-008(frozen backbone ceiling) 실증 ② SAM2 계열 피쳐 **rank-1 붕괴**(depth 1.1, fused 1.3) + 모달 비정렬(CKA~0.1) vs DINOv3 rank 10-20 + 정렬(0.85) ③ additive-bias 3세대 연속 no-op(P32 CoRB·P31 RBMA-eval·P34 λ1/λ2) vs **P31 router +10.7~13.8 유일 대형 기여** ④ P31 calibration loss만 geometry AUROC 수리(lidar .38→.97). 구 P29 per-domain 로그(mean 59.06)는 프로토콜 상이 확정 — 폐기. 발견 버그 수정: val.py num_classes(DELIVER=25), pipeline --label, feature_stats 채널 수 상이. P34 분석 지원(빌더 분기·eval 스태시·mm_lora 패턴·fusion 토글 5종) develop 반영.

### 표준 분석항목 1–4 도구 스위트 완성 (모델 분석 전담 체계) — 2026-07-12

**배경**: 사용자가 Seg SOTA 도달을 위한 **모델 분석 전담** 지시 + 표준 항목 4개 지정 (① VFM adapter 모달 적응도 ② 모달별 피쳐 수치+시각화(전체 테스트셋) ③ fusion/모듈 전후 비교 ④ 모델별 클래스×도메인 격차→극복 지점). **P31/32/33/34+ 공통 재사용**이 요구사항.

**산출물** (`feat/seg-analysis`→develop): worktree-p30-det의 4차원 파이프라인(seg_analysis_pipeline/adapter_health)을 develop에 포팅 + 신규 도구 4종으로 갭 마감 —
- `tools/modal_adaptation.py` (D3B, 항목①): adapter 출력이 additive인 점을 이용한 **adapter on/off A-B** — per-modal Δfeat/Δcos/Δacc로 non-RGB 적응도 직접 측정
- `tools/feature_stats.py` (D2N, 항목②): full-testset per-modal 수치 통계 — norm/dead-channel/effective-rank/cross-modal CKA + fused 통계 + PCA png
- `tools/module_ablation.py` (D5, 항목③): 모듈 toggle 전후 per-class ΔmIoU + fused-feat cos/shift — **no-op 모듈 감지기**(ISSUE-022류) 겸용
- `tools/compare_models.py` (항목④): N모델 D1 산출물 통합 → STRUCTURAL/DESIGN-GAP/DOMAIN-GAP/SOLVED 자동 분류 digest. **실데이터 검증 완료** (P29 ep100 vs ep146: STRUCTURAL=Ground/Other, Water 16pt 격차 검출)
**canonical 문서 = `tools/README_seg_analysis.md` 상단 매핑표** (experiments/00_MOC.md 등재). 원칙: 새 모델 분석 코드를 새로 짜지 말고 이 매핑으로 실행, 부족하면 도구를 확장. 산출물은 `/mnt/HDD2/src/logs/<model>_eval_<date>/` 누적.

### 차세대 아키텍처 브레인스토밍 + deep-research 완료 — 2026-07-08

**산출물**: `research/vault/material/brainstorm_next_arch_20260708.md` (proposal, 미승인). 내부 문서 전수 + 신규 deep-research 2트랙(VFM/fusion) 기반 후보 카드 5개(A DINOv3-RBMA / B SAM2-RBMA v2 / C SAM3-RBMA 2.0 / D C-RADIOv4 / E Det-deformable) → **추천 top-2 = A(본명)+B(안전판, 병행)**. 선행 검증 실험 6건 제안(최우선 = B-1: 학습 없는 consistency-신호 AUROC 스왑). doc 12 §5·doc 10 상단에 포인터 기록. 다음 액션 = 사용자 승인 + B-1 실행.

---
### P32 (CoRB) 구현·학습·분석 — corroboration 신호는 OK, 라우팅 실패 — 2026-07-06

**배경**: P32 로드맵(doc 23 seg-arch-proposals-P32, `worktree-p32-corrb` 브랜치, branch `worktree-p32-corrb`) §7 step 1-2. RBMA self-entropy(event/LiDAR anti-calibrated AUROC .30/.22)를 무학습 cross-modal corroboration으로 교체.
**구현**: `tools/eval_reliability_auroc.py`(Phase 0 진단) + `LoRA_Sam_P32(LoRA_Sam_P31)._compute_bias_source` override(corr_veto = leave-one-out 합의 Bhattacharyya + unique-info veto blend, temperature-free, config `CORROBORATION.ENABLE`, OFF→P31 byte-identical) + `val.load_model`에 P29~P32 structural threading 미러링(eval mismatch 수리) + config `b200-deliver_rgbdel_P32_physaug`(순수 ablation: P28 base+corroboration). 검증: py_compile·수식==검증도구(4.7e-6)·B200 GPU smoke.
**Phase 0(무학습 게이트) PASS**: P28 self-entropy [.77,.62,.30,.22] 재현 → corroboration이 event/LiDAR를 .54/.81로 반전. v2 재측정으로 신호형=corr_veto 확정(P31 depth workhorse .90→.28 붕괴를 veto가 .71 회복). 상세 [p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md).
**학습 결과(B200 4-GPU DDP)**: plateau **Test 53.45@ep40 / Val 61.65@ep30 < P28 55.27/63.40·P31 54.75/63.20** (−1.8/−1.8).
**분석(hinton GPU, `/mnt/HDD2/src/logs/P32_eval_20260706/`)**: **"신호는 맞고 라우팅 실패"** — corr_veto AUROC 양호(event .59/lidar .85)나 drop-modality Δ[img6.2,depth15.6,event~0,lidar~0]로 **event/LiDAR 여전히 미사용(Mode C)**. per-modal competence event/lidar≈0(viz: pred:event/lidar=노이즈). corroboration이 약모달을 up-weight해 P28 self-entropy(down-weight=쓰레기무시)보다 소폭 악화. per-domain spread 3.79(class-transfer 문제, Mode B). 구조적 사망 Bridge/Water/Wall/Other IoU~0. **교훈: signal AUROC ≠ routing 이득, hard selection 필요.**
**다음**: P32-C(PruneMem: modality dropout+hard pruning)로 drop-Δ≈0 직격.

---

### 2026-07-08 — 리포 전면 재구조화 (문서 IA + 코드 모듈화 + configs 재편 + Obsidian 볼트 수리)

**브랜치 `restructure/ia-taxonomy` (develop 기반 15커밋, draft PR). 사용자 승인 설계(Phase A 인벤토리 3종 → B 설계 → C 실행 → D 회의적 검증 PASS).**

- **문서**: `.claude_logs` 평면 번호체계 → 폴더 택소노미(status/models/experiments/det/datasets/research/decisions/infra/issues/meta/archive). 01 분할(current/history-H1/H2, diff 무손실 검증), 폴더별 00_MOC, `experiments/registry.md` 신설(실험↔config↔ckpt↔문서 허브), 00_INDEX에 구번호 매핑표. 루트 README(upstream 것 아카이브 후 교체)·CLAUDE.md·AGENTS.md·.cursorrules(타 프로젝트 잔재 제거) 갱신.
- **코드**: 메가파일 `sam_lora_image_encoder_seg.py`(375KB, 40클래스) → `lora_sam/` 패키지(base/heads/p08~p31/det/legacy + `MODEL_REGISTRY` 41종, `eval()` 제거), `sam_lola_utils.py` → `modules/{moe,fusion,reliability,common}.py`. 구경로 전부 re-export shim으로 무중단 (P9 ep131 ckpt 로드 missing/unexpected 0 + forward 검증). 위생: `.wandb_key` 추적중단(히스토리 purge 별도), val_mm.py 중복 제거, 데드 스크립트 `_archive/oneoff/` 이동.
- **configs**: 데이터셋별 재편(deliver 31/multiaqua 21/eval 39/archive 27, **파일명 불변** — output 매핑 보존), `profiles/` 서버별 경로 참조 + README(신규 명명 규칙: 서버접두어 금지). B200 학습 중 config는 구경로 symlink 보호.
- **Obsidian**: NAS 원본 볼트 수리(번호충돌 46→93·90_jepa→91, alias 14노트로 bare wikilink 해소, MOC 정비, `VAULT_CHANGELOG_2026-07-08.md`) + `scripts/sync_research_vault.sh` 신설로 repo 사본을 생성물화(손편집 금지).
- **데드 outputs**: 18개 디렉토리 ~176G **전량** → `/drone_nas/home/jemo_archive/MemorySAM_dead_outputs_20260708/` 이동 완료(검증 후 원본 삭제, HDD1 ~170G 회수). 이 과정에서 **ISSUE-023**(/mnt/HDD2 ENOSPC = NTFS MFT 레코드 고갈) 발견 → 판별 실험으로 원인 확정 → 아카이브 NAS 소산으로 완화 완료(쓰기 정상화 검증). `outputs/ARCHIVE_MANIFEST.md` 참조.
- **⚠️ 운영 주의**: 원격 서버(B200 P31 seg, jarvis det, bengio det final_full)는 **진행 중 학습 종료 전까지 이 브랜치 pull 금지** (taskboard R3). 병합 후 새 실행부터 config 신경로 사용.

*(분할 시점 2026-07-08 기준, 구 `01_project_status.md` history에는 2026-07-01 이후 날짜의 엔트리가 없었음 — 최신 엔트리 2026-06-24. 이후 이력은 여기에 누적.)*
