# RA-L 논문 제출 트랙 — 포인터 & 실행 계획 (2026-07-15 시작)

> **원본 작업 위치 = NAS 볼트** `/nas_jm/Research/26_MultimodalSeg/_paper_submission/` (이 문서는 repo-측 포인터).
> 세션 "MMSAM | *Paper" 소유. 다른 세션은 이 문서로 논문에 필요한 실험 슬롯을 확인하고 결과를 회신한다.

## 개요

- **타깃**: RA-L (IEEE Robotics and Automation Letters) — DGFusion·CAFuser가 실린 곳. 템플릿 ieeeconf.cls + IEEEtran.bst 적용 완료.
- **논문 모델명 = ReliaDINO** (내부 코드 P34/P35/P36은 논문 텍스트 금지):
  - P34 = headline 후보 (frozen DINOv3 ViT-L/16 + per-modality LoRA r8 + cross-modal attn + calibrated competence gate) — **legal(val-selection) 최선 val 68.19 / test 56.62**
  - P36 = +Per-class Reliability Router — router 기여 ≈ +0.1 (ablation으로 강등 예정)
  - ⚠️ 07-15 판정: test-SOTA(DGFusion 56.71) **미돌파(−0.09)** — 구 "57.60 돌파"는 test-best 선정이라 철회. 논문 주장 = "동급 성능을 seg 라벨만으로(감독 최소성)" + val 목표 상회.
- **초안 상태**: abstract~conclusion 전 섹션 + references.bib + TikZ 그림 3종 + 그림 계획 작성 완료, pdflatex 컴파일 통과 (현재 9p, RA-L 한도 6+2p → 축약 필요). PDF = `_paper_submission/ReliaDINO_RAL_latest.pdf`.

## 다른 세션이 채워야 할 실험 슬롯 (canonical = 볼트 `_paper_submission/notes/experiment_plan.md`)

| # | 항목 | 채울 곳 | 우선순위 |
|---|------|---------|----------|
| 1 | P36 최종 수치 재확인 + B200 ckpt/로그 회수 검증 (`/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/B200_backup_20260715`) | Tab-ablation | 🔴 완료 여부 확인 |
| 2 | **T1: PhysAug-off + val-selection no-router 재학습** (headline 공정화; B200 소멸 → hinton/jarvis) | Tab-main headline | 🔴 |
| 3 | MUSES 공식 프로토콜 학습+test 제출 (내부 letterbox val 74.24는 비교 불가) | Tab-MUSES | 🟠 |
| 4 | MULTIAQUA ReliaDINO 학습 | Sec. IV 확장 | 🟠 |
| 5 | multi-seed (≥3) — sub-1pt 주장 전부 이것 없이는 금지 | 전체 | 🟠 |
| 6 | G0d eval-resize 관례 패리티 체크 (내부 요동 ~2pt > 마진 0.09) | Tab-main 각주 | 🔴 |
| 7 | 정성 그림 자산: per-condition 마스크 (tools/eval_per_domain.py, 최종 ckpt) + baseline 예측 | fig:qual | 🟡 |
| 8 | per-class test 감사 원표(M0-a) 소재 확인 — Wall/Bridge/Water 사망 주장의 근거 | Tab-perclass | 🔴 |

## 미해결 사실확인 (fact-check 게이트)

- MUSES SOTA 79.72/79.49(메모리) vs 벤치표 DGFusion 79.5 — 인용 전 해소.
- MM-SAM-adapter(2509.10408) DELIVER test 57.35(2-modal)·MUSES 81.07 — SOTA 주장 스코프 제한 필수.
- C1 "최초 DINOv3 멀티모달 seg" — 제출 직전 arXiv 재스윕 (근접: MMMS 2509.12963, DINOv2).
- references.bib의 `% TODO verify` 엔트리(저자/venue) 일괄 검증.

## 폴더 맵 (볼트 `_paper_submission/`)

```
ReliaDINO_RAL_latest.pdf   ← 항상 최신 렌더 (스텝마다 갱신)
drafts/                    ← 버전 스냅샷
00_PAPER_INDEX.md          ← 섹션 상태 + TODO 집계
latex/  (root.tex, sections/, figures/, references.bib, ieeeconf.cls)
figures/figure_plan.md     ← 그림 5종 스펙 + 손그림용 설명
notes/  (fact_method / fact_experiments / fact_related / experiment_plan)
```
