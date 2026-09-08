# 00_PAPER_INDEX — ReliaDINO RA-L 투고 패키지

최종 갱신: 2026-07-15 (root.tex 조립 + 무에러 컴파일 완료)

## 1. 폴더 맵

```
_paper_submission/
├── 00_PAPER_INDEX.md          ← 이 파일 (front door)
├── latex/
│   ├── root.tex               ← 메인 파일 (조립 완료, 컴파일 OK)
│   ├── ieeeconf.cls           ← IEEE conf 클래스 (letterpaper, 10pt, conference)
│   ├── IEEEtran.bst / IEEEtranS.bst / IEEEabrv.bib
│   ├── references.bib         ← 33개 엔트리 (일부 "% TODO verify" 서지 미검증)
│   ├── template_reference_root.tex  ← ieeeconf 프리앰블 참고용 (빌드에 미사용)
│   ├── root.pdf               ← 빌드 산출물 (9쪽)
│   ├── sections/              ← 0_abstract / 1_intro / 2_related / 3_method
│   │                             / 4_experiments / 5_conclusion (.tex, body-only)
│   └── figures/               ← fig_teaser / fig_arch / fig_router (TikZ 완성)
│                                 fig_qual / fig_analysis (placeholder 프레임)
├── figures/                   ← (에셋 export 예정 위치, scripts/ TODO)
└── notes/
    ├── fact_experiments.md    ← 모든 수치의 단일 출처
    ├── fact_method.md / fact_related.md
    ├── figure_plan.md         ← Fig 4/5 에셋 소스 경로
    └── experiment_plan.md     ← ★ 다음 액션의 단일 출처
```

## 2. 빌드 방법

```bash
cd /nas_jm/Research/26_MultimodalSeg/_paper_submission/latex
pdflatex -interaction=nonstopmode root.tex
bibtex root
pdflatex -interaction=nonstopmode root.tex
pdflatex -interaction=nonstopmode root.tex
```

- 검증 상태 (2026-07-15, TeX Live 2023): **LaTeX 에러 0, Overfull hbox 0, 미해결 ref/citation 0, 중복 label 0.**
- 표 2개(`tab:sota`, `tab:ablation`)는 `\resizebox{\columnwidth}{!}{...}`로 폭 고정.
- `\todo{...}` 매크로 = 빨간 `[TODO: ...]` (root.tex에 정의).

## 3. 현재 페이지 수

**9쪽** (참고문헌 포함). RA-L 한도 = 6쪽 + 초과 2쪽(비용) = **최대 8쪽 → 현재 1쪽 초과**.
단, Fig. 4(qual)·Fig. 5(analysis)가 placeholder 프레임 상태이고 TODO 텍스트·MUSES 표 블록 등이 남아 있어 실제 분량은 유동적. 콘텐츠 삭제는 하지 않았음 — 최종 수치 반영 후 압축 판단 필요.

## 4. 섹션 상태표

| 파일 | 상태 | 남은 일 |
|---|---|---|
| `sections/0_abstract.tex` | 초안 완료 | 최종 test mIoU 1건 |
| `sections/1_intro.tex` | 초안 완료 | 최종 val/test 수치 2건 + CMNeXt per-class 확인 |
| `sections/2_related.tex` | 초안 완료 | 제출 직전 DINOv3-multimodal arXiv 재스윕 |
| `sections/3_method.tex` | 초안 완료 | ViTDet 인용 추가, full-model 파라미터 수 |
| `sections/4_experiments.tex` | 초안 완료 (수치 = fact_experiments.md 정합 확인) | 최종 수치·MUSES·G0d 패리티·T1 재실행 등 12건 |
| `sections/5_conclusion.tex` | 초안 완료 | 최종 수치, M0-a 출처 확인, 사사 |
| `figures/fig_teaser.tex` | TikZ 완성 | 최종 별표 위치/마진 갱신 |
| `figures/fig_arch.tex` | TikZ 완성 | — |
| `figures/fig_router.tex` | TikZ 완성 | — |
| `figures/fig_qual.tex` | **placeholder** | 16개 셀 이미지 + baseline 예측 + 범례 |
| `figures/fig_analysis.tex` | **placeholder** | 패널 (a)(b) matplotlib export |
| `root.tex` | 완료 | 저자/소속/펀딩 블록 |
| `references.bib` | 33 엔트리, 키 전부 매칭 | camera-ready 전 "% TODO verify" 서지 검증 |

일관성 패스 결과: 모델명 ReliaDINO로 통일, 내부 버전코드(P34/P35/P36)는 본문 렌더링 텍스트에 없음(LaTeX 주석의 출처 포인터만 잔존), `tab:main`→`tab:sota` 오참조 1건 수정, 모든 `\cite` 키 references.bib에 존재.

## 5. 통합 TODO 리스트 (`\todo{}` grep, file:line — 렌더링되는 것만)

### 최종 수치 (P36=full 학습 완료 후 일괄 반영 — B200, 2026-07-15 완주)
- `sections/0_abstract.tex:15` — final: currently 57.14, training in progress
- `sections/1_intro.tex:23` — final test mIoU; best so far 57.14
- `sections/1_intro.tex:23` — final val mIoU; best so far 67.74
- `sections/4_experiments.tex:94` — tab:sota full 행: 67.74 / ≥57.14
- `sections/4_experiments.tex:103` — 57.14 (best checkpoint at time of writing)
- `sections/4_experiments.tex:146` — tab:ablation full 행: 67.74 / 57.14
- `sections/4_experiments.tex:166` — final router pair after training completes
- `sections/5_conclusion.tex:18` — final full-model test mIoU
- `figures/fig_teaser.tex:70` — ReliaDINO 57.1 [final] (별표·마진 갱신 = root.tex:65)

### 데이터/검증 갭
- `sections/1_intro.tex:20` — 공식 CMNeXt per-class test 수치 확인 (M0-a)
- `sections/4_experiments.tex:238` — 동일 (M0-a audit artifact)
- `sections/5_conclusion.tex:34` — 동일 (M0-a 출처 확인 후 문구 확정)
- `figures/fig_analysis.tex:31` — 동일 (캡션)
- `sections/4_experiments.tex:22` — MUSES 공식 프로토콜 수치 (test-server 제출 대기)
- `sections/4_experiments.tex:96` — MUSES 표 블록 (DGFusion 79.5, CAFuser-CAA 78.5)
- `sections/4_experiments.tex:30` — full-model 정확 파라미터 수 (training log)
- `sections/3_method.tex:79` — 동일
- `sections/4_experiments.tex:56` — PhysAug-free no-router 재실행 (T1, harmonized pair)
- `sections/4_experiments.tex:272` — 동일 (fairness 절)
- `sections/4_experiments.tex:91` — DGFusion val 수치 (val n/a 확인)
- `sections/4_experiments.tex:119` — eval-protocol 패리티 확인 (G0d)
- `sections/2_related.tex:23` — 제출 시점 DINOv3-multimodal-seg arXiv 재스윕
- `sections/3_method.tex:32` — ViTDet 인용 추가 (references.bib에 엔트리도 추가)

### 그림 에셋
- `figures/fig_qual.tex:16–28` — 4조건 × (RGB/GT/baseline/ours) 16셀 이미지
- `figures/fig_qual.tex:33` — baseline(CMNeXt/CAFuser) 체크포인트 provenance
- `figures/fig_qual.tex:37` — 최종 체크포인트 예측 + 클래스 컬러 범례
- `figures/fig_analysis.tex:18` — 패널(a) per-class test IoU bar chart export
- `figures/fig_analysis.tex:23` — 패널(b) AUROC + drop-modality bar chart export
- `figures/fig_analysis.tex:38` — 가능하면 최종 full model로 진단 재생성

### 메타 (제출 요건)
- `root.tex:38–40` — 저자명 / 소속·주소·이메일 / 펀딩 footnote
- `sections/5_conclusion.tex:41` — Acknowledgments
- `references.bib` — "% TODO verify" 표시 엔트리 서지 검증 (camera-ready 전)
- 페이지 수 8쪽 이내로 압축 (현재 9쪽)

## 6. 다음 액션

**`notes/experiment_plan.md`를 볼 것** — 남은 실험(T1 재실행, MUSES 제출, M0-a 감사, G0d 패리티)의 우선순위·담당이 거기에 정리되어 있음. 최종 수치 확정 트리거 = B200 P36(full) 학습 완주(2026-07-15) 후 val-only 프로토콜로 legal pair 선택.
