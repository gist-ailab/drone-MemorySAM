---
created: 2026-09-18
author: 문서체계 감사 서브에이전트 (읽기 전용 감사 — 이 문서 외 어떤 파일도 수정하지 않았다)
scope: CLAUDE.md · .claude_logs/ 전체(status/models/experiments/det/datasets/research/decisions/infra/issues/meta/archive) · 각 00_MOC.md · 00_INDEX.md · .claude/skills/*/SKILL.md 의 문서 참조
excluded: .claude_logs/meta/analysis-session-protocol.md (다른 에이전트가 작성 중 — 감사 시점에 파일 없음)
method: find/grep 로 실제 파일 목록을 뽑아 MOC 링크와 대조, 744개 상대 md 링크 전수 해석, 본문 문자열 카운트. 판정 근거는 전부 파일:줄로 표기했고 확인 못 한 것은 "미확인"으로 적었다.
---

# 문서 체계 감사 (2026-09-18)

---

## ① 요약 — 문제 상위 10개

심각도: 🔴 = 지금 잘못된 판단을 유발한다 · 🟠 = 새 세션이 필요한 정보를 못 찾거나 늦게 찾는다 · 🟡 = 구조가 앞으로 깨진다

| # | 심각도 | 문제 | 근거 |
|---|---|---|---|
| 1 | 🔴 | **현재 상태의 단일 출처가 오늘의 반증을 반영하지 않았다.** `issues/issues-and-fixes.md:20` 이 2026-09-18 자로 ISSUE-036 을 열어 "DELIVER 헤드라인 56.99 는 legal 채점이 아니라 1024 축소 GT 채점이었고 같은 ckpt 의 native 재측정은 test 55.18" 이라고 적었는데, `status/current.md` 에는 **ISSUE-036 이 0회, 55.18 이 0회**이고 56.99 가 헤드라인으로 6회 남아 있다(`status/current.md:34,45,51` 등). 같은 09-18 에 MUSES 헤드라인(79.29±0.71)은 current.md 에 반영됐으므로, 한 파일 안에서 **한쪽 벤치만 갱신되고 다른 쪽은 안 된 부분 전파**다 | `issues/issues-and-fixes.md:20` · `status/current.md:34,45,51,52` · 스크립트 대조: current.md 의 `ISSUE-036` 0건 |
| 2 | 🔴 | **철회 대상 수치가 25개 파일·94곳에 복제돼 있다.** `56.99` 는 `.claude_logs` 25개 파일에 94회 등장한다(카드 19 · issues 9 · analysis D3 7 · fair-eval-metric-protocol 6 · current 6 · registry 4 · glossary 4 …). ISSUE-036 이 확정되면 이 94곳을 사람이 손으로 쫓아야 한다. 같은 구조로 `56.71`(SOTA 기준선)은 28개 파일 172회, `79.788` 은 32개 파일 86회 | 문자열 카운트(본 감사 스크립트), 상세는 §③ |
| 3 | 🔴 | **CLAUDE.md 본문의 프로젝트 정체성이 한 세대 이상 낡았다.** §프로젝트 개요(`CLAUDE.md:102-114`)는 MULTIAQUA 챌린지 단일 벤치, §모델 버전 요약표(`CLAUDE.md:201-213`)는 P25 에서 멈추고 "현재 최선 모델 = P9 hardaug8 ep131" 이라고 단언한다. 실제 현재 제안 모델은 RxDINO(P52)이고 벤치는 DELIVER·MUSES·MCubeS 3종(`status/current.md:39-60`). 새 세션이 CLAUDE.md 만 읽고 멈추면 **틀린 최선 모델과 틀린 벤치**를 들고 시작한다 | `CLAUDE.md:104,206,213` vs `status/current.md:39-60` |
| 4 | 🔴 | **CLAUDE.md 가 존재하지 않는 파일로 명령을 안내한다.** §환경 설정의 `val_multiaqua_P9.py`(`CLAUDE.md:137-139`), §핵심 코드 구조의 `diagnose_moe_gate.py`(`:182`)·`outputs/MMSamP8~P11`(`:192-196`)·`semseg/.../checkpoints/sam2.1_hiera_base_plus.pt`(`:191`)이 **리포에 없다**(실측). §1.5 의 `lora_sam/pNN.py`·`modules/`(`:34`)도 루트에 없고 실제 경로는 `semseg/models/sam2/sam2/lora_sam/`·`semseg/models/sam2/sam2/modules/`(conventions.md:44-45 는 맞게 적혀 있다) | 경로 존재 실측 스크립트 |
| 5 | 🟠 | **00_INDEX 가 권장 읽기 순서 4번으로 STALE 판정된 문서를 지목한다.** `00_INDEX.md:14` 가 "④ meta/taskboard.md — 내 세션에 할당된 태스크 확인" 이라 하는데, 그 문서 스스로 `meta/taskboard.md:9` 에서 "🔴 STALE (2026-08-08 판정) … 이 보드의 태스크 할당을 따르지 말 것" 이라고 적는다. CLAUDE.md:27 도 같은 지목을 반복한다 | `00_INDEX.md:14` · `CLAUDE.md:27` · `meta/taskboard.md:9` |
| 6 | 🟠 | **세션 분담(감시/판정/재학습)이 진입 경로 어디에도 정의돼 있지 않다.** "생각정리 세션" 은 `.claude_logs` 16개 파일에 101회 등장하고 `status/current.md:36` 같은 진입 문서 본문에도 쓰이지만, 정의는 CLAUDE.md·00_INDEX.md·meta/conventions.md·meta/bot-roles.md·meta/00_MOC.md 어디에도 **0건**이다. `meta/bot-roles.md` 가 정의하는 역할 넷(코드분석봇/코딩봇/실험분석봇/그림봇)은 현재 운용되는 분담과 다른 어휘다 | grep 결과 0건 · `meta/bot-roles.md:14,48` |
| 7 | 🟠 | **CLAUDE.md §1 읽기 순서가 오늘 쓰는 문서 넷을 빠뜨린다.** §1 Step 1(`CLAUDE.md:18-27`)에 `experiments/plan.md`(실행 중·대기열의 단일 출처), `decisions/2026-09-07-daily-cycle-experiment-cards.md`(현재 실험 프로그램 본체), `research/hypothesis-ledger.md`(재제안 금지 목록), `meta/experiment-glossary.md`(약어 규칙 참조 정본)가 없다. 대신 `status/current.md:67-74` 의 "활성 런" 절은 2026-09-02~09-08 상태라 plan.md(09-17 실측)와 어긋난다 | `CLAUDE.md:18-27` vs `experiments/plan.md:59-90` · `status/current.md:67` |
| 8 | 🟠 | **MOC 미등록 27건.** `00_INDEX.md:3` 이 "2026-08-08 MOC 전면 재등록 — 미등록 40건 해소" 라고 선언했는데 그 뒤 6주 만에 다시 27건이 미등록이다: experiments/analysis 12건(08-09~09-08 판정 문서 전부), decisions 3건(P51·P52 개정·oracle 프로브), meta 12건(아티팩트 원본 html/svg + figures/). 즉 등록은 **일괄 청소로만 일어나고 문서 생성 시점에는 일어나지 않는다** | §② 대조표 |
| 9 | 🟡 | **한 파일이 무한히 자라는 구조가 셋 있다.** 카드 문서 1,671줄·§5 소절 35개(`decisions/2026-09-07-daily-cycle-experiment-cards.md`), `experiments/monitor-log.md` 6,410줄, `models/arch-evolution.md` 3,610줄. 카드 문서의 §5 소절 번호는 **본문 위치 순서와 일치하지 않는다**(등장 순 34,33,1,2,9,31,31b,32,30,29,…,3) — "최신을 위로" 와 "번호는 증가" 두 규칙이 충돌한다. `experiments/plan.md:3` 의 `updated:` 필드는 **한 줄에 두 달치 변경 이력이 전부 들어간 단일 라인**이다 | 줄 수 실측 · `/tmp` 헤더 파싱 |
| 10 | 🟡 | **정정·철회의 양이 구조적이다.** vault 제외 `.claude_logs` 에서 정정 126 · 무효 98 · 철회 78 · 유실 51 · 재판정 35 · 오기 15회. 상위 파일은 monitor-log 79 · 카드 54 · arch-evolution 35 · registry 32 · history-2026H2 32. 이는 "여러 세션이 같은 파일을 고치고 나중에 바로잡는" 흐름이 예외가 아니라 정상 운영이라는 뜻이다 | §⑤ |

---

## ② 구조 대조표 (폴더별 파일 수 · MOC 등록 수 · 미등록 목록)

`.claude_logs` 총 파일 327개. research/vault(143개)는 NAS 볼트 동기화 사본이라 MOC 가 폴더 1행으로만 등록하는 것이 규약(`research/00_MOC.md:12`, `meta/conventions.md:63`)이므로 미등록으로 세지 않았다.

| 폴더 | 실제 파일 | MOC 내부 링크 | 미등록 | 미등록 목록 |
|---|---|---|---|---|
| `.claude_logs/` (루트) | 22 | — | — | 00_INDEX.md + 구번호 리다이렉트 스텁 21개(각 6줄). 스텁은 매핑표(`00_INDEX.md:43-71`)로 대체 등록됨 |
| `status/` | 4 | 3 | 0 | — |
| `models/` | 17 | 16 | 0 | — |
| `experiments/` | 61 | 48 | **12** | analysis/2026-08-09-probea2-backbone-scaling · 08-17-p47-d1-muses-official-test-verdict · 08-19-spatial-modality-oracle-verdict · 08-20-fusion-mechanism-double-negative · 08-20-oracle-realizability-control-verdict · 08-20-p46-seed-variance-verdict · 08-20-spatial-axis-closure-h16 · 08-25-n4-mcubes-first-entry-verdict · 08-26-p51-cmlc-verdict-h19 · 08-27-n4b-dose-response-confirmed · 08-31-p50-gate-pass-n2-mixer-verdict · 09-08-daily-cards-E0-feature-probe |
| `det/` | 9 | 8 | 0 | — |
| `datasets/` | 3 | 2 | 0 | — |
| `research/` (vault 제외) | 8 | 8 | 0 | — |
| `research/vault/` | 143 | (폴더 1행) | 규약상 0 | — |
| `research_vault/` (구경로 잔재) | 1 | — | **1** | `research_vault/architecture/P34_ReliaDINO_design_20260712.md` — `research/00_MOC.md:15` 이 2026-08-08 에 "이관 누락, 트리 내 참조 0" 으로 미결 표기했고 6주째 그대로 |
| `decisions/` | 27 | 23 | **3** | 2026-08-18-spatial-modality-oracle-probe-proposal · 2026-08-21-p51-crossmodal-lora-coupling-proposal · **2026-08-31-p52-rxdino-adaptive-amendment** (마지막 것은 `status/current.md:61` 이 "정본" 으로 지목하는 문서인데 decisions MOC 에 없다) |
| `infra/` | 5 | 4 | 0 | — |
| `issues/` | 2 | 1 | 0 | — |
| `meta/` | 20 | 7 | **12** | approach-eval-retrospective.html · novelty-map.html · p52-validity-audit.html · rxdino-p52.html · rxdino-p52-template.html · fig1-overview.svg · fig2-c3.svg · fig3-unibal.svg · fig4-pretrain.svg · rxdino-arch.svg · figures/fig1_measurement_resolution.py · figures/fig1_measurement_resolution.svg (`status/current.md:64` 이 "아티팩트 3부작 원본 전부 meta/*.html" 이라고 가리키는 대상들) |
| `archive/` | 5 | 4 | 0 | — |

**MOC 에 있으나 존재하지 않는 파일: 0건.** (모든 MOC 내부 링크가 실재 파일로 해석됨.)

**상대 링크 깨짐 (744개 중 실질 6건, 나머지 43건은 `feats[i](...)` 같은 코드 텍스트 오탐)**

| 파일:줄 | 링크 | 문제 |
|---|---|---|
| `experiments/analysis/2026-08-06-pq-first-measurement-p48-gate.md:64` | `../decisions/2026-08-05-p48-instance-supervision-proposal.md` | 깊이 오류 — `../../decisions/` 여야 함 |
| `experiments/analysis/2026-08-16-p49-1-muses-official-verdict.md:33` | `../decisions/2026-08-10-p49-air-...md` | 같은 깊이 오류 |
| `experiments/analysis/p32-phase0-results.md:4` | `23_seg_arch_proposals_P32.md` | 구번호 참조. 이 번호는 스텁조차 없고 `00_INDEX.md` 매핑표에도 23 행이 없다 |
| `experiments/log.md:196` | `24_p32_phase0_results.md` | 깊이 오류(스텁은 `.claude_logs/` 루트) |
| `experiments/monitor-log.md:20,54` | `../synthesis/12_novelty_and_related_work.md` | `synthesis/` 폴더는 존재하지 않는다. 2026-07-08 철회된 볼트 미러 구조의 잔재 |
| `scripts/servers.conf:2` (문서 밖 소비자) | `.claude_logs/13_servers_and_launch.md` | 13번 스텁은 만들어지지 않았다(스텁 존재 번호 = 01–08,10,11,12,16–22,24,27). 매핑표에는 있으나 파일은 없다 |

**명명 규약 위반** (`meta/conventions.md:37` "kebab-case 영문 · 번호 프리픽스 신규 부여 금지")

- 구번호 스텁 21개 + `P13_design_guide.md` — 규약 이전 산물이고 스텁이므로 위반으로 치지 않되, **스텁이 22개나 루트에 떠 있어 루트가 인덱스처럼 보이지 않는다.**
- 신규 생성분 중 위반: `experiments/benchmark_roadmap.md`(snake_case) · `experiments/analysis/MUSES_TEST_RESULTS_INDEX.md`(대문자+언더스코어, 본문에서 "canonical" 자칭) · `experiments/analysis/2026-07-20-muses-official-test-P38-m2f-ep156.md`·`2026-08-03-muses-official-test-P46-c3only-lam02.md`·`2026-09-08-daily-cards-E0-feature-probe.md`(대문자 혼입) · `det/det-cert-D1-realtime.md`·`det/det-cert-D1-vitsp-handoff.md`(대문자) · `status/history-2026H1.md`·`history-2026H2.md`(대문자 — conventions.md:27 이 이 형식을 명시적으로 허용하므로 규약 내).

**두 곳 이상에서 정본을 주장하는 문서**

| 주제 | 정본 주장자 | 충돌 |
|---|---|---|
| 현재 상태 | `status/current.md:8` ("single source of truth") | `CLAUDE.md:213` 이 "현재 최선 모델 = P9" 를 직접 단언, `meta/status-report.html`(파생물로 규정됨, conventions.md:58), 노션 논문 페이지(CLAUDE.md:94 "레포 문서가 정본") |
| 실험 판정·헤드라인 | `CLAUDE.md:94` = "plan.md·카드 §5·registry 가 정본" (셋 다) | 셋 + `status/current.md` + `research/hypothesis-ledger.md` 가 같은 수치를 각자 보관. §① 문제 1·2 의 원인 |
| MUSES 공식 test | `experiments/analysis/MUSES_TEST_RESULTS_INDEX.md`(자칭 canonical) | `experiments/registry.md`(79.788 을 10회) · `status/current.md:36,52` · `models/arch-evolution.md`(8회) |
| 실험 결과 | `experiments/log.md`("canonical", `experiments/00_MOC.md:9`) | `experiments/registry.md`("허브"), `monitor-log.md`, 카드 §5 |
| 아키텍처 | `models/arch-evolution.md`("canonical", `models/00_MOC.md:7`) | 최신 항목이 P47-2(2026-08-04)에서 멈춰 P48~P53·RxDINO 는 `decisions/` 에만 있다 |
| 구조 규칙 | `meta/conventions.md:7`("단일 출처") | `CLAUDE.md:29-41` 이 요약본을 두는데 §1.5 의 코드 경로가 conventions.md 와 다르다(§① 문제 4) |

---

## ③ 쓰레기·모순 목록

### 3-1. CLAUDE.md 본문

| 파일:줄 | 무엇이 왜 문제인가 | 권고 |
|---|---|---|
| `CLAUDE.md:104-114` | 프로젝트 개요가 MULTIAQUA 챌린지 전용. 챌린지는 종료·고정(`status/current.md:56`)이고 현재 트랙은 DELIVER·MUSES·MCubeS + det 인증 | **정정** — 개요를 현재 3벤치+det 로 교체, MULTIAQUA 는 "종료된 트랙" 한 줄로 |
| `CLAUDE.md:201-213` | 모델 버전 요약표가 P25 까지. "현재 최선 = P9, M-score 81.98" 은 `experiments/registry.md:18`(82.10)과도 어긋나고 계보상 여섯 세대 전 | **삭제** — 표를 지우고 `models/00_MOC.md` + `status/current.md` 로 포인터 한 줄 |
| `CLAUDE.md:137-139,182,191-196` | 없는 파일로 실행 명령·트리를 안내(`val_multiaqua_P9.py`·`diagnose_moe_gate.py`·`outputs/MMSam*`·SAM2 ckpt) | **정정** — 실측 트리로 교체(`val.py`, `train_reliadino.py`, `semseg/models/reliadino/`, `tools/`) |
| `CLAUDE.md:34` | "새 모델 버전은 `lora_sam/pNN.py`, 공통 모듈은 `modules/`" — 루트에 그 경로가 없다. 정확한 경로는 `semseg/models/sam2/sam2/lora_sam/`·`.../modules/`(conventions.md:44-45) | **정정** — 요약에서 경로를 빼고 conventions.md 로만 보내거나 전체 경로로 |
| `CLAUDE.md:219-223` | 주의사항 1~4 가 전부 MULTIAQUA/P9 시대(`val_multiaqua.py` 포맷, Val 93-94% vs Test 58-70%, MoE gate uniform, NIGHT_AUG hardaug4). 현재 지배적인 함정은 ISSUE-033(채점 드라이버)·ISSUE-034(덤프 파일명 평탄화)·ISSUE-036(GT 해상도) | **이동** — 옛 4건은 `archive/` 로, 자리에는 issues 상단 표 포인터 |
| `CLAUDE.md:18-27` | 읽기 순서에 plan.md·카드·hypothesis-ledger·experiment-glossary 누락, 대신 STALE taskboard 지목 | **정정** — §④ 참조 |
| `CLAUDE.md:59-66` (§1.7) | "✅ P37 병합 완료(2026-07-28 확인)" 로 **이미 해소된 경고**가 본문에 남아 있다. 같은 절의 "📦 브랜치 통합(2026-07-28)" 도 완료 공지 | **이동** — 해소 공지는 history 로, §1.7 본문은 규칙만 |
| `CLAUDE.md:157-162` vs `meta/conventions.md:57` | 산출물 저장 위치가 서로 다르다: CLAUDE.md = `/drone_nas/.../drone-MemorySAM/{ckpts,analysis_logs,train_logs}` (2026-07-17 재확정) / conventions.md = `/mnt/HDD2/src/logs/<model>_eval_<date>/` (ISSUE-023 시절) | **정정** — conventions.md:57 을 현행 루트로 교체 |

### 3-2. 상위 규칙(GLM.md 등)과의 충돌

| 충돌 | 내용 | 권고 |
|---|---|---|
| 워커 위임 vs 모델 위임 | 전역 `GLM.md` 는 "코드를 쓰는 단계마다 **먼저 사용자에게 어느 워커(labcode/glmcode)로 할지 묻고** 위임" 이라 하고, 프로젝트 `CLAUDE.md:43-57`(§1.6)·`meta/conventions.md:71-78` 은 "코드를 만지는 일 = **해당 세션의 opus/fable 이 직접**, sonnet 은 집행" 이라 한다. 두 규칙은 같은 행위(코드 작성)에 다른 주체를 지정한다 | **판정 필요** — 프로젝트 CLAUDE.md 가 우선한다는 것이 전역 `OPERATING.md` 의 명문이므로 CLAUDE.md §1.6 에 "이 리포에서는 GLM.md 의 워커 위임을 적용하지 않는다(또는 적용한다)" 를 한 줄 명시 |
| 워크트리 git | 전역 메모리 `worktree-git-via-usr-bin-git` 는 "워크트리 격리 세션에서 rtk 훅이 `git` 을 거부 → `/usr/bin/git` 절대경로" 를 기록하지만, `meta/conventions.md:14-18` 의 git 규칙에는 이 제약이 없다. 실제로 이 감사 중에도 rtk 훅이 복합 bash 명령을 거부했다(재현됨) | **정정** — conventions.md §1 에 워크트리 세션 제약 한 줄 추가 |
| lecun 배치 금지 | 전역 메모리(user 2026-09-17)는 "lecun 에 어떤 작업도 더 배치하지 않음"인데, `scripts/servers.conf:23` 의 lecun 행에는 `policy` 필드가 없어 `remote_exp.sh` 가 여전히 기동을 허용한다. `experiments/plan.md:67` 은 "입출력 병목으로 학습 자리에서 제외" 라는 **다른 이유**를 적는다 | **정정** — servers.conf lecun 행에 `off` 정책, plan.md 에 사유를 user 지시로 교정 |

### 3-3. 낡은 내용·해소된 경고가 활성 폴더에 남은 것

| 파일:줄 | 문제 | 권고 |
|---|---|---|
| `meta/taskboard.md:9-70` | 스스로 STALE 선언(2026-07-03 이후 미갱신). 목표 수치(DELIVER val≥66.51/test≥56.71, det 0.85)가 현행(69.60/57.35, det 달성 0.9321)과 다르고, 태스크 전부 P29~P31/B200 시대 | **이동** → `archive/2026-07-03-supervisor-taskboard.md`. 00_INDEX:14·CLAUDE.md:27 의 지목도 함께 제거 |
| `meta/bot-roles.md:25,36-37,56,64,68` | 로깅 대상으로 `02_model_arch.md`·`04_issues_and_fixes.md`(리다이렉트 스텁), 설계 가이드로 `P13_design_guide.md`(archive), 분석 대상으로 `val_multiaqua_P9.py`(없는 파일), 모델 범위 `LoRA_Sam_P8~P19` 를 적는다. **CLAUDE.md Step 0 이 세션 최초로 읽으라는 문서**가 이렇다 | **정정**(경로·범위 갱신) 또는 현 분담 어휘로 **재작성** |
| `models/arch-evolution.md:8` | "최종 업데이트 2026-08-04" + 최신 절이 P47-2. P48·P49·P50·P51·P52(RxDINO)·P53 부재. `models/00_MOC.md:7` 은 "P8~P31" 로, `CLAUDE.md:20` 도 "P8~P31" 로 적어 **세 곳의 범위 표기가 전부 다르다** | **정정** — 범위 표기 통일 + P48~P53 절 추가(또는 "P48 이후는 decisions/ 가 정본" 을 명시) |
| `models/arch-evolution.md:481` | 절 제목이 "P9: … (현재 최선)" | **정정** |
| `experiments/registry.md:3` | frontmatter `updated: 2026-08-08` 인데 본문 행은 2026-09-17 까지 갱신돼 있다 | **정정** |
| `experiments/registry.md:28,39` | P31 행 "🟡 active (학습 중, 현 최선 DELIVER)" · SAM3-RBMA 행 "🟡 학습/디버깅 중" — B200 은 2026-07-15 상실(`experiments/plan.md:68`)이라 그 학습은 존재할 수 없다 | **정정** — 상태를 종료/동결로 |
| `experiments/registry.md:26-27` | 한 셀에 수백 단어의 서사(DGFusion 행은 단일 셀에 NaN 원인·20개 ckpt 전체 수치·정정 이력까지). "한눈표" 라는 문서 역할(`experiments/00_MOC.md:10`)과 정반대 | **분리** — 서사는 analysis 문서로, registry 셀은 결론 1행 + 링크 |
| `issues/issues-and-fixes.md:9` | "최종 업데이트: 2026-08-06" 인데 표에 ISSUE-034~036(09-17~18)이 있다 | **정정** |
| `issues/issues-and-fixes.md:40-55` | ISSUE-001~017 대부분이 MULTIAQUA/P9~P26 시대이고 상태가 "🟠 진행"·"⚪ 예정" 으로 1년 가까이 고정. 상단 인덱스 표를 "먼저 보라" 고 하는데 표 절반이 사문 | **이동** — 종결 판정 후 "해결된 이슈" 절 또는 archive 로 |
| `research/novelty-and-related-work.md:8` | canonical 문서 본문이 `10_related_work.md`(구번호)를 가리킨다 | **정정** — `related-work-raw.md` 로 |
| `research/00_MOC.md:15` | "⚠️ 미결(2026-08-08): 구경로 `research_vault/architecture/…` 잔존" — 6주째 미해소, 파일도 그대로 | **판정 필요** — NAS 볼트 대조 후 이동 또는 삭제 |
| `det/diagnosis-plan.md:8,11` | 목표 대비 현재값이 0.4455/0.2490(2026-07-02)로 적혀 있으나 det 는 0.9321 로 종결 국면(`status/current.md:55`). 코드 위치로 `.claude/worktrees/p30-det/`·`p29-det/` 를 지목하는데 그 브랜치들은 2026-07-28 통합·태그 보존됐다(`CLAUDE.md:65`). 워크트리 실재 여부는 **미확인**(이 감사 세션은 워크트리 안이라 메인 체크아웃의 `.claude/worktrees/` 를 볼 수 없다) | **이동** → archive, 또는 det 종결 요약으로 정정. CLAUDE.md:27 의 "det 작업은 diagnosis-plan.md" 지목도 함께 |
| `experiments/monitor-log.md:20,54` | `../synthesis/` — 2026-07-08 에 철회된 볼트 미러 구조의 잔재 링크 | **정정** |
| `archive/` 에 가야 할 것 | `meta/taskboard.md` · `meta/2026-08-06-session-handover-a830ad4d.md`(종료된 세션 인계) · `det/diagnosis-plan.md` · `issues` 하단 종결 이슈 · `decisions/` 의 폐기 확정 제안(P48 `decisions/00_MOC.md:21` 이 "폐기(2026-08-06)" 로 표기하나 파일은 활성 폴더) | **이동** |

### 3-4. 같은 사실이 세 곳 이상에서 반복되는 것 (갱신 누락의 원인)

| 사실 | 등장 파일 수 / 총 횟수 | 대표 위치 |
|---|---|---|
| DELIVER 헤드라인 `56.99` | **25 / 94** | 카드 19 · issues 9 · analysis-D3 7 · fair-eval-protocol 6 · current 6 · registry 4 · glossary 4 · artifact-locations 3 · monitoring-handover 3 |
| SOTA 기준선 `56.71` | **28 / 172** | monitor-log 100 · history-H2 9 · 카드 7 · registry 7 · arch-evolution 4 · current 4 |
| MUSES `79.788` | **32 / 86** | registry 10 · history-H2 10 · arch-evolution 8 · log 7 · monitor-log 6 |
| MUSES SOTA `82.39` | **20 / 45** | monitor-log 9 · history-H2 5 · current 4 |
| DELIVER 5시드 평균 `54.39` | **12 / 30** | 카드 12 · plan 3 · current 3 |
| MCubeS `58.07` | **10 / 21** | 카드 5 · plan 3 · registry 3 · current 3 |

같은 숫자가 열 곳 넘게 흩어져 있으면 판정이 뒤집힐 때 전수 갱신이 사실상 불가능하다. ISSUE-036 이 바로 그 상황이다.

---

## ④ 세션 시작 커버리지 표

CLAUDE.md §1(Step 0 → Step 1) 을 **그대로** 따랐을 때(= bot-roles → 00_INDEX → current.md → arch-evolution → log.md → issues → novelty → servers-and-launch → environment) 무엇을 알게 되는가.

| 항목 | 어디서 알게 되나 | 몇 번째 문서 / 몇 줄째 | 판정 |
|---|---|---|---|
| **(a) 헤드라인 수치 + 규칙**(하네스·ckpt 선택·프로토콜) | `status/current.md:26-36`(legal 프로토콜·val-best top1 고정·56.99/79.29/58.07), `:65`(하네스 가드 `tools/eval_harness_guard.py --check` 필수) | 3번째 문서, 102줄 중 26~36줄 = **이르다** | 🟠 **부분** — 값은 이르게 나오지만 **ISSUE-036(오늘 그 값을 반증)이 이 경로에 없다.** ISSUE-036 은 `issues/issues-and-fixes.md:20`(§1 읽기 순서의 5번째)에는 있으나 current.md 와 모순된 채 병존한다. 두 문서를 다 읽어도 어느 쪽이 유효한지 판단 근거가 없다 |
| **(b) 진행 중 실험·대기열** | `status/current.md:67-74` "활성 런/대기" | 3번째, 67줄째 | 🔴 **틀림** — 이 절의 날짜는 "2026-09-02" 이고 내용은 P52 본런·P50-EXT 시대다. 실제 현황은 `experiments/plan.md:59-90`(2026-09-17 03:40 실측, 11런)인데 **plan.md 는 CLAUDE.md §1 읽기 목록에 없다.** 00_INDEX:27 의 experiments 행 안에 "GPU 잡기 전 필독" 으로 묻혀 있어 폴더 MOC 까지 들어가야 보인다 |
| **(c) 재시도 금지 축** | `status/current.md:89`(반증 확정 목록 1줄 — 근거로 **외부 artifact "D절"** 을 가리킨다) / `research/hypothesis-ledger.md`(H1~H22 canonical) / `research/synthesis-tried-map.md`(재제안 금지 목록) | current 89줄째 = 문서 끝. ledger·tried-map 은 **읽기 목록에 없음**(research/00_MOC.md:7,12 를 열어야 도달, 4단계) | 🟠 **늦다** — 한 줄 요약은 보이나 근거가 repo 밖 artifact 이고, 실제 원장 두 개는 진입 경로 밖이다. `research/00_MOC.md:9` 스스로 "새 모델 제안 전 필독" 이라 적지만 CLAUDE.md·00_INDEX 는 그 문서를 지목하지 않는다 |
| **(d) 서버·GPU 규칙** | `CLAUDE.md:224-228`(빈 GPU 판정 ≤2000MiB/≤10%, 런처별 자동 선택) · `infra/servers-and-launch.md` · `experiments/plan.md:50-68`(GPU 점유 원칙·예약 현황) | CLAUDE.md 228줄 중 **224~228줄 = 문서 맨 끝** | 🟠 **늦다** — 가장 자주 쓰는 안전 규칙이 CLAUDE.md 최하단. 게다가 예약 현황(`plan.md:59-68`)은 읽기 목록 밖이고, `plan.md:63,66` 은 스스로 "jarvis GPU0 예약 여부 확인 필요"·"bengio 양도 전제와 모순 — 확인 필요" 라고 미해결 상태를 적고 있다 |
| **(e) 세션 간 분담 + 메시지 규약** | `meta/bot-roles.md`(Step 0, 231줄) — 코드분석봇/코딩봇/실험분석봇/그림봇 · `CLAUDE.md:43-57`(§1.6 = 모델 위임이지 세션 분담 아님) | Step 0, 1번째 문서 | 🔴 **누락** — 실제 운용 분담(생각정리=판정 / 감시=집행 / 분석 / 재학습)은 **어디에도 정의가 없다**(진입 문서 grep 0건, `.claude_logs` 전체 101회 사용). `meta/monitoring-session-handover.md`(감시 인계 명세, 685줄)는 읽기 목록·00_INDEX 어디에도 지목이 없다. 메시지 규약(세션 간 전달은 의뢰서로)은 `meta/conventions.md:59` 에 있으나 §1.5 요약에는 빠져 있다 |
| **(f) 약어 설명 규칙** | `meta/experiment-glossary.md:4`(규칙 문장) · `meta/00_MOC.md` 첫 행(등록됨) | CLAUDE.md 0건 · 00_INDEX 0건 · meta/00_MOC 를 열어야 도달 = 4단계 | 🟠 **늦다** — 이 규칙(user 2026-09-17)이 지켜지지 않으면 보고가 반려되는데, 진입 두 문서에 한 줄도 없다. CLAUDE.md:94(노션 절)에 "실험 약어는 풀어 쓴다" 가 있으나 **노션 기록 맥락에 한정**돼 있어 일반 규칙으로 읽히지 않는다 |

**요약**: (a)는 이르지만 모순을 안고, (b)는 틀린 스냅샷을 먼저 보여주고, (c)(d)(f)는 도달하지만 3~4단계 뒤이며, (e)는 아예 없다.

---

## ⑤ 갱신 경로 점검 (규칙 vs 실제 준수)

### 5-1. 규칙이 닫혀 있는가

| 사건 | 규칙이 지정하는 갱신처 | 닫혀 있나 |
|---|---|---|
| 새 실험 launch | `experiments/registry.md` 행 추가(`CLAUDE.md:35`, `conventions.md:55`) + `experiments/plan.md` "실행 중" 표(`plan.md:44`) + 노션 §4·§6 같은 날(`CLAUDE.md:94`) | ⚠️ 두 문서(registry·plan)에 같은 행을 중복 기입해야 하고, 어느 쪽이 먼저인지 규칙에 없다 |
| 실험 판정 | 분석 문서 신설 → `experiments/00_MOC.md` 행 추가(`conventions.md:38`) → registry 상태·수치 → `status/current.md` 덮어쓰기 → `status/history-<반기>.md` 최상단 append(`CLAUDE.md:89`) → 노션 → (카드 실험이면) 카드 §5 소절 | ❌ **순서가 명시돼 있지 않고 대상이 6~7곳**이다. `research/hypothesis-ledger.md` 갱신은 `conventions.md:59` 의 "결과 기록처" 에만 나오고 CLAUDE.md §3 에는 없다 |
| 새 이슈 | `issues/issues-and-fixes.md` 상단 인덱스 표 + 본문 양쪽(`00_INDEX.md:80`) | ✅ 닫혀 있다 |
| 새 결정 | `decisions/YYYY-MM-DD-<slug>.md` 신설 + 폴더 MOC 등록(`conventions.md:32,38`) | ✅ 규칙은 닫혀 있다(준수는 아래) |
| 새 도구 | `tools/` 에 두고 `tools/README_seg_analysis.md` 매핑(`experiments/00_MOC.md`), 일회성은 `_archive/oneoff/`(`conventions.md:48`) | ⚠️ README 갱신 의무가 규칙 문장으로 명시돼 있지 않다. 예: `tools/legal_rescore_v2.py`(커밋 c214ce9)·`tools/baseline_failure/`(issues:22 에서 언급)가 어느 문서에도 매핑되지 않았다 — **미확인**(README 내용은 이번 감사에서 읽지 않았다) |

### 5-2. 최근 30일 표본의 실제 준수

이 워크트리는 2026-09-10 에 체크아웃돼 **대부분 파일의 mtime 이 09-10 으로 뭉개졌다.** 따라서 mtime 으로 "최근 편집" 을 판정할 수 있는 것은 09-10 이후에 실제로 쓰인 파일뿐이다(그 이전 편집 이력은 git 로그가 필요한데 이 세션은 git 명령 금지라 **미확인**). 09-10 이후 변경 파일 20개를 표본으로 삼았다.

| 규칙 | 준수 | 근거 |
|---|---|---|
| 새 분석 문서 → `experiments/00_MOC.md` 행 추가 | 🟡 부분 | 09-17·09-18 신설 3건(`2026-09-17-baseline-failure-analysis-plan`·`2026-09-18-baseline-failure-d3-test-findings`·`2026-09-18-muses-official-test-p39_1-seed20260825`)은 **등록됨**. 반면 08-09~09-08 신설 12건은 **전부 미등록**(§②) — 규칙은 최근에만 지켜지고 있다 |
| 새 결정 문서 → `decisions/00_MOC.md` 행 추가 | 🟡 부분 | 09-17 신설 2건 등록됨(MOC 첫 두 행). 08-18·08-21·08-31 3건 미등록 |
| 판정 변경 → `status/history-2026H2.md` 최상단 append | ✅ | `status/history-2026H2.md:12,14` 가 2026-09-18 엔트리 2건으로 최상단에 있다 |
| 판정 변경 → `status/current.md` 덮어쓰기 | 🟡 부분 | MUSES 79.29 는 반영(`current.md:36,46,52`), **ISSUE-036/DELIVER 56.99 는 미반영**(0건). 같은 날 같은 파일에서 한쪽만 갱신 |
| 노션 동기화 문구(`CLAUDE.md:94`) | ✅ 기록상 | `history-2026H2.md:12` 가 "노션 논문 페이지 §0·§3.6·§4·§6 + 차트 재생성·교체(audit 통과)" 를 적는다. 노션 실물은 이 감사에서 확인하지 않았다 — **미확인** |
| registry 행 갱신 | ✅ | DGFusion·CAFuser 기준선 행이 09-14·09-17 수치까지 반영(`registry.md:26,27`) |
| frontmatter `updated` 갱신 | ❌ | `registry.md:3`=2026-08-08, `issues-and-fixes.md:9`=2026-08-06, `arch-evolution.md:8`=2026-08-04, `00_INDEX.md:3`=2026-08-08 — 전부 본문보다 뒤처졌다 |

### 5-3. 충돌·유실의 흔적 (vault 제외 `.claude_logs` 전수 카운트)

| 표지 | 총 횟수 | 상위 파일 |
|---|---|---|
| 정정 | **126** | monitor-log 36 · 카드 24 · history-H2 12 · registry 6 |
| 무효 | **98** | registry 12 · 카드 11 · monitor-log 10 · arch-evolution 9 |
| 철회 | **78** | monitor-log 17 · arch-evolution 9 · 카드 8 · history-H2 8 · registry 7 |
| 유실·소실 | **51** | monitor-log 11 · history-H1 6 · issues 6 · arch-evolution 6 |
| 재판정 | **35** | registry 5 · arch-evolution 4 |
| 오기 | **15** | 카드 7 · arch-evolution 2 |
| 덮어씀 | **8** | issues 2 |

문서에 명시된 실제 사고 사례:
- `decisions/…-cards.md:26` — "§5-6 초판의 **분모 오기(54.19, 정답 54.69)** 가 E13·E4b 판정에 전파된 사고(09-10)" → 이후 "24클래스 값은 매번 원자료에서 재계산하고 다른 절에서 옮겨 적지 않는다" 라는 규칙이 생겼다. **같은 숫자를 여러 절에 적은 것이 원인이었다는 자기 진단이 이미 문서에 있다.**
- `registry.md:26` — "(정정 2026-09-13: 앞선 기록의 'NaN 4회' 는 같은 재개 기록이 `log.txt` 와 `dgfusion_resume_80k.log` 에 **중복돼 한 건을 두 번 센 것**)"
- `status/history-2026H2.md:20` — "9/9 **다른 세션의** '80k 정본 확정·200k 포기' 판단을 완주 재시도로 번복"
- `meta/monitoring-session-handover.md` frontmatter — "같은 이름을 다른 세션이 가질 수 있어 이름으로 지목하면 혼선이 난다(2026-09-08 **실제로 발생**: 구 p30-det 세션이 구식 로직 감시 4건을 겹쳐 걸었다가 정리됨)"
- `experiments/plan.md:63,66` — "⚠️ GPU0 예약 여부 확인 필요 … 사용자 확인 대기", "⚠️ 아래 '중단 런' 절의 양도 전제와 모순 — 확인 필요" (같은 표 안에서 두 세션의 기록이 어긋난 채 남아 있다)

### 5-4. 구조적으로 줄이는 방법 (제안)

1. **세션별 append 전용 파일.** 공유 파일을 여러 세션이 동시에 고치는 지점은 `plan.md`·`registry.md`·`current.md`·카드 §5 넷이다. 이 중 **append 성격인 것**(카드 §5 판정, plan "실행 중" 갱신)을 `experiments/inbox/<YYYY-MM-DD>-<세션>.md` 로 분리하고, 정본 파일은 **하루 한 번 한 세션이 병합**한다. 동시 편집이 사라지면 "번복·중복 계수" 사고가 구조적으로 불가능해진다.
2. **수치 정본 한 곳 + 파생 자동 생성.** 헤드라인 수치를 `experiments/headline.yaml`(벤치 · 값 · 선택 규칙 · ckpt · 하네스 버전 · 근거 문서 · 상태[유효/보류/철회]) 한 파일에 두고, `current.md` 벤치 표·`status-report.html`·노션 §0 을 스크립트로 생성한다. ISSUE-036 같은 사건에서 **상태를 "보류" 로 한 줄 바꾸면 94곳이 아니라 1곳만 고치면 된다.**
3. **숫자 복제 금지 규칙.** 카드 §0 이 이미 24클래스에 대해 도입한 "다른 절에서 옮겨 적지 않는다"(`cards:26`)를 헤드라인 수치 전반으로 확대. 분석 문서는 자기가 측정한 값만 적고, 남의 값은 **링크만** 한다.
4. **frontmatter `updated` 자동 검사.** 본문 최신 날짜 > frontmatter `updated` 이면 실패하는 검사(§⑥).

---

## ⑥ 계층 파괴 위험과 최소 규칙 제안

### 6-1. 앞으로 깨질 지점

| 위험 | 지금 상태 | 깨지는 방식 |
|---|---|---|
| **카드 문서 단일 파일 비대** | 1,671줄, §5 소절 **35개**, §5-1~§5-34 + 무번호 1개(`:396`) | 소절 번호가 **본문 순서와 무관**(등장 순 34,33,1,2,9,31,31b,32,30,…)해 "§5-29 를 보라" 는 참조가 위치 검색을 요구한다. 이미 `5-31b` 같은 알파벳 접미가 나왔고, 다음 충돌은 두 세션이 같은 번호를 동시에 쓰는 것이다. 문서를 통째로 읽지 않으면 최신 판정을 놓친다 |
| **current.md 로 결정·이력·규칙이 흘러듦** | 102줄이지만 안에 스냅샷(`:20-56`) + 규칙(`:32-35` ckpt 선택 규칙, `:65` 측정 프로토콜, `:98-102` 재현성 규약) + 이력(`:69-74` 런별 사고 경위, `:91-96` 이관 기록) + 결정(`:58-66` 제안 모델 확정)이 섞여 있다 | "덮어쓰기만" 규칙(`:8`)과 충돌한다 — 규칙·결정은 덮어쓰면 안 되는 내용이라 실제로는 아무도 덮어쓰지 못하고 **append 되다가 2026-08-08 에 한 번 폭발했다**(`current.md:9` 가 "이전에 22개 엔트리가 여기 적층돼 스냅샷 기능을 잃었던 사고" 라고 스스로 기록). 같은 일이 반복 중 |
| **markdown 표 파손** | `current.md:39-40` 에 표 헤더+구분선이 있고, `:41-49` 에 블록인용이 끼어든 뒤 `:51-56` 에 표 행이 온다 | 이미 렌더링이 깨져 있다. 사람이 손으로 표에 절을 끼워 넣다 생긴 사고이고, 자동 생성이 아닌 한 반복된다 |
| **MOC 손 유지** | 등록이 **일괄 청소로만** 일어난다(`00_INDEX.md:3` "미등록 40건 해소", `experiments/00_MOC.md:7` "analysis/ 전체 재등록(29건 미등록 상태였음)") | 6주 만에 27건 재발(§②). 다음 청소 전까지 계속 늘어난다 |
| **plan.md frontmatter** | `:3` 의 `updated:` 가 **두 달치 이력이 들어간 한 줄** | 한 줄이 계속 길어져 diff 가 불가능하고, 두 세션이 같은 줄을 고치면 반드시 충돌한다 |
| **monitor-log 6,410줄** | `conventions.md:39` 가 "100KB 넘으면 반기/월 단위 분할" 을 규정 | 규칙은 있으나 분할 여부 **미확인**(파일 크기를 바이트로 재지 않았다). 줄 수로 보면 이미 분할 대상일 가능성이 높다 |

### 6-2. 최소 규칙 제안 (넷)

**R1. 파일 크기 상한과 분할 기준**
- `.claude_logs` 의 어떤 md 도 **1,200줄을 넘기지 않는다.**
- 넘으면: 날짜 기반 로그(monitor-log·history)는 **월/반기 파일로 롤오버**(이미 conventions.md:39 에 있음, 상한을 줄 수로 구체화), 프로그램 문서(카드)는 **§5 를 `decisions/cards/2026-09-<주차>-verdicts.md` 로 주 단위 분할**하고 본문에는 "규약(§0~§4) + 최신 3개 판정 + 분할 파일 목록" 만 남긴다.
- 소절 번호는 **부여하지 않고 날짜+주제로 제목**을 단다(`### 2026-09-16 MUSES 확정 — E13M 3페어 통과`). 번호가 없으면 순서·중복 충돌이 원천적으로 사라지고, 현재 이미 번호가 순서와 어긋났으므로 손해도 없다.

**R2. current.md 는 60줄 이내, 네 블록만**
- ① 벤치별 최선 표(값 + 선택 규칙 이름 + 상태[유효/보류/철회] + 근거 링크) ② 지금 도는 것(plan.md 링크 한 줄, 표 복제 금지) ③ 열린 블로커 5개 이내 ④ 다음 판정 대기 목록.
- 규칙(ckpt 선택·측정 프로토콜·재현성 규약)은 **`meta/conventions.md` 또는 `experiments/protocol.md` 로 이관**하고 current.md 는 링크만 둔다. 이력·경위는 한 줄도 두지 않는다(history 로).

**R3. 문서 하나에 정본 하나**
- 각 문서 frontmatter에 `owns:` 필드를 두어 그 문서가 정본인 주제를 선언한다(예: `owns: [deliver-headline, muses-headline]`).
- 같은 `owns` 값이 두 문서에 있으면 검사 실패. 정본이 아닌 문서는 수치를 **인용하지 말고 링크**한다.

**R4. 자동 검사 스크립트 `tools/docs_lint.py`** (커밋 전 또는 일 1회 cron)

| 검사 | 실패 조건 |
|---|---|
| MOC 등록 | 폴더 내 `*.md` 중 해당 `00_MOC.md` 에서 링크되지 않은 파일이 있다(vault 제외) |
| 링크 무결성 | 상대 md 링크가 실재 파일로 해석되지 않는다(코드 텍스트 오탐은 백틱 안 제외로 회피) |
| 크기 상한 | 1,200줄 초과 md |
| frontmatter 신선도 | 본문에 등장하는 최대 `YYYY-MM-DD` > frontmatter `updated` |
| 표 파손 | 표 헤더/구분선 다음에 표 행이 아닌 줄(블록인용·빈 줄 아닌 본문)이 온다 |
| 정본 중복 | 같은 `owns:` 주제를 두 문서가 선언한다 |
| 수치 확산 | `experiments/headline.yaml` 에 등록된 헤드라인 값이 정본 아닌 문서에 **새로** 등장한다(경고) |
| 명명 | 신규 파일이 kebab-case(또는 `YYYY-MM-DD-` 접두)가 아니다 |
| 스텁 보호 | 루트 구번호 스텁 파일이 6줄을 넘는다(= 누가 구경로에 썼다) |

기존 도구와의 접점: 채점기 동결 검사(`tools/eval_harness_guard.py`)가 이미 "규칙을 스크립트로 강제" 하는 선례이므로, 같은 방식으로 문서 규칙을 강제하면 운영 방식이 일관된다.

---

## ⑦ 권고 조치 목록

작업량: S = 30분 이내 · M = 반나절 · L = 하루 이상

| # | 우선 | 조치 | 대상 | 량 | 판정 세션 승인 |
|---|---|---|---|---|---|
| 1 | 🔴 즉시 | **ISSUE-036 을 `status/current.md` 에 반영**한다 — DELIVER 헤드라인 56.99 의 상태를 "🟡 보류(ISSUE-036, native 재측정 55.18)" 로 바꾸고 issues·카드 §5-34 를 링크. 확정 전까지 다른 문서로 값을 복제하지 않는다 | `status/current.md:34,45,51` | S | **필요** — 수치 판정이다 |
| 2 | 🔴 즉시 | **CLAUDE.md 낡은 절 3개 처리**: §프로젝트 개요(:102-114) 정정 · §모델 버전 요약표(:201-213) 삭제 후 포인터 · §핵심 코드 구조(:166-197) 실측 트리로 교체 · §주의사항 1~4(:219-223) archive 이동 | `CLAUDE.md` | M | 불필요(사실 정정) |
| 3 | 🔴 즉시 | **CLAUDE.md §1 읽기 순서 개정**: `experiments/plan.md`·현행 카드 문서·`research/hypothesis-ledger.md`·`meta/experiment-glossary.md` 추가, `meta/taskboard.md`·`det/diagnosis-plan.md` 지목 제거. GPU 규칙(:224-228)을 §1 직후로 끌어올림 | `CLAUDE.md:18-27,224-228` · `00_INDEX.md:9-15` | S | **필요** — 진입 규칙 변경 |
| 4 | 🔴 즉시 | **세션 분담·메시지 규약 문서화.** "생각정리(판정) / 감시(집행) / 분석 / 재학습" 각각의 권한·금지·인계 방법을 한 문서에 정의하고 CLAUDE.md Step 0 에서 지목. (다른 에이전트가 쓰고 있는 `meta/analysis-session-protocol.md` 와 범위가 겹칠 수 있으니 **합쳐 쓸지 분리할지 먼저 확인할 것**) | `meta/` 신규 + `CLAUDE.md:12-15` | M | **필요** — 역할 정의다 |
| 5 | 🟠 이번 주 | **미등록 27건 MOC 등록** + `00_INDEX.md:31`(decisions 최신 = 08-08 → 09-17), `:3`(갱신일) 정정 | 3개 MOC | S | 불필요 |
| 6 | 🟠 이번 주 | **STALE 문서 archive 이동 4건**: `meta/taskboard.md` · `meta/2026-08-06-session-handover-a830ad4d.md` · `det/diagnosis-plan.md` · `decisions/2026-08-05-p48-instance-supervision-proposal.md`(폐기 확정). 각 파일 상단에 🗄 ARCHIVED 헤더(`conventions.md:35`) | 4 파일 + MOC | S | **필요** — det 진단서 폐기 여부는 판단이 섞인다 |
| 7 | 🟠 이번 주 | **frontmatter `updated` 4건 정정**(registry:3 · issues:9 · arch-evolution:8 · 00_INDEX:3) + `current.md:14,67` 의 날짜 라벨 | 5 파일 | S | 불필요 |
| 8 | 🟠 이번 주 | **깨진 링크 6건 수정**(§② 표) + `scripts/servers.conf:2` 의 `13_servers_and_launch.md` → `infra/servers-and-launch.md` | 6 파일 | S | 불필요 |
| 9 | 🟠 이번 주 | **규칙 충돌 3건 명시적 해소**: GLM.md 워커 위임 vs CLAUDE.md §1.6 / 워크트리 git 제약 conventions 반영 / lecun 배치 금지를 `servers.conf` 정책(`off`)과 plan.md 사유에 반영 | `CLAUDE.md:43` · `conventions.md:14,57` · `servers.conf:23` | S | **필요** — 전역 규칙과의 우선순위 결정 |
| 10 | 🟠 이번 주 | **`current.md:39-56` 깨진 표 복구** — 블록인용을 표 앞/뒤로 빼고 표를 한 덩어리로 | `status/current.md` | S | 불필요 |
| 11 | 🟡 2주 | **current.md 재설계(R2)** — 60줄 4블록으로 축소, 규칙은 `experiments/protocol.md` 신설로 이관, 이력·경위는 history 로 | `status/current.md` + 신규 1 | M | **필요** — 단일 출처 구조 변경 |
| 12 | 🟡 2주 | **카드 문서 분할(R1)** — §5 를 주 단위 판정 파일로 쪼개고 번호 대신 날짜+주제 제목. 기존 "§5-NN" 참조는 분할 파일 안에 앵커로 보존 | `decisions/…-cards.md`(1,671줄) | L | **필요** — 기존 인용이 전부 깨진다 |
| 13 | 🟡 2주 | **`experiments/headline.yaml` 정본화 + 파생 자동 생성**(R2·5-4 ②) — current 벤치 표·status-report.html·노션 §0 을 생성물로 | 신규 + 빌더 | L | **필요** |
| 14 | 🟡 2주 | **`tools/docs_lint.py` 작성**(R4 9종 검사) + 일 1회 실행 | 신규 | M | 불필요(도구) |
| 15 | 🟡 2주 | **registry 셀 다이어트** — DGFusion·CAFuser 행의 서사를 `experiments/analysis/` 문서로 옮기고 셀은 결론 1행 + 링크. P31·SAM3-RBMA 행의 "학습 중" 상태 정정 | `experiments/registry.md:26,27,28,39` | M | **필요** — 상태 판정 |
| 16 | 🟡 2주 | **`plan.md:3` frontmatter 이력 분리** — `updated:` 는 날짜만, 변경 이력은 본문 하단 절로 | `experiments/plan.md` | S | 불필요 |
| 17 | 🟡 2주 | **`research_vault/architecture/P34_ReliaDINO_design_20260712.md` 처리**(6주 미결) — NAS 볼트 대조 후 이동 또는 삭제 | 1 파일 | S | **필요** — 볼트 규약 |
| 18 | ⚪ 추후 | **monitor-log(6,410줄)·arch-evolution(3,610줄) 롤오버/보강** — monitor-log 는 월 단위 분할, arch-evolution 은 P48~P53 추가 또는 "P48 이후는 decisions/ 가 정본" 명시 | 2 파일 | L | **필요** — 정본 범위 결정 |
| 19 | ⚪ 추후 | **issues 하단 ISSUE-001~017 종결 판정** — 1년 가까이 "진행/예정" 으로 고정된 항목을 해결/폐기/이관 중 하나로 | `issues/issues-and-fixes.md:40-55` | M | **필요** |
| 20 | ⚪ 추후 | **`meta/bot-roles.md` 재작성 또는 폐기** — 현 세션 분담 어휘와 다르고 경로가 전부 구번호. #4 와 합칠지 결정 | `meta/bot-roles.md` | M | **필요** |

### 미확인으로 남긴 것

- **git 이력 기반 준수 검사**: 이 워크트리는 2026-09-10 체크아웃이라 대부분 파일의 mtime 이 09-10 으로 뭉개졌고, 감사 지시가 git 명령을 금지해 커밋 단위 준수 여부(누가 언제 registry 행을 빠뜨렸는지)는 확인하지 못했다.
- **노션 논문 페이지 실물**: `history-2026H2.md:12` 의 동기화 기록이 사실인지 노션에서 대조하지 않았다.
- **`.claude/worktrees/` 실재**: 이 세션은 워크트리 내부라 메인 체크아웃의 워크트리 목록을 볼 수 없다. `det/diagnosis-plan.md:11` 이 가리키는 `p29-det`·`p30-det` 워크트리의 생존 여부는 확인 불가.
- **`monitor-log.md` 바이트 크기**: `conventions.md:39` 의 100KB 롤오버 기준 충족 여부를 바이트로 재지 않았다(줄 수만 확인).
- **`tools/README_seg_analysis.md` 내용**: 새 도구(`legal_rescore_v2.py`·`baseline_failure/`)가 그 매핑표에 등재됐는지 읽지 않았다.
- **`research/vault/` 내부 정합성**: NAS 원본과의 동기화 상태는 대조하지 않았다(손편집 금지 대상이라 감사 범위 밖으로 두었다).
