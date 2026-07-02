# 📁 .claude_logs 인덱스 (Master Index)

> 최종 업데이트: 2026-06-24
> 이 폴더의 **front door**. 모든 문서를 6개 카테고리로 분류한다. 새 세션은 `CLAUDE.md` → **이 인덱스** → 작업 카테고리 순으로 진입.
> (번호는 역사적 생성순이라 카테고리와 1:1이 아니다. **카테고리는 아래 표를 따른다.**)

---

## 🧭 새 세션 권장 읽기 순서

1. `CLAUDE.md` — 세션 규칙 + 프로젝트 개요
2. **이 인덱스(00)** — 어디에 뭐가 있는지
3. [01_project_status.md](01_project_status.md) **상단 "📌 현재 상태 스냅샷"만** — 지금 무엇을 하는 중인지
4. 작업 카테고리 문서로 이동 (아래)

---

## 📚 카테고리별 문서

### 1. 🎯 프로젝트 설명 (Project)
| 문서 | 내용 |
|------|------|
| `../CLAUDE.md` (프로젝트 개요) | 목표(MACVi MULTIAQUA), 핵심 아이디어, 데이터셋/클래스, 평가지표(M-score) — **canonical** |
| [01_project_status.md](01_project_status.md) | **상단 스냅샷 = 현재 상태 단일 출처**, 하단 = 역시간순 진행 로그(history) |

### 2. 🏗 아키텍처 (Architecture)
| 문서 | 내용 |
|------|------|
| [02_model_arch.md](02_model_arch.md) | P8~P28 + SAM3-RBMA 모델 상세(forward/모듈/한계/결과). **아키텍처 본문 canonical** |
| [08_architecture_figures.md](08_architecture_figures.md) | 논문/발표용 ASCII 피규어. ⚠️ **P26까지만** (P27/P28/SAM3-RBMA 미작성) |

### 3. 📚 Related Works (논문용)
| 문서 | 내용 |
|------|------|
| [12_novelty_and_related_work.md](12_novelty_and_related_work.md) | **canonical** — RBMA 노벨티 포지셔닝, 선행연구 vs RBMA 차별표, 리뷰 방어. **먼저 읽기** |
| [10_related_work.md](10_related_work.md) | deep-research 원시 로그(시계열). 12의 근거 |
| [11_sam3_rbma_plan.md](11_sam3_rbma_plan.md) | SAM3 포팅 플랜 & 체크리스트 |

### 4. 🧪 실험 로그 (Experiments)
| 문서 | 내용 |
|------|------|
| [03_experiment_log.md](03_experiment_log.md) | **메인** — 전체 결과 M-score 표 + 버전별 상세 + 진단. **실험 canonical** |
| [15_training_monitor_log.md](15_training_monitor_log.md) | **진행 중 학습 실시간 모니터 로그** — `/loop` 세션이 주기적으로 append. 학습 추세 분석/판단은 여기서. (현재 RUN-1=B200 P28 DELIVER) |
| [16_failure_analysis_P28_P29.md](16_failure_analysis_P28_P29.md) | **P28(RBMA)·P29(SDC) 체계적 실패분석 + P30 커버리지 판정(✅/🟡/❌) + P31 프로토타입(high-res class-token decoder)**. DELIVER 로그 파싱 기반 |
| [20_p31_design_proposal.md](20_p31_design_proposal.md) | **P31 재설계 제안(2026-07-02)** — research_vault 전수 매핑 기반. Seg core(A 재보정/B consistency/C MS-HR decoder/D complementary) + 학습 레버 + Det 분리 트랙. **P31 구현(`LoRA_Sam_P31`, `feat/p31-seg`)의 설계 근거** |
| [05_result_analysis_P9_P12.md](05_result_analysis_P9_P12.md) | 🗄 ARCHIVED — P9~P12 심층 분석(2026-02 동결) |
| [06_result_analysis_P13.md](06_result_analysis_P13.md) | 🗄 ARCHIVED — P13 심층 분석 |
| [07_result_analysis_P14.md](07_result_analysis_P14.md) | 🗄 ARCHIVED — P14 분석 |

### 5. ⚙️ 환경 · 인프라 (Env/Infra)
| 문서 | 내용 |
|------|------|
| [13_servers_and_launch.md](13_servers_and_launch.md) | **원격 서버 레지스트리 & 실험 자동 실행 매뉴얼**. "X를 <서버>에서 돌려줘" 지시 시 **먼저 읽기**. 단일출처=`scripts/servers.conf`, 실행=`scripts/remote_exp.sh` |
| [14_environment_and_infra.md](14_environment_and_infra.md) | 실행 환경/명령, 데이터·가중치 경로, 체크포인트 포맷, DDP, B200 튜닝, VRAM 프로브 |
| `../CLAUDE.md` (환경 설정) | conda/명령어 canonical |

### 6. 🐞 이슈 핸들링 (Issues)
| 문서 | 내용 |
|------|------|
| [04_issues_and_fixes.md](04_issues_and_fixes.md) | **상단 "이슈 상태 인덱스 표" 먼저** → 액션 필요 이슈 식별. 코딩 전 필독 |

### 🤖 메타 (Meta)
| 문서 | 내용 |
|------|------|
| [09_bot_roles_guide.md](09_bot_roles_guide.md) | 세션 역할(코드분석봇/코딩봇/실험분석봇/그림봇) 지침 |
| [P13_design_guide.md](P13_design_guide.md) | 🗄 ARCHIVED — P13 설계 시점 가이드 |

---

## 🗂 유지보수 규칙

- **현재 상태**는 `01` 상단 스냅샷 한 곳만 갱신 (history는 append).
- **새 이슈**는 `04` 상단 인덱스 표 + 본문 양쪽 갱신.
- **새 선행연구/노벨티**는 `12`(canonical) 먼저, 원시 로그는 `10`.
- **환경/인프라 변경**(GPU·경로·파이프라인)은 `13`에 기록.
- 새 문서 추가 시 **이 인덱스의 카테고리 표에 등록**.
