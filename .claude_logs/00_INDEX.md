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
| [18_research_digest.md](18_research_digest.md) | **옵시디언 리서치 볼트 다이제스트(2026-07-02 동기화)** — 벤치마크 정량표([val]/[test] 태그), 경쟁자 메커니즘 taxonomy, P29/P30 구현 참고, 2026 위협 워치, 논문 문단 후보, 아이디어 회의 어젠다 |
| [research_vault/](research_vault/) | 옵시디언 볼트 원본 노트 사본(94개, NAS 불필요) — 논문별 synthesis 노트. 벤치마크 숫자 canonical = `research_vault/relatedworks/09_benchmark_tables_deliver_muses_mcubes.md` |
| [10_related_work.md](10_related_work.md) | deep-research 원시 로그(시계열). 12의 근거 |
| [11_sam3_rbma_plan.md](11_sam3_rbma_plan.md) | SAM3 포팅 플랜 & 체크리스트 |

### 4. 🧪 실험 로그 (Experiments)
| 문서 | 내용 |
|------|------|
| [03_experiment_log.md](03_experiment_log.md) | **메인** — 전체 결과 M-score 표 + 버전별 상세 + 진단. **실험 canonical** |
| [15_training_monitor_log.md](15_training_monitor_log.md) | **진행 중 학습 실시간 모니터 로그** — `/loop` 세션이 주기적으로 append. 학습 추세 분석/판단은 여기서. (현재 RUN-1=B200 P28 DELIVER) |
| [16_failure_analysis_P28_P29.md](16_failure_analysis_P28_P29.md) | **P28(RBMA)·P29(SDC) 체계적 실패분석 + P30 커버리지 판정(✅/🟡/❌) + P31 프로토타입(high-res class-token decoder)**. DELIVER 로그 파싱 기반 |
| [19_det_diagnosis_plan.md](19_det_diagnosis_plan.md) | **P29/P30-Det 성능 진단 계획(2026-07-02)** — mAP50 0.45/0.25 vs 목표 0.85 원인 분석. 핵심: P30 하락은 router↔query-head confound(단일 s16 query decoder가 소물체 병목), 외부 baseline 전무. Phase 0(무학습 ckpt 분석)→1(YOLO 기준점)→2(통제 ablation)→3(breakthrough) 실험 체크리스트 + 결과 기록표. **det 작업 전 필독** |
| [21_egofill_dataset.md](21_egofill_dataset.md) | **lidar egofill 데이터셋 v20260703_egofill(2026-07-03)** — RGB15Hz/LiDAR10Hz 주기차 결손(라벨프레임 37%)을 ego-motion 보정 최근접 스캔 재투영으로 fill (6,403장, 커버리지 98.6%). 역공학 확정값(pcd=카메라좌표, RealSense 공장 K, 실측 extrinsic — calib yaml 믿지 말 것) + train 2.01배 확장 + P29-Det 재학습(bengio) 실험. 파이프라인=sensors/.../scripts/egofill/ |
| [20_p31_design_proposal.md](20_p31_design_proposal.md) | **P31 재설계 제안(2026-07-02, proposal 상태)** — research_vault 전수 매핑 기반. Seg core: reliability 재보정(event/LiDAR AUROC .30/.22 수리)+consistency 2차 bias+multi-scale HR class-token decoder+complementary assignment / 학습 레버: backbone unfreeze·RGB-zero·타깃 증강 / Det 분리 트랙: COCO-pretrained Deformable-DETR head 이식+RBMA-in-head+데이터 복원(5,862→13,712). 우선순위·ablation·novelty 방어 포함 |
| [23_seg_arch_proposals_P32.md](23_seg_arch_proposals_P32.md) | **P32 아키텍처 제안 5종(2026-07-05, proposal 상태)** — 라우팅 실패 4원인(R1 게이트입력에 조건부재 / R2 상수가 loss 지름길 / R3 self-entropy 신호 붕괴 AUROC .30/.22 / R4 soft가중은 select 불가) 분해 후 각각 직격. **B=CoRB**(reliability를 self-entropy→cross-modal 상호검증으로, training-free 유지, `_compute_bias_source`만 override, 무학습 사전검증 가능) · **C=PruneMem**(memory token hard-pruning+modality dropout+null token) · **A=PhysCond**(증강파라미터θ 자가지도 조건인코더) · **D=ProtoTable**(무학습 클러스터→per-cluster 라우팅 테이블, 게이트함수 제거) · **E=CCR**(condition-contrastive 정칙화). §0.5=코드 seam 검증(직접 확인), §7=Phase 0 무학습 진단 로드맵, §8=lit-check TODO. **라우팅 재설계 논의 전 필독** |
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
