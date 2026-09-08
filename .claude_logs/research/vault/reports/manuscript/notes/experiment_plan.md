# EXPERIMENT PLAN — 논문 빈칸 채우기 실행 계획 (다른 Claude 세션 실행용)

작성 2026-07-15 ~15:40 KST. 근거: `notes/fact_experiments.md`(갭 목록 g-1~14), repo `.claude_logs/experiments/monitor-log.md` (RUN-22 P35 / RUN-23 P36 / RUN-24 MUSES), vault `architecture/P35_design_20260713.md` (G0a–e, T1, T2).

**🔴 최상위 제약: B200은 오늘 2026-07-15 23:59 KST까지만 사용 가능.** 이후 모든 B200 산출물(ckpt/log/config) 접근 불가 → 항목 1이 다른 모든 것에 우선한다.

**네이밍 규칙 (논문 텍스트)**: P36 = "ReliaDINO (full)", P34 = "ReliaDINO (no router)", P35 = stripped-recipe ablation 행. 내부 코드 P3x는 논문에 절대 노출 금지. 이 파일(내부 노트)에서는 P3x 사용 OK.

**표/그림 슬롯 명명**: Tab-MAIN = DELIVER test-CLDE 비교표(+val dual column) · Tab-ABL = ablation 표(라우터/모듈/백본probe/LoRA/PhysAug) · Tab-FAIR = 감독신호/공정성 표 · Tab-PERCLASS = per-class×condition test 감사 표 · Tab-MUSES = MUSES 표(채워지면) · Fig-ARCH/Fig-QUAL/Fig-AUROC/Fig-LINEAGE = figures/figure_plan.md 기준 (항목 8 참조).

---

## 한눈표 (우선순위순)

| # | 항목 | 우선순위 | 데드라인 | owner 서버 | config/도구 | 채우는 슬롯 |
|---|---|---|---|---|---|---|
| 1 | P36 완주 수치 확정 + B200 전량 회수 | **P0 (오늘, 즉시)** | **07-15 23:59 KST** | B200 (ssh port 39701) | `configs/b200-deliver_rgbdel_P36_router.yaml` | Tab-MAIN full행 · Tab-ABL 라우터행 · 본문 모든 P36 \todo |
| 2 | T1: PhysAug-off no-router 완주 재학습 | P1 | 제출 전 (~1.5일 학습) | hinton 또는 jarvis (아래 판단표) | P35 yaml 이식 → `configs/deliver/` 신규 | Tab-ABL PhysAug행 · (성공 시) headline 교체 후보 |
| 3 | MUSES: 공식 프로토콜 재평가 + 제출 | P1 | 서버 제출 리드타임 감안 최우선순위 내 | 회수 후 lecun/levine (데이터 NFS 마운트됨) | `configs/b200-muses_rgbel_P34_reliadino.yaml` (회수본) | Tab-MUSES · 결론 future-work → 본문 승격 |
| 4 | MULTIAQUA ReliaDINO 이식 | P2 | 리비전 대비 | levine/bengio | 신규 `configs/multiaqua/` (P34 레시피, [img,thermal,lidar]) | 일반화 절/표 (조건-shift showcase) |
| 5 | 멀티시드 (headline config ×2 시드) | P2 | 리비전 대비 | T1과 동일 서버 | T1 config + SEED만 변경 | Tab-MAIN 각주 mean±std · sub-1pt 주장 해금 |
| 6 | G0d: eval resize-parity 체크 | **P1 (차단급, eval-only)** | Tab-MAIN 확정 전 | 로컬(ailab_mat2 DELIVER) 또는 hinton | `tools/eval_reliadino_ckpt.py` | Tab-MAIN 각주 (프로토콜 패리티) — +0.89/+0.43 마진 주장의 전제 |
| 7 | 정성 그림 + per-domain 산출물 | P1 | 그림 마감 전 | hinton (ckpt-only inference 용도 검증됨) | `tools/viz_features.py` · `tools/eval_per_domain.py` · `tools/analyze_per_domain.py` | Fig-QUAL · Fig-AUROC · Tab-PERCLASS 검증 |
| 8 | figure_plan.md별 asset 일괄 생성 | P1 | 그림 마감 전 | 항목 7과 동일 | `figures/figure_plan.md` (⚠ 작성 대기) | 전체 Fig-* |

---

## 항목 1 — P36 완주 수치 + B200 최종 회수 (P0, 오늘)

**상황**: RUN-23(P36)은 ep112/200 시점(07-15 00:20 KST) 이후 monitor-log 엔트리 없음. 완주 예상 ~12:20 KST — **이 계획 작성 시각 기준 이미 완주했을 것**. val은 ep76 이후 62–63 밴드 열화 고착이라 best 갱신 가능성 낮음(사실상 val 67.74@ep52 / test 57.14@ep58 잠김)이나, **확정은 로그로만** 한다.

절차 (B200 접속: `ssh -p 39701` + key `~/.ssh/b200_new_key`; **B200 시계 = UTC(−9h)**, mtime 말고 로그 내용 timestamp 신뢰):
1. `logs/p36_20260714_012614.log` tail → ep200 완주 확인 + 전 구간 best(val-best ep / test-best ep) 추출.
2. **합법 선정 프로토콜 적용**: val-best ckpt 하나를 고르고 그 ckpt의 test 수치를 기록 (P34 선례: ep120 val 68.20/test 56.64). P36 val-best가 ep52면 legal pair = (67.74, 57.14 — 같은 스케줄에서 측정됨, fact_experiments (f)).
3. `outputs/ReliaDINO/b200_deliver_rgbdel_P36_router/`에서 **val-best top-k + test-best top-k ckpt + train.log + config yaml** → `rsync`로 `/nas_jm/drone_ckpts/B200_backup_20260715/`에 재동기화 (기존 스냅샷 ep52/ep58은 진행중 백업 — 최종본으로 갱신).
4. RUN-24(MUSES, ~08:05 완주 예상) 산출물도 동시 회수: `outputs/ReliaDINO/b200_muses_rgbel_P34_reliadino/` ckpt + `logs/muses_P34_reliadino_20260714_155005.log` + `configs/b200-muses_rgbel_P34_reliadino.yaml`.
5. **문서 갭 마감 (회수한 파일에서 읽기만 하면 됨)**:
   - g-6: P36 yaml에서 PhysAug OFF·val-selection 설정 실물 확인.
   - fact_method todo-3: P36 train.log 런치 배너에서 exact total/trainable param 수.
   - g-12: repo `experiments/log.md` + `experiments/registry.md`에 P34/P35/P36/MUSES 행 추가 (현재 P33.1이 마지막 — 논문 provenance가 monitor-log에만 의존 중).
6. 갱신 반영: `notes/fact_experiments.md` §(0)·(a)·(b-1) 수치 교체 → 각 섹션 .tex의 P36 \todo 치환.

**채우는 슬롯**: Tab-MAIN "ReliaDINO (full)" 행(현재 \todo), Tab-ABL b-1 라우터 on/off 행, 결론·초록·인트로의 headline 문장.

**⚠ 서사 주의**: P36이 P34 대비 val −0.45/test −0.46 열위 확정 시, "router = mid-training 가속(test-SOTA를 ep58 vs ep116에 도달) + 최종은 게이트/라우터 붕괴로 미달"이 정직한 프레임. P34 vs P36은 순수 router toggle이 아님(레시피 confound — 순수쌍은 P35 vs P36, 단 P35는 ep120 사망). Tab-ABL 캡션에 각주 필수.

## 항목 2 — T1: PhysAug-off no-router 완주 재학습 (P1)

**왜**: DGFusion은 표준 aug만 사용 → PhysAug-on P34를 headline로 쓰면 UNFAIR-OURS (P35_design §1). 깨끗한 "no-router + PhysAug-off + val-only selection" **완주** run이 존재하지 않음 (P35가 그 역할이었으나 B200 리부트로 ep120 사망, best 67.61/56.14).

**서버 판단표** (B200 소멸 후 대안):
| 후보 | 장점 | 리스크 | 판정 |
|---|---|---|---|
| **hinton** | DELIVER 로컬(/SSDd 또는 /SSDb), 07-06부터 ssh 재개통 | repo가 non-git(`/home/jemo_maeng/src/drone-MemorySAM`) → git adoption/코드 동기화 선행 필요; GPU 규모/가용성 세션에서 실사 필요 | **1순위** |
| jarvis | det 학습 실적 있음 | **DELIVER가 sshfs 마운트 — hang(D-state) 전력** → 학습 불가급. 로컬 디스크로 DELIVER(~수백GB) 선복사 시에만 가능 | 2순위 (복사 여유 있을 때만) |
| lecun/levine/yeon | ailab_mat2 NFS 마운트 | GPU 수/속도로 200ep 소요 시간 김; `PYTHONPATH` 이슈는 ReliaDINO 무관(SAM2 의존 없음) | 3순위 |

절차:
1. 회수된 `configs/b200-deliver_rgbdel_P35_paper.yaml`(B200_backup_20260715/configs/)을 기준으로 신규 config 작성 — conventions에 맞춰 `configs/deliver/<server>-무접두 논쟁은 회피하고 현행 명명 유지` → 실제로는 `configs/deliver/deliver_rgbdel_T1_physoff.yaml` 권장 (서버접두어 금지 규칙 준수).
2. 동결 레시피 그대로: GATE/VETO/CALIBRATION 유지, ATTN_BIAS·CONSISTENCY 제거, PhysAug OFF, val-only selection, LORA_NORM_CAP off, 200ep, seed 3407 (P35_design §6 T1 동결, commit 059b1b2).
3. 학습 전 CLAUDE.md GPU 규칙 준수(빈 GPU 확인, `scripts/remote_exp.sh status <server>` 또는 hinton 수동 확인).
4. 완주 후 val-only 선정 → val/test 쌍 보고. 성공 기준: test ≥ 56.71 & val ≥ 66.51. 실패 시 fallback = P34(PhysAug-on)를 "our best, aug 정직 공개" + PhysAug를 Tab-ABL 행으로.

**채우는 슬롯**: Tab-ABL PhysAug on/off 행 (P34-on vs T1-off); 성공 시 Tab-MAIN no-router 행을 T1 수치로 교체(공정성 서사 강화). 학습 ~8min/ep×200 기준 B200에서 ~27h였음 → 대체 서버에서 1.5–3일 예상.

## 항목 3 — MUSES 공식 프로토콜 평가 + 테스트 서버 제출 (P1)

**상황**: RUN-24 (P34 레시피, [img,lidar,event], 300ep, B200 GPU 0,1,6,7)가 07-15 00:50 KST 기동, ~08:05 KST 완주 예상. 내부 letterbox-1024 val 74.24@ep10은 **공식 프로토콜 아님** — 리더보드(DGFusion 79.5 등)와 비교 금지. 로더 `semseg/datasets/muses.py` + config + 투영 파이프라인은 **develop에 병합 완료**(`8d8f4b0..b4d69c1`) — 코드 소실 위험 없음. 데이터는 NFS `/ailab_mat2/dataset/MUSES`(lecun/bengio/levine/yeon 마운트)와 NAS zip 원본 보존.

절차:
1. (항목 1-4에서) B200 MUSES ckpt/log 회수 완료 확인.
2. **원해상도(1080×1920) 공식 프로토콜 재평가** 구현: letterbox 1024 학습 → eval 시 원해상도 복원 규약을 CAFuser/DGFusion eval 코드 기준으로 정합 (val 250장, GT 공개). 서버: lecun 또는 levine.
3. val 수치가 리더보드권(≥75)이면 **test 서버 제출**(test 750장 GT 비공개; semantic track). 제출 절차/계정은 muses.ethz.ch 벤치마크 페이지 — \todo 세션에서 확인.
4. **g-7 해소**: "MUSES SOTA 79.72/79.49"(memory) vs BENCH DGFusion 79.5 — 출처 원문 대조 후 fact_experiments에 확정 기재. 논문에는 확정 전 인용 금지.
5. 300ep 완주본이 부족하면 MUSES 재학습은 lecun/levine에서 (데이터 NFS, 코드 develop에 있음).

**채우는 슬롯**: Tab-MUSES (신규, 2번째 벤치마크) — 성사 시 결론의 future-work에서 본문 결과로 승격, C1/C3 일반화 증거. 실패/미제출 시 결론 문장 유지.

## 항목 4 — MULTIAQUA (P2)

- ReliaDINO run 전무. CMNeXt-DH 기준선 93.58 val-day / 74.25 test-night (BENCH:392–394) — 조건-shift showcase 용.
- 절차: DELIVER config를 모달 `[img, thermal, lidar]`·클래스 4로 이식한 신규 config (`configs/multiaqua/`), levine 또는 bengio에서 학습 (MULTIAQUA 경로 `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night`). MACVi 서버 제출은 `val_multiaqua.py --macvi` 계열 파이프라인 참고 (단 ReliaDINO는 trainer/eval 별도 — `train_reliadino.py` 계열로 이식 필요, MUSES 이식이 선례).
- **채우는 슬롯**: 일반화 절 한 단락 + 소형 표 (리비전 카드). 결론 future-work 문장의 실증.

## 항목 5 — 멀티시드 (P2)

- 현재 전부 seed 3407 단일시드 → sub-1pt 주장 금지 정책이 걸려 있음 (Tab-ABL 캡션에도 명기).
- 절차: 항목 2의 T1 config에서 `SEED`만 2개 추가(예: 42, 1234)로 2회 재학습, 같은 서버. headline 구성 mean±std 산출.
- **채우는 슬롯**: Tab-MAIN 각주(±std) + limitations 문장 완화. 학습 예산이 없으면 리비전 대응 카드로 보류.

## 항목 6 — G0d: eval resize-parity 체크 (P1, 차단급, eval-only)

- **왜 차단급**: DELIVER 1042×1042 → 1024 리사이즈 규약(NEAREST 등) 차이로 내부 요동 ~2pt — headline 마진(+0.89 test-best / +0.43 P36)보다 큼 (ARCH:76, g-9). DGFusion은 CMNEXT_EQUIVALENT_EVAL 사용.
- 절차: P34 ep120 ckpt(NAS `/nas_jm/drone_ckpts/P34_final_20260713/`)로 `tools/eval_reliadino_ckpt.py`를 (a) 우리 학습-eval 미러 규약 (b) CMNeXt 공식 eval 규약 두 가지로 test split 평가, Δ 기록. GPU 1장·수 시간, DELIVER 로컬 보유 서버(로컬 박스 ailab_mat2 또는 hinton) 어디서나.
- **채우는 슬롯**: Tab-MAIN 캡션/실험셋업의 프로토콜 문장. Δ가 마진보다 크면 head-to-head 델타 서술을 전면 완화해야 함 → **Tab-MAIN 문구 확정 전 필수**.

## 항목 7 — 정성 그림 + per-domain 산출물 (P1)

- 도구 (repo `tools/`, 모델 무관 — 코드 새로 짜지 말 것, `tools/README_seg_analysis.md` 매핑표 참조):
  - `tools/eval_per_domain.py` → per-condition 러너 (cloud/fog/night/rain/sun)
  - `tools/analyze_per_domain.py` → per-class 분류표 (Tab-PERCLASS 원천 재검증)
  - `tools/viz_features.py` → feature/게이트/라우터 패널 (Fig-QUAL, Fig-AUROC 보조)
- 대상 ckpt: **P36 최종 val-best**(항목 1 회수본) + P34 ep120/ep140 (기존 산출물 `/drone_nas/drone/analysis_logs/P34_eval_20260713/` 재사용 가능 — 새로 돌리기 전 확인).
- 서버: hinton (ckpt-only inference 용도로 검증된 사용처; DELIVER 로컬). 산출물 회수처: **NAS `/drone_nas/drone/analysis_logs/<model>_eval_<YYYYMMDD>/`** (HDD2는 ISSUE-023 쓰기 불가 재발 이력 — NAS가 현행 canonical).
- P36 라우터 전용 추가 패널: per-class 라우터 가중치 w^r 히트맵 (night RoadLine img/depth 역선택 교정 스토리 — fact_method §6 동기 서사의 시각 증거).
- **채우는 슬롯**: Fig-QUAL(조건별 정성 비교), Fig-AUROC(4모달 균형 [.85,.78,.87,.70] 세대 대비), Tab-PERCLASS 수치 재검증, 라우터 가중치 Fig(신규 후보).

## 항목 8 — 그림별 asset 생성 (figure_plan.md 매핑) (P1)

- **⚠ 의존성**: `figures/figure_plan.md`가 아직 없음 (figures/ 폴더 빈 상태, 담당 에이전트 작성 대기). 파일이 생기면 그 표의 Fig ID ↔ asset 경로 매핑을 따르고, 이 계획의 잠정 목록으로 선작업 가능:
  - Fig-ARCH: 아키텍처 다이어그램 — 데이터 흐름은 fact_method §0 ASCII가 원천. 수작업 드로잉 (실험 불필요).
  - Fig-QUAL: 항목 7 산출물에서 조건×모달 패널 선별 (night/fog 위주, GT/CMNeXt급 baseline/ours).
  - Fig-AUROC: `AN34/module_diag.json`(기존) + P36 동등물(항목 7) → 세대 대비 bar/radar.
  - Fig-LINEAGE: 백본 계보 (SAM2 4세대 +0.7 vs 백본 스왑 +4.1; BBR 데이터) — 기존 수치만으로 작도 가능, 실험 불필요.
  - 학습곡선 Fig(후보): monitor-log 마일스톤 or train.log 파싱 (P34 vs P36 val/test 곡선 — router 가속 서사).
- 모든 asset은 `_paper_submission/figures/`에 저장 (파일 규칙 준수 — 다른 vault 폴더 수정 금지).
- **채우는 슬롯**: 전체 Fig-*. figure_plan.md 확정 후 이 항목을 그 파일 기준으로 재정렬할 것.

---

## 실행 순서 요약 (오늘 → 제출)

1. **[지금 즉시] 항목 1** — B200 소멸 전 회수. 다른 모든 것보다 우선.
2. **[오늘~내일] 항목 6 (eval-only, 반나절)** + **항목 2 launch** (T1 재학습, 서버 확보되는 대로) + **항목 3-2 (MUSES 공식 재평가)**.
3. **[T1/MUSES 진행 중] 항목 7·8** — 그림 asset (기존 P34 산출물로 대부분 선작업 가능).
4. **[리비전 카드] 항목 4·5** — MULTIAQUA·멀티시드는 학습 자원 여유 시.

각 항목 완료 시: `notes/fact_experiments.md` 수치 갱신 → 해당 .tex \todo 치환 → repo `experiments/registry.md`/`log.md` 행 추가 (g-12).
