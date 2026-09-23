---
created: 2026-09-23
updated: 2026-09-23 15:30 KST
owns: [zcode-handoff-2026-09-23]
status: 인수인계 완료 — 클로드 코드 세션이 이 문서부터 읽고 이어갈 것
author: ZCode 코딩 세션 (2026-09-22 14:55 ~ 09-23 15:30)
---

# 🤝 ZCode → Claude Code 인수인계 (2026-09-23 15:30 KST)

> **읽는 순서**: 이 문서 → [current.md](current.md) ③④ → [../experiments/plan.md](../experiments/plan.md) GPU 표·실행 중 표. 이 문서 하나로 09-22~23 사고·재개·결과·남은 일 전부 파악된다.
> ⚠️ 생각정리 판정(09-23): "DGFusion 초과" 문구 철회 — 같은 시드의 R1 이 두 사본(jarvis test 56.58 / hpca100 56.80)으로 존재하고 유리한 쪽만 고를 수 없음, 얇은4 56.39·50.81 은 TrafficSign 을 넣은 비정본 정의(정본 = Pole·Pedestrian·Static·TrafficLight). 판정은 3시드 후([judgment-ledger 09-23 행](../experiments/judgment-ledger.md))
>
> **한 줄 요약**: yeon 리부트와 hpca100 종료 세션을 전부 재개했고, **R1(경계 prior)이 legal v2 test 56.80으로 40ep 스크린 단계에서 DGFusion(56.71)을 넘었다. R2도 짝 대비 +0.62. QAF(Q2/Q3)는 오늘 밤~내일 새벽 완주.**

## 1. 확정 결과 (판정 재료 — 15:30 기준 전부 실측)

### R1 vs R2 legal v2 재채점 (40ep 스크린, 시드 821, 짝 = E1 40ep 스크린 v2 **test 55.94 / val 68.56**)

| 카드 | test(25클) | val | 24클래스 | 얇은4 | Δtest vs 짝 | Δval vs 짝 | DGFusion 56.71 대비 |
|---|---|---|---|---|---|---|---|
| **R1** (BOUNDARY_REFINE) | **56.80** | **69.09** | 56.65 | **56.39** | **+0.86** | **+0.53** | **+0.09 초과** |
| **R2** (COMPONENT 손실) | **56.56** | **68.88** | 56.56 계산: **56.46** | **50.81** | **+0.62** | **+0.32** | −0.15 미달(근접) |

- 얇은4 내역 — R1: Ped 76.28·Pole 51.81·Sign 50.94·**Light 46.54** / R2: Ped 77.7·Pole 48.41·Sign 50.0·**Light 27.11**
- **읽기**: R1이 전 축(전체·24클·얇은4·val)에서 R2 우위. R2는 전체는 짝을 넘지만 **얇은 축(특히 TrafficLight)에서 R1에 크게 밀림** — "손실 재가중"만으로 윤곽 묘사 결함을 못 막는다는 방증. 두 카드 모두 짝 대비 양수라 결합(Q3+R) 근거는 살아 있음.
- ⚠️ **판정 잔여 재료**: ① 짝(E1 스크린 s821)의 24클래스·얇은4 집계 — 아직 안 됨(덤프/재채점 로그에서 가능) ② **"찾고도 못 그린 비율" 러너**(커밋 21dfb04, tools/baseline_failure 세 표 재사용) — R1의 진짜 표적 지표. ③ RMM depth 열화 Δ(R1/R2 둘 다) — 아직 0회 측정.
- 출처: hpca100 `logs/r1_rescore_v2_20260922.log`, `logs/r2_rescore_v2_20260923.log`. ckpt: R1 `epoch25_67.62_top1`, R2 `epoch40_67.6_top1`.

### 그 밖의 완주

| 런 | 결과 | 비고 |
|---|---|---|
| **E1-shared s821** (E1+완전공유 LoRA r16, 40ep) | 완주(Total 14:05:54), trainer val best **65.72@ep35** | **부진** — E1 스크린 trainer val(~66-67)보다 낮음. 게이트는 legal Δ24 ≥ −0.3 + RMM depth 절반 → **legal 재채점 전 판정 불가** |
| **e1scr_s903** (E1 40ep 스크린 s903) | 완주, trainer val 65.80@ep40 | legal 재채점 전. E1/R1/R2 시드 902·903 짝 구성용 |
| **Q1b-1 품질 헤드 프로브 15ep** (09-21) | **PASS(여유 얇음)** — depth·LiDAR만 신호(RGB 0.794·event 0.584) | → QAF 마스크 범위 **depth·LiDAR로 축소** 확정. **Q1b-2(연산자별 분해, RGB 범위) 결과 회수 아직** — jarvis `/SSDb/jemo_maeng/qaf/q1b2_probe.log`(09-22 16:24 갱신) |
| **DGFusion 기준선 (b) 열화 재학습** | 완주(09-22 02:47, 200k, model_final.pth) | `/SSDb/jemo_maeng/dgfusion_train/output/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade/` — **val-best 사후 스윕(ckpt 20개) 미실시** |

## 2. 실행 중 (09-23 15:30 실측)

| 런 | 서버/GPU | tmux | 상태 | 완주 ETA |
|---|---|---|---|---|
| **Q2** (P54 증류 대조군, 40ep) | jarvis 0 🔴ban | (p54_q2 세션 소실, 프로세스 생존 PID 1408491) | ep30/40 @10:30 → 현재 ~ep34 추정 | **~오늘 22:00** |
| **Q3** (QAF 본 카드, 40ep) | jarvis 4 | (〃 PID 1284291) | ep30/40 @10:30 → 현재 ~ep32 추정. ep25: val 65.99·test best 55.85@20 (Q2 val 65.76·test 54.60 — Q3 우세) | **~내일 01:00** |
| **muphys_826** (MUSES PhysAug-off 셋째, 300ep) | jarvis 1,2 | `muphys_826` | 09-23 01:40 기동 | ~09-25 |
| **muphys_824** | yeon 4,5 | `muphys_824` | ep163/300 @15:00 | **~09-24 낮** (사망 시 best 80.66@68) |
| **muphys_825** | yeon 6,7 | `muphys_825` | ep162/300 @15:00 | ~09-24 낮 (〃 81.30@74) |

**가용 GPU(15:30)**: hpca100 **1,2,3 전부 해방**(E1-shared·재채점 종료) · jarvis **3,5,6,7 해방**(minkyoung_chun 종료, GPU0=Q2·1,2=826·4=Q3) · yeon GPU0-3은 sangtae_park 점유 유지 · bengio 전면 고장 · lecun 배치 금지.

## 3. 체인·자동화 상태

| 체인 | 상태 |
|---|---|
| hpca100 `chain_e1shared` | **소진**(R1 재채점 후 E1-shared 기동 → 완주) |
| hpca100 `chain_r2rescore` | **소진**(R2 종료 감지 → 재채점 → 완료) |
| jarvis `chain_muphys826` | **소진**(e1scr_s903 종료 → 826 기동, 실행 중) |
| **Q2/Q3 완주 후 재채점 체인** | **없음 — 수동 필요.** 완주 확인 후: ① legal v2 재채점(`tools/legal_rescore_v2.py`, 가드 `--check` 먼저) ② 강건 벤치 EMM 15조합·RMM r∈{.75,.5,.25}·NM 저/중/고(`tools/missing_modality_eval.py`) ③ η̂ 치환 검정(`tools/qaf_permutation_test.py`) — QAF 판정의 주 전장 |

## 4. 다음 세션 할 일 (우선순위)

1. **R1/R2 판정 완성**(생각정리 세션 소재): 짝의 24클·얇은4 집계 + "찾고도 못 그린 비율" 러너(21dfb04) + RMM depth Δ. R1이 유력 통과 후보.
2. **Q2/Q3 완주 확인 → legal+강건+치환 검정**(오늘 밤~내일 새벽 완주). hpca100 1,2,3이 비어 있으니 여기서.
3. **E1-shared legal v2 재채점 + RMM depth 검정** — 게이트(Δ24 ≥ −0.3 & RMM 절반) 판정. trainer val 부진이라 기대치는 낮춤.
4. **DGFusion (b) val-best 사후 스윕**(ckpt 20개) — G-robust-vs-DGFusion (b) 행 재료.
5. **Q1b-2 결과 회수**(jarvis `qaf/q1b2_probe.log`) — RGB 품질 헤드 범위 최종 확정(스칼라만 쓸지/제외할지).
6. 🔴 **ISSUE-038**: hpca100 미커밋 R1/R2·QAF 코드(+566줄, cddc319 위) develop 커밋 — **그 전까지 hpca100 repo pull/checkout 금지**.
7. 미기동 스크린(여유 GPU에): E1-shared 시드 902·903(configs 9종에 있음, bengio- prefix — 경로만 교체), R1/R2 시드 902·903(**R1/R2 s821 판정 후**).
8. 배경 판정 대기: DELIVER 넷째 페어(시드4) 판정, MUSES 공정선 제출본(E13M 81.95) user 제출 여부.

## 5. 사고·주의 기록 (반복 방지)

- **yeon 리부트 사고**(09-22 12:55→13:28, 원인 불명): tmux 전소실, 3런 사망(ckpt 전부 보존). 리부트 직후 GPU0-3을 타인이 선점 → "유휴 즉시 재기동" 원칙의 중요성 재확인. 교훈: 재개 체인/러처는 tmux+AUTO_RESUME(매-에폭 원자적 last_checkpoint.pth)이 정답.
- **R2 평가 config YAML 들여쓰기 사고**(09-23): COMPONENT 블록을 4-space로 넣어 파싱 즉사 + tee를 마지막 명령에만 걸어 원인이 로그에 안 남았음. 수복: 2-space 수정 + `yaml.safe_load` 검증 + **그룹 tee 패턴** `{ a && b ; } 2>&1 | tee`. 재사용 config 만들 때 이 패턴 강요할 것.
- **hpca100 ckpt 확인 함정**(09-22): 체크포인트는 `SAVE_DIR/DELIVER_ReliaDINO-ViTL16_idel/` **서브디렉터리 안** — 1단계 ls 로 "없음" 판단 금지.
- **Q2가 jarvis GPU0(policy ban:0)에서 실행 중** — user 확인 요청된 채 미회신. 종료 시 해방.
- 감시 함정 10 재확인: ckpt 자동 선택 시 `test_` 접두어 제외(`ls -t epoch*top1*.pth | grep -v test_`).

## 6. 경로 레퍼런스

- **hpca100** repo `~/SSDb/jemo_maeng/src/drone-MemorySAM`(🔴 ISSUE-038) · config `/home/jovyan/SSDb/jemo_maeng/queue_cfg/`(R1/R2/E1shared hpca100용) + `configs/eval/cfg_{R1,R2}_eval1024_hpca100.yaml` · 로그 `logs/{r1_s821,r2_s821,e1shared_s821}_launch_20260922.log`, `logs/{r1_rescore_v2_20260922,r2_rescore_v2_20260923}.log` · 런 출력 `outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_{R1,R2,E1shared}/`
- **jarvis** repo `/SSDb/jemo_maeng/src/drone-MemorySAM`(b99ce18, QAF 코드는 `/SSDb/jemo_maeng/qaf/`+워킹트리) · `queue_cfg/`(e1scr903, muphys826) · Q2/Q3 로그 `logs/p54_q{2,3}_screen40_launch.log` · e1scr_s903 출력 `outputs/ReliaDINO/yeon_deliver_rgbdel_P46_c3only_seed20260903_screen40_E1/`(ckpt md5 이송본)
- **yeon** repo `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop` · muphys 로그 `logs/muphys_8{24,25}_resume_20260922.log`
- **비교 기준값**: E1 스크린 짝 55.94/68.56 · E1 확정 3시드 56.24±0.42 / 69.51±0.15 · DGFusion 56.71 · MM SAM-adapter 69.60/57.35 · CAFuser-CAA 68.79
- 판정 규칙·게이트: [../decisions/2026-09-20-p54-quality-aware-fusion-proposal.md](../decisions/2026-09-20-p54-quality-aware-fusion-proposal.md) §4 · [../experiments/judgment-ledger.md](../experiments/judgment-ledger.md)

## 7. 이 세션의 변경 이력 요약

09-22: yeon 리부트 사고 파악·muphys 824/825 AUTO_RESUME 재개 · hpca100 user 종료 세션(R1/R2) ckpt 발견·원상 재개 · R1 완주(19:17) · autoplace/체인 4건 설치 · R1 legal 재채점·E1-shared 기동 · jarvis e1scr_s903 ckpt 이송 재개 · ISSUE-038 등록 · 로깅 파이프라인 전면 갱신(current/history/plan/monitor/issues).
09-23: R1 재채점 확정(56.80/69.09) · R2 완주+재채점 사고 수복+확정(56.56/68.88) · e1scr_s903 완주 · E1-shared 완주(부진) · muphys_826 기동 · R1/R2 설명 아티팩트 `docs/2026-09-23-r1-r2-explainer.html`.
