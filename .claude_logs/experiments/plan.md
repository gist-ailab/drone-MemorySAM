---
created: 2026-07-16
updated: 2026-09-24 15:50 (monitoring 세션 — P54-Q2 계열 시드 재현(s902 hpca100 GPU2·s903 jarvis GPU5·noKD/noDeg jarvis)·hpca100 GPU1 강건성 Q2→Q3 전환·jarvis muphys_826 완주(GPU1,2 반납, 후속 MUSES 공식 재채점 착수)·bengio GPU5 하드웨어 고장+GPU0-3 CUDA 컨텍스트 불능(관리자 리부트 요청, 미완료)·jarvis CAFuser(b) 09-24 12:x 중단(dgfusion 세션, 체크포인트 iter90000 보존, 자연완주분과 별개 사용자 지시)·E1-shared 게이트2 s903 통과(저하비율 0.46~0.49 ≤0.5)·E1확정 시드2·3 부분 프로토콜 hpca100→jarvis 이관 재착수(bengio 소실분) 반영 — 판정 전부 judgment-ledger.md 참조, 여기엔 상태만) ; 2026-09-22 17:50 (yeon 리부트 사고 — muphys_824/825 ep75 사망→GPU4-7 AUTO_RESUME 재개, e1scr_s903 미재개·GPU0-3 타인 선점, hpca100 R1/R2 user가 죽인 세션 ckpt 원상 재개·미커밋 코드 주의, jarvis Q2/Q3 생존·DGFusion (b) 09-22 02:47 완주) ; 2026-09-21 18:15 (MUSES-PhysAug-off 착수 근거 기록, 빈 GPU 자동 배치 파이프라인 등재, R1·R2 를 jarvis 에서 resume, QAF 카드 메모리 수정 실측) ; 2026-09-20 22:20 (E17 확정 2런 취소 반영, bengio 고장으로 배치 제외, E-LoRA 재채점 jarvis·yeon 분산, P54 준비 3건 hpca100) ; 2026-09-18 11:20 (registry 낡은 행 4건 실측 갱신 — C3-only 시드903·E-LoRA arm C 시드903 완주 legal 반영, E17 신규 등재, E7 풀 런 공식 val 81.88 반영 — 후 "실행 중"·"GPU 예약·점유 현황" 표를 네 서버 전수 실측으로 교체) ; 2026-09-12 18:00 (E13M 공식 80.7998 게이트 통과·E13 확정 시드1 완주 확정본·E13s3/E14/armB 재채점·자동 연쇄 3건 설치 반영) ; 2026-09-12 (실행 중 표를 네 서버 전수 실측으로 교체 — 종결 런 10건 제거·신규 6건 추가, 빈 GPU 0장/작업 14건, E1 확정 ep138 급등 반영) ; 2026-09-09 (카드 넷 legal 완결 E3 +1.40·E1 +1.07·E2 +0.72·E4 −0.03, MUSES 공식 E7/E7c/E1M, DGFusion 발산 3회·80k 재현 성립, E3b·E-LoRA B/C 기동, bengio 양도 반영) ; 2026-09-08 (노션 논문 페이지 리팩토링과 동기화 — "실행 중" 표를 실상태(카드 E1/E2/E3/E4/E7c·P52·E-LoRA·DGFusion/CAFuser 재학습)로 교체, B0 legal 수치·E0/E9/E7 판정 반영, DAILY-CARDS 행 2일차 갱신, 노션 동기화 규칙 명시) ; 2026-07-21 (ISSUE-025 MUSES radar 디코딩 버그 수정 반영 → 대기열 #3 "P39-4모달 radar-fix 재실험" 신설 + 사고 기록 1줄, 이하 대기열 번호 +1) ; ISSUE-026 ColorAugSSD RGB 붕괴 버그 반영 → hpca100 P39-DPC resume 오염 표기 + 사고 기록 1줄 + 대기열 #1 클린런 표기 ; 2026-07-23 대기열 #1 "P39.1 Rank 수리" MUSES-jarvis 분기 착수 → 실행중 표에 행 추가(jarvis 2,3,4,5, 기동검증 통과) ; 2026-07-26 P43-MUSES 완주(val 82.51@ep156, seed2 미돌파) → 대기열 #11 P44-BMR을 hpca100 GPU2,3에 착수(develop 678c493, 기동검증 통과); 2026-07-27 seed4 완주(81.92)→해방 GPU에 첫 4-modal(P39.1+radar) 착수(yeon 0,1,5, 305b030); seed2 분석 완료(trunk+2~7·VICReg lidar rank 78~100 검증); 2026-07-27 jarvis 리부트(드라이버 595.84 복구)→DELIVER 2실험 착수(P39.1-rank GPU0-3 / P44-BMR GPU4-7, BS1, develop be2603c) — DELIVER 첫 캠페인 실험; 2026-07-27 4-modal ep2 eval OOM→EVAL BS1+expandable_segments 수정 재기동(9f199be), ep4 eval 통과 확인; 2026-07-28 P44-MUSES 완주(80.71)→해방 A100에 2번째 4-modal(P44-BMR+radar) 착수(hpca100 0-3, 1cf1e66, BS1 OOM수정); 2026-07-28 hpca100 4모달 HF 백본 이중고장(offline=RANDOM INIT/online=hang) 확진 → RELIADINO_LOCAL_BACKBONE env fix(encoder.py 697a10a) → P39.1+radar seed2 클린 기동(ep2 47.61); 2026-07-28 seed3 완주(81.89@204, 5-seed variance 완결) → P44-DELIVER seed2 yeon6,7 수동기동; P44-MUSES(80.71) test staging+분석 lecun; 2026-07-28 P46-CTR 제안 등재(DELIVER SOTA class-transfer, 내부신호 RCS+MIC+prototype) ; 2026-08-03 P46 C3-only λ0.2 DELIVER 완주(200/200, test-best 57.05@ep108) = **DELIVER test SOTA 돌파 확정**(DGFusion 56.71 대비 +0.34, @768 동일 프로토콜) → λ 스윕 상단탐색 λ0.3을 jarvis GPU4-7(회수됨)에 착수, 기동검증 PASS ; 2026-08-03 λ0.2 SOTA 재현성 검증을 위해 seed2를 jarvis GPU1-3(4090×3, GPU0=user 예약)에 착수, config `jarvis-deliver_rgbdel_P46_ctr_c3only_lam02_seed2.yaml`(develop b925c90), 기동검증 PASS ; 2026-08-04 **정정**: 57.05는 test-best 체크포인트 값으로 규약상 무효 확인됨 — legal 재계산(val-best/final-iter) 결과 최고 test 55.62~55.69, DGFusion 56.71 대비 −1.0로 **SOTA 미달**(base 대비 실제 이득은 test +1.35~1.74/val +0.97로 견고, λ 최적 0.05~0.2 평탄). 상세 [experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md](analysis/2026-08-03-p46-c3only-lambda-sweep.md) ; 2026-08-06 MUSES val PQ 첫 측정(P47-MUB D-1 ep172, native, tools/eval_pq.py b6d3da0) → things PQ 22.87 ≤ 30 = P48(쿼리 경로 인스턴스 감독) 사전등록 게이트 미달 → **설계 폐기** (analysis/2026-08-06-pq-first-measurement-p48-gate.md) ; 2026-08-08 대기열·예약표 청소(완주 4건 제거, bengio 잔재 제거, CEA 프로브 등재) ; 2026-08-18 "실행 중" 표 청소(3~4주치 완주 런 박제 제거, 실상태 2런만 유지) + 대기열 #13 spatial-modality oracle 등재
---

# 🗓 실험 계획 / 큐 (Experiment Plan & Queue)

> **역할**: **앞으로 뭘 돌릴지**의 단일 출처. 모든 세션·에이전트가 **여기를 먼저 읽고, 여기에 갱신한다.**
> 구분: [registry.md](registry.md)=이미 launch된 실험 한눈표(과거·현재) · [monitor-log.md](monitor-log.md)=실시간 진행 · **이 문서=미래(큐·우선순위·GPU 예약)**.

## 🅿️ GPU 홀드 런 (placeholder) — **user 허락/요청 시에만**

> **왜**: 연구실 GPU는 비우는 순간 뺏긴다(lecun 7장 실증 상실). 다음 실험이 확정되기 전 **공백 구간에 GPU를 유지**하기 위한 장치.
> **🔴 자동 실행 금지 — user가 허락하거나 요청했을 때만 띄운다.** 세션이 임의로 올리지 마라.

**무엇을 돌리나**: **실제 P34 학습**을 돌린다(가짜 연산 아님). 산출물만 tmp 디렉터리에 **계속 덮어써서** 디스크를 안 먹게 한다.

**이름 규칙**: **"dummy/더미"로 명명하지 마라.** 실제로 도는 것이 P34이므로 **그대로 `P34_hold`**로 쓴다 — 이름이 곧 내용이라 허위가 없고, 로그·`ps`에서 무엇인지 바로 읽힌다.

```bash
# 예시: bengio에서 8장 홀드
cd /SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # 필요한 장수만큼
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# config: P34 레시피 그대로, SAVE_DIR만 tmp(덮어쓰기), EPOCHS 크게(어차피 중간에 죽임)
setsid nohup /home/jemo_maeng/anaconda3/envs/MMSS_SAM/bin/torchrun \
  --standalone --nproc_per_node=8 --master_port=29900 \
  train_reliadino.py --cfg configs/muses_P34_hold.yaml \
  > logs/P34_hold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
- `configs/muses_P34_hold.yaml` = 현행 P34 config 복사 + **`SAVE_DIR: './outputs/_tmp_hold'`**(매번 덮어씀) + ckpt 보관 최소화.
- **장수는 필요에 맞춰**: 8장 잡을 거면 `nproc_per_node=8`, 4장이면 4.

**해제 규칙 (중요)**:
1. **진짜 실험이 준비되면 즉시 죽이고 양보.** 홀드는 대기열보다 항상 후순위다.
2. **팀 내 다른 에이전트/세션이 그 GPU를 실제로 필요로 하면 즉시 양보.** (2026-07-16 실사례: 8배치 대기 잡을 위해 Arm A 종료.)
3. 홀드 런의 **산출물은 쓰지 않는다** — 성능 수치로 인용 금지. 이건 자리 유지용이지 실험이 아니다.
4. 홀드를 띄웠으면 **"실행 중" 표에 `P34_hold`로 명시**해 다른 세션이 오해하지 않게 한다.

## 📌 사용 규약 (모든 세션 필독)

1. **GPU를 잡기 전 이 문서의 "GPU 예약 현황"을 확인**하라. 남의 예약을 덮어쓰지 마라.
2. **실험을 띄우면** → "실행 중" 표에 행 추가(서버/GPU/PID/config/로그/**완주 ETA**/우선순위) + 이 문서 `updated` 갱신.
3. **끝나거나 죽이면** → "실행 중"에서 제거하고 "완료·판정" 표에 1행(결론 포함). 상세는 monitor-log.md.
4. **새 실험 제안** → "대기열"에 우선순위와 함께 추가. 근거 1줄 필수.
5. **🔴 EPOCHS를 반드시 명시하라.** 진단용 런에 300ep을 두면 며칠간 GPU를 막는다(2026-07-16 실사고, 아래 참조).
6. **우선순위 수정 자유** — 단 바꾼 이유를 1줄 남긴다.

## 🔴 GPU 점유 원칙 (user 지정 2026-07-16)

**연구실 서버 부족 → 비우면 즉시 뺏긴다. 진행에 치명적.**
- 학습 사이 **유휴 구간을 만들지 마라.** 다음 실험 미정이면 **기존 학습이라도 재기동**해 점유 유지.
- **프로세스명은 실제 실험명으로.** "더미" 표시 금지.
- **끝나기 전에 다음 잡을 준비**해 둔다. "끝나면 그때 생각"은 곧 상실.
- 실증: 2026-07-16 새벽 lecun 분석 완주 후 7장을 비우자 **즉시 타인(openvla)이 24GB×7 전부 점유** → TTA 실측 무기한 보류.
- ⚠️ **단 타인 GPU에 얹지 마라** — CLAUDE.md "빈 GPU(≤2000MiB, util≤10%)" 규칙 유지. 이 원칙은 *우리 것을 놓치지 말라*는 뜻.

## 🖥 GPU 예약·점유 현황 (2026-09-24 15:50 KST 실측 — monitoring 세션)

> 🔴 **2026-09-20 전면 교체**: 아래는 이 세션이 직접 확인한 것만 싣는다. 이 갱신에서 확인하지 못한 GPU 는 "이번 갱신 미확인"으로 표시하고 예전 상태를 그대로 베끼지 않는다.

| 서버 | 우리 점유 | 타 사용자 | 비고 |
|---|---|---|---|
| **jarvis** (4090×8) | GPU1,2,6 **E1확정 시드2·3(s902/s903) 부분 프로토콜**(RMM 4부분집합×3비율+NM, bengio 소실분 재기동, tmux `e1s23_gpu{1,2,6}`+`e1screen902_retry`) · GPU3 **Q2noDeg**(원천분리: 열화off·증류on, tmux `q2_noKDnoDeg_queue`) · GPU5 **Q2 s903 재현**(v2 재시도, tmux `q2_s903_noKD_retry2`) · GPU0 = 예약 해제(user 2026-09-30, 일반 규칙 적용) | GPU4 등 = minkyoung_chun(dgss) 수시 점유(변동 심함, 배치 전 매번 실측 필수) · GPU3,5,6,7은 09-24 12:x까지 dgfusion 세션의 **CAFuser(b)**(train_net.py, detectron2)가 잠시 얹혀 있다 user 지시로 중단(체크포인트 iter90000 `output/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade/model_0089999.pth` 보존, 자연완주 포기 아님 — 필요시 dgfusion 세션이 resume) | Q2noKD는 GPU3 완료 후 자동 착수 예정(같은 tmux 체인). muphys_826(GPU1,2, MUSES 셋째 시드) 09-24 완주(ep300, trainer val 82.06/best 82.43@172) → GPU1,2 반납, dgfusion 세션이 GPU1에서 legal MUSES 공식 재채점 착수(pid 700196) |
| **yeon** (3090×8) | GPU4,5 **muphys_824**(MUSES PhysAug-off 기준선 시드20260824, 300ep) · GPU6,7 **muphys_825**(시드20260825) — 09-24 15:50 기준 ep248/247, trainer val best 82.71@202 / 82.34@230, ETA ~09-25 | GPU0-3 sangtae_park(dice-rl) 지속 | 재학습 세션(dgfusion deliver training) 소관, 이 세션은 감시만 |
| **hpca100** (A100×4) | GPU1 **Q2→Q3 강건성 전 프로토콜**(EMM15·RMM4부분×3비율·NM3밀도, Q2 먼저: 판정세션 지시 09-24, tmux `rb_g1_q2q3`) · GPU2 **Q2 s902 재현 학습**(5차 시도로 정상 기동 확정 — 표준 env 4종+venv 절대경로 필요, tmux `q2s902_gpu2e`, ep2/40) · GPU3 **RMM+NM 전 프로토콜**(E1 확정 s821, tmux `rb_g3`, 장기 진행 중) | GPU0 = 타 tenant(~34.7GB, util 0%) | 이관④(`hpca_part4_chain.sh`, E1확정 s902/s903 부분 프로토콜)는 GPU1을 Q2→Q3로 돌리며 PAUSE — s902 2/5만 완료한 채 정지, 나머지는 jarvis GPU1,2,6로 이관해 새로 돎(위 jarvis 행) |
| **bengio** (8장) | 없음 | — | 🔴 **09-23 20:52 GPU5 하드웨어 고장**(PCIe, `Unable to determine the device handle`) → **09-24 실측: GPU0-3도 CUDA 컨텍스트 생성 자체가 불능**(`torch._C._cuda_init(): CUDA unknown error`, nvidia-smi는 정상으로 보여 착시 주의) — 노드 전체 드라이버 손상 추정, 관리자 리부트 요청 완료(user, 시각 미상)·아직 미완료(15:20 기준 SSH도 간헐적 거부). **배치 전면 금지**, 리부트 완료 후 재검증(실컴퓨트 스모크 테스트) 필수 |
| **lecun** (24GB×7) | 없음 | sangmin_park · youngjin_lee 등 수시 | GPU 대부분 유휴. 2026-09-17 배치 중단은 일시 요청이었고 상시 금지가 아님(user 2026-09-30). 일반 빈 GPU 규칙으로 배치 가능. 단 I/O 병목 이력 있음 |
| ~~B200~~ | — | — | 🔴 상실(07-15 마감) |

## 🔬 실행 중 (2026-09-24 15:50 KST 실측 갱신 — monitoring 세션)

> ⚠️ **이 표는 "지금 도는 것"만 담는다.** 완주·종결된 런은 registry/analysis로 즉시 이동.
> 🔗 **노션 동기화 규칙(2026-09-08, CLAUDE.md §3)**: 이 표·대기열이 바뀌면 같은 날 노션 논문 페이지(`Drone Object Detection for RGB-IR Fusion`, `33d05310…`) §4·§6을 `.claude/skills/notion-experiment-log/paper_page_builder.py`(절 단위 교체, 멱등; 차트는 `paper_page_charts.py`)로 함께 갱신한다.
> 🔴 **판정은 이 표에 적지 않는다.** 전부 `experiments/judgment-ledger.md`에 있다 — 여기는 실행 상태만.

| 실험 | 서버/GPU | 데이터셋 | 진행 | ETA | 목적·게이트 |
|---|---|---|---|---|---|
| **Q2 s902**(P54 증류 대조군 시드 재현, 동결교사+두패스, F·Q off) | hpca100 2 | DELIVER 4모달 | **학습 완주 + legal v2 재채점 완주(09-27)** — 트레이너 val ep40 66.06(top1) → legal v2 **test 55.98/24cls 56.95/얇은4 51.44, val 67.30/24cls 69.32/얇은4 68.17**. 짝(E1스크린s902 test 56.87/24cls 56.84) 대비 Δ25 −0.89 · Δ24 +0.11 | 완료(1/3쌍) | Q2(=T′) 시드 재현 — 클래스별 상승/하락 전표는 E1스크린s902 원자료 없어 보류, 판정은 생각정리 |
| **Q2 s903**(〃 시드903) | jarvis 5 | DELIVER 4모달 | **학습 완주(09-26 19:32, val 66.33@ep40/test 55.82@ep30) + legal v2 재채점 완주(09-28)** — test 56.07/24cls 56.52/얇은4 49.26, val 67.88/24cls 69.89/얇은4 68.26. 짝(E1스크린s903 test 56.92/24cls 56.71/얇은4 50.15) 대비 Δ25 −0.85·Δ24 −0.19·Δ얇은4 −0.89 | 완료(3/3쌍 재채점 완결, 3시드 평균은 생각정리) | 〃 |
| 🔴 **판정(09-28)**: Q2 3시드 평균 Δ24 +0.41 — 사전 기준(+0.6) 미달로 **clean 개선 카드로는 기각**(clean 손해는 없음). Q2 가치는 강건 축이므로 아래 강건성 측정으로 이어감 | — | — | — | — | — |
| **Q2 s902 강건성**(EMM15+RMM4부분×3비율+NM3, clean 게이트 55.98±0.1) | hpca100 2 | DELIVER 4모달 | **완주(09-29 00:17)**: clean 55.98(Δ=0.001 OK), EMM avg 47.92·E(0.2)=53.49. 단일모달만 존재(EMM): img 54.80·depth 45.73·event 31.43·lidar 22.87. **RMM(🔴 09-30 저하값 재정정: `MM_PRESENT_ONLY` n_combos=2 라 summary.json의 avg는 clean·열화 두 값의 평균 — 열화값=2·avg−clean, 09-29에 적었던 값은 avg를 열화값으로 잘못 표시했던 것, 생각정리 세션 지적)**: img(RGB)열화 저하 .25/.5/.75=3.17/3.76/4.69(열화값 52.81/52.22/51.29) · depth열화 저하 0.31/1.27/1.83(열화값 55.67/54.71/54.15) · event열화 저하 −0.05/−0.12/−0.14(열화값 56.03/56.10/56.12, clean과 사실상 무차) · lidar열화 저하 0.31/0.30/0.29(열화값 55.67/55.68/55.69). NM d=.05/.1/.2 = 54.10/52.71/49.53(이 값은 정정 불필요 — NM은 단일값). 원자료 NAS 보존(`analysis_logs/robust_q2_20260929/`, md5 대조 완료, summary.json만 있음·csv 없음) | 완료 | Q2 강건 축 가치 판정 |
| **Q2 s903 강건성**(〃 clean 게이트 56.07±0.1) | jarvis 6(재개) | DELIVER 4모달 | **완주(09-29 03:54)**: clean 56.07(Δ=0.002 OK), EMM avg 47.57·E(0.2)=53.30. 단일모달만: img 52.05·depth 45.97·event 31.17·lidar 23.43. **RMM(09-30 저하값 재정정, 위와 동일 사유)**: img(RGB)열화 저하 2.49/3.15/4.26(열화값 53.59/52.93/51.82) · depth열화 저하 1.57/2.94/3.60(열화값 54.51/53.13/52.47) · event열화 저하 0.00/−0.05/0.08(열화값 56.07/56.12/55.99, 사실상 무차) · lidar열화 저하 0.53/0.59/0.65(열화값 55.55/55.48/55.43). NM d=.05/.1/.2 = 52.39/50.57/48.05(정정 불필요). 원자료 NAS 보존(위 경로)·생각정리 세션에 전표 송부 완료 | 완료 | 〃 |
| **E1스크린s902 추가분**(EMM15+img결측RMM+NM — depth결측RMM은 기존 rmm_depth_e1screen902 재사용, ckpt md5 7e9b36e9 확인) | jarvis 3 | DELIVER 4모달 | **완주(09-28 22:52)**: EMM clean 56.87(Δ=0.006 OK) · RMM(img) clean 56.87(Δ=0.006 OK) · NM clean 56.87(Δ=0.006 OK), 3단계 전부 게이트 통과 | 완료 | Q2 짝 강건 대조용 |
| **E1스크린s903 추가분**(〃, ckpt md5 c9e2c8e6 확인) | jarvis 5 | DELIVER 4모달 | **완주(09-28 22:53)**: EMM clean 56.92(Δ=0.003 OK) · RMM(img) clean 56.92(Δ=0.003 OK) · NM clean 56.92(Δ=0.003 OK), 3단계 전부 게이트 통과 | 완료 | 〃 |
| **Q2noDeg**(원천분리: DEGRADE_P=0, KD 유지 — 증류만의 순기여) | jarvis 3 | DELIVER 4모달 | 진행 중 | 미확인 | Q2 이득 원천 분리 |
| **Q2noKD**(원천분리: KD_W=0, 열화만 유지) | 대기(4차 미착수) | DELIVER 4모달 | **3차(yeon 0) 09-26 17:49~18:16경 크래시 — ep5 Val mIoU 60.15 기록 후 test-eval(1897장) 100% 완료 시점 checkpoint `torch.save` 중 Traceback, tmux 창·프로세스 소멸.** 🔴 근본원인 확인: OOM 아니라 **디스크 부족** — yeon `/SSDb` 100% 사용, **가용 0바이트**(`df -h /SSDb` 확인). `jemo_maeng` 계정만 1.4TB 점유(`src/` 1.2TB가 최대), 전체 3.6TB 중 타사용자 몫도 큼. 1·2차(jarvis OOM)와 원인이 다르다 — 재시도 전 디스크 확보 필요(대량 삭제라 사용자 판단 대기). | 미확인 | 〃 |
| 🔴 **tmux 세션명 오해 주의**: `q2_s903_noKD_retry2`(jarvis GPU5)는 이름과 달리 **Q2 s903 재현**(v2 재시도, 노KD 아님)을 돈다 — 판정 세션 지시(09-25)로 명시. **다음 기동부터 세션명을 실제 내용에 맞출 것.** noKD/noDeg의 902·903 전용 config는 만들지 않는다(s821 두 결과 먼저 확인 후 결정, 판정 세션 지시 09-25). | — | — | — | — | — |
| **E1확정 시드2·3 부분 프로토콜**(RMM 4부분집합×3비율+NM, bengio 소실분) | jarvis 1,2,6 | DELIVER 4모달 | 09-24 15:46 기동(8개 부분집합 작업, hpca100에서 2/5만 하다 중단된 s902 포함 전량 재기동) | 미확인 | E1 확정 강건성 전체 재구성(bengio 고장으로 유실됐던 것) |
| **e1screen902 재시도**(RMM depth, 09-24 08:44 OOM) | jarvis (여유 GPU 대기) | DELIVER 4모달 | 대기(GPU1/2/6 중 여유 시 자동 착수) | 미확인 | E1-shared 게이트2 s902 짝 완성 |
| **Q2→Q3 강건성 전 프로토콜**(EMM15·RMM4×3·NM3, Q2 먼저) | hpca100 1 | DELIVER 4모달 | Q2 EMM 진행 중(09-24 01:50 UTC 시작) | 미확인(EMM 1건 과거 8~9h대) | QAF(Q3)가 강건축에서 Q2를 못 넘으면 Q3 종료(판정세션 지시) |
| **RMM+NM 전 프로토콜**(E1 확정 s821) | hpca100 3 | DELIVER 4모달 | 장기 진행 중(완료 로그 없음) | 20:37 확인 크론(`51f7c601`) | E1 확정 강건성 |
| **E1-스크린821 EMM+RMM+NM v3**(40ep 스크린 규약 시드821 짝, Q2s821과 강건성 대조용, 09-25·09-26~27 두 차례 yeon `/SSDb` 포화로 무산 후 09-29 디스크 정리 뒤 anti-collision 구조로 재기동) | yeon GPU5(emm)+GPU6(rmm_nm) | DELIVER 4모달 | **완주(09-30 02:30, clean 55.9423, Δ=0.002 OK)**. EMM(15조합) avg **39.228** · E(0.2)=50.1109 · E(0.1)=53.2609 · E(0.05)=54.6647. 단일모달만 존재: img 39.79·depth 45.61·event 6.46·lidar 3.75. **RMM(img열화, depth+event+lidar 보호)**(🔴 09-30 저하값 재정정: `MM_PRESENT_ONLY` n_combos=2라 summary.json avg는 clean·열화 평균 — 열화값=2·avg−clean, rmm_r0.75.csv 직접 대조로 확인, 생각정리 세션 지적) r.25/.5/.75 = 열화값 53.3429/52.8115/51.7709 · **저하 2.60/3.13/4.17**(summary.json avg는 54.6426/54.3769/53.8566, 참고용). NM d.05/.1/.2 = 53.1816/50.3566/44.3066(저하 2.76/5.59/11.64, 정정 불필요). Q2 s821 대비 비교는 그쪽도 같은 방식 재정정 중이라 이 행에는 병기하지 않음 — 최신 비교값은 생각정리 세션 `judgment-ledger.md` 09-30 행 참조. 원자료 yeon `robust_out/yeon_e1screen821_v3/{emm,rmm_img,nm}/test/missing_modality/`(csv 포함 보존) | 완료 | Q2s821과 짝 비교(스크린40 규약 기준 강건성) |
| **muphys_824 = MUSES PhysAug-off 기준선 시드20260824** | yeon 4,5 | MUSES 3모달 | ep248/300(best 82.71@202) | ~09-25 | 공정선 3페어(재학습 세션 소관) |
| **muphys_825 = 〃 시드20260825**(학습 완주 09-25, 826과 동일 절차로 공식 val 재채점·NAS 보존·제출 zip 준비 완료) | yeon 6 | MUSES 3모달 | 완주 300ep, val-best epoch258_82.4 · 공식 val **82.19** | 완료(09-25) | 공정선 3페어 완결(824 완주 대기) |
| **muphys_826 legal MUSES 공식 재채점**(val-best ep172, 82.43) | jarvis 1 | MUSES 3모달 | 기동 중(dgfusion 세션, pid 700196) | 미확인 | 공정선 3페어 완결(학습은 09-24 완주) |

> 🔗 **완료(09-24)**: R1 종료(게이트 미달) · E1-shared 게이트1 통과·게이트2 s821 미달(0.67>0.5)·**s903 통과(저하비율 0.46~0.49≤0.5, 902는 e1screen902 재시도 대기)** · R2 902/903 종료 후보(재채점 완료, ckpt md5 NAS `screen40_bengio_20260923/`와 일치 확인) · muphys_826 완주. 판정 상세는 judgment-ledger.md.
> 🔴 **CAFuser(b)**(dgfusion 세션, detectron2, jarvis GPU3,5,6,7): 09-24 12:x user 지시로 중단(iter90000 ckpt 보존). bengio 이관 시도했으나 09-24 15:xx 기준 bengio CUDA 전면 불능(리부트 대기) — 재개 시점 미정.

> 🔴 **09-23 10:2x 갱신 — 밤새 결과**: **R1 legal v2 재채점 완료: test 56.80 · val 69.09**(24클래스 56.65 · 얇은4 56.39) — **E1 스크린 짝 대비 Δtest +0.86 · Δval +0.53, test 56.80 = DGFusion 56.71 초과(+0.09, 25클래스 축)를 40ep 스크린 단계에서 달성.** 잔여 = 짝의 24클래스·얇은4 집계 + "찾고도 못 그린 비율" 러너. **R2 학습 완주**(09-23 07:14, best 67.60@ep40) — 재채점 1차 기동이 config YAML 들여쓰기로 즉사했던 것을 10:15 수복 재기동(~13:30). **e1scr_s903 완주**(01:30, trainer val 65.80@ep40, legal 재채점 전) — GPU1,2는 체인이 muphys_826으로 전환(01:40 기동).

> 🔴 **09-20 완주**: **4탭+센서별 프로토타입 확정 시드4(E13)**(yeon 4,5) 는 09-20 05:09 완주(트레이너 val 최고 68.35@110), legal v2 재채점도 같은 날 끝났다 — **test 55.67 / val 69.40**. GPU4,5 는 이후 E-LoRA arm C 시드902 재채점으로 재배정(위 GPU 표). **E-LoRA arm C 시드902** 학습도 완주(Total Training Time 16:59:24, 트레이너 val 최고 68.69@ep60).

> 🔴 위 진행 값은 전부 트레이너 val 이며 legal 이 아니다. 판정은 `val.py` 하네스(1024·native-GT BS1) 또는 `tools/eval_muses_official.py` 재채점으로만 한다.
> ✅ **09-16 MUSES 확정 판정(생각정리)**: E13M 이 세 게이트를 3페어로 통과 — **MUSES 에서는 E13 이 E1 보다 낫다**. E1M 은 "일관된 소폭 양성(+0.52), 통과선 미달"(§5-31). 📌 §0-1 에 MUSES 조건별 표본 규칙 추가(§5-31b).
> ✅ **09-16 E13M 시드3 완주·공식 80.8403** — MUSES 3페어 완결: E13M−E7 평균 **+0.938**(페어 +0.72/+1.53/+0.56), E13M−E1M 평균 +0.421, 시드3 은 여덟 조건 모두 E7 대비 양수 → 게이트 ①②③ 통과(§5-31). rain/night 손실은 시드3 에서 재현 안 됨.
> ✅ **09-16 조건별 평가 3시드 확장**: E1·E13 모두 §0 악조건 규칙 유지(최저 Δ24 = E1 +0.06 · E13 −0.11). 조건별 평균 Δ24 는 다섯 조건 모두 E1 ≥ E13(§5-30 ①).
> ✅ **09-16 MCubeS 같은 코드 3페어 완결**: 시드 3407 +0.38 · 시드 0828 **−0.26** · 시드 0827 **+0.19**(기준선 최종 57.65 대 개입 최종 57.84) → **3페어 평균 +0.10** 으로 사전 등록 게이트(+0.5)에 미달하고 부호도 갈린다. 사전 등록한 종결 조건에 해당하므로 MCubeS 시드를 더 늘리지 않는다(최종 판정은 생각정리, §5-30 ②).
> ✅ **09-16 E1 확정 시드3 완주**: legal test 54.81 / **val 67.89(내부 legal val 최고 경신)** · 24클래스 55.35(§5-29). E1 3시드 평균 test 54.93 · 24클래스 55.49 · val 67.58. 판정은 같은 코드 기준선 902·903 완주 후. jarvis 2·4 비어 있음(다음 카드 결정 대기).
> ✅ **09-15 hpca100 복구**: 09:24~09:40 공유 볼륨 가득 참(Errno 28)으로 6런 사망 → 사용자 승인 방식 B(판정 끝난 12개 디렉터리의 test_*·val top2~5 63개, 약 108G 삭제, 카드 §4.5)로 여유 0→112G → 12:24~12:25 E1M 풀 런 이어 받기 + MCubeS 3런 기동. 새 MCubeS 런은 `TRAIN.SAVE_TOPK 1`·`SAVE_TEST_CKPT false`(40843ed). E7 시드 902·903 은 완주(79.79 / 80.45), E1M 시드3 은 ep35 val-best(80.86)로 판정(결손 각주) — 세 런 MUSES 공식 재채점 대기.
> ✅ **09-15 DELIVER 조건별 평가(E1·E13 확정 시드1, yeon 4·5 새 체크아웃, 로드 0/0)**: 24클래스 Δ vs seed821 기준선 — E1 +0.06~+1.19 · E13 +0.14~+1.18, 두 카드 모두 −0.5 미만 조건 없음(§5-27). 판정은 생각정리. yeon 4·5 는 09:38 부터 비어 있음(다음 카드 결정 대기).
> ✅ **09-15 E1 확정 시드2 완주**: legal test **55.35** / val **67.47** · 24클래스 **55.85**(§5-26). E1 2시드 평균 24클래스 55.55. 판정은 같은 코드 기준선 902·903 완주(09-17) 후.
> 🔴 **09-15 07:30 철회 — MCubeS E1Mc 게이트 통과 무효**: yeon 체크아웃(378864f)에 TAPS 코드가 없어 E1Mc 0827·0828 과 yeon E1 확정 시드4 가 TAPS 없이 돌았다(total_trainable = 기준선). 같은 시드 재실행이 +1.1~1.35 흔들려 MCubeS 는 E1·E13 모두 판별 불가(§5-25). 재판정 = hpca100 동일 코드 기준선 B0Mc 3시드(9628ec6). yeon E1 시드4·C3 시드4 는 사용자 동의 후 새 체크아웃에서 재기동 예정(생각정리).
> 🟡 **(b) 잠정**: 5시드 top1 규칙 24클래스 54.66 대비 E13 확정 3시드 Δtest +1.31 · Δ24 +0.67 — 잠정 통과(§5-24), 확정은 (c).
> ✅ **09-14~15 완주·확정**: E13 확정 3시드(평균 test 55.14 · 24클래스 55.33, 게이트는 분모 (c) 완주 후 판정 — §5-22) · E-LoRA arm C(legal test 55.72 / val 67.16 · 24클래스 55.09 — 3자 1페어 §5-23) · MCubeS E13Mc(같은 척도 +0.40, 게이트 근소 미달 — §5-21 정정).
> 📌 **자동 연쇄**: `scripts/run_legal_rescore_jarvis_chain.sh`(770eaf5, REPO 환경변수로 yeon 겸용) — 대기 중 = jarvis `rescore_e1conf_s2`·`rescore_e1conf_s3`.
> 🔴 **yeon 4·5 상실**(12:10): 다른 사용자(sangmin_park)가 점유 → MCubeS 0827 쌍은 lecun 0·1 로 이전(데이터 11G 복사, config 5eb32f7). yeon 0~3 은 12:20 새 체크아웃으로 E1·C3 시드4 v2 재기동(옛 두 런 중단, 사용자 동의).
> 🗑 이 갱신에서 내린 종결 런: E13 확정 시드3 · E-LoRA arm C(재채점 완료) · MCubeS hpca100 4런(09-14 16:53 완주)

**카드 넷 legal 완결(09-09, val-best ckpt · `val.py` 1024 BS1 native GT)** — 판정 주체 = 생각정리 세션:

| 카드 | 축 | 트레이너 val-best | legal val | legal test | Δval | Δtest | 판정 |
|---|---|---|---|---|---|---|---|
| B0 | 기준선 | 65.40@40 | 64.97 | 53.78 | — | — | 대조군 |
| **E3** | 센서별 클래스 prototype | 65.97@25 | 미측정 | **55.18** | +0.57* | **+1.40** | 통과선 초과(카드 최고) |
| **E1** | 중간층 4탭 읽기 | 67.25@35 | 66.90 | **54.85** | +1.93 | **+1.07** | ✅ 스크린 통과(확정) |
| E2 | 전 선형층 LoRA r32 | 68.65@40 | 68.38 | 54.50 | +3.41 | +0.72 | 회색지대(보류) — E2s2·E12로 재판정 |
| E4 | 혼동 쌍 margin(auto k5) | 65.92@15 | 65.86 | 53.75 | +0.89 | −0.03 | ❌ 폐기(확정) |

\* E3의 Δval은 트레이너 val 기준(legal val 미측정). 관측: 트레이너 val 순위(E2>E1>E3>E4)와 legal test 순위(E3>E1>E2>E4)가 역전, val 이득이 클수록 test 전이율 낮음(E2 21% · E1 55% · E3 증폭). 클래스별: E1 RailTrack +13.13·TrafficLight +5.01·SideWalk +3.09, Wall +0.26·Static +0.25·Water −1.71 / E2 RailTrack +15.77·TrafficLight +12.49, Wall −6.40·Static −5.44·Water −5.47 / E4 RailTrack +13.73·Ground +10.88·Water +5.86, Pole −8.87·TwoWheeler −5.10·GroundRail −4.51·RoadLine −3.74. 🔴 **트레이너 val − 정본 val = B0 +0.43 · E2 +0.27 · E1 +0.35 · E4 +0.06** → 편차가 커 고정 보정값으로 쓰지 말 것.

**MUSES 공식 재채점 세 건(`tools/eval_muses_official.py`, 250장, 공식 native 1080×1920)**: **E7**(PhysAug-off, 4탭 없음) **80.0756** / E7c(PhysAug-on) 79.9417(−0.134 → **PhysAug-off로 통일 확정**) / **E1M**(off + 4탭) **80.416**(+0.340 — 게이트 문면상 미달이나 **폐기 아님: 중립~미소 양성, 채택 저지 사유 없음**(생각정리 세션 판정, 카드 문서 §5); 트레이너 val-best 80.64@35, Δ(E1M−E7) ep10 이후 7지점 연속 양수; 조건별 fog/day 85.43 ~ rain/day 67.24). letterbox 1024: 80.2872 / — / 80.6358.

**정본 ckpt NAS 보존**: E2·E7·E7c val-best → `/drone_nas/…/ckpts/daily_cards_20260908/`(md5 대조), hpca100 `/tmp` 산출물은 새 val-best마다 자동 회수(E1M ep20 첫 회수 md5 일치). E1 ep35·E3 ep25·E4 ep15 ckpt도 보존 대상.

**직전 완결(09-07~08)**: **B0 기준선 스크린 완주 → legal test 53.78 / val 64.97**(트레이너 val 65.4, 정본 대비 +0.43 — 판정은 정본으로만) · E0 특징 프로브(기각: raw 27.8 < adapted 35.6 < fused 45.1, depth 중간층에 Water 47.6·RailTrack 23.9 잔존 → E1·E3 유지, E5·E6 하향) · E9 로짓 보정 폐기(53.57/53.45/51.89) · E7 MUSES PhysAug-off 완주(트레이너 val 80.29@40, 공식 재채점 대기) · P52 MCubeS seed1 완주 58.18@174(final 57.96, 3-seed 정본 대역 안 = 동률) · P50-EXT 게이트 기각(ep30 53.10) · N7 VICReg-off 완주(val-best 66.56@32).
**직전 완결(08-31)**: P50 파인튠 게이트 통과(+0.74, H22✓) · N2 믹서 판정(mean 55.45, H21✗) · N6 재선택 5/5(54.39±0.76) · MCubeS 3-seed(58.07±0.49) · MUSES 시드 3점(spread 0.66) · P51 페어2 완주(각주) — 전부 analysis/registry 반영됨.

## ⏸ bengio 중단 런 — 재개 대기 (2026-09-09 04:20 KST)

> 🔴 **전제 모순 확인 필요(2026-09-17)**: 아래는 "bengio 를 후배에게 넘겨야 해서" 중단한 기록인데, **지금 bengio 8장에서 우리 E-LoRA A·B 네 런이 돌고 있다.** 서버가 돌아온 것이라면 세 런(E4b·B0s2·E1s2)은 `AUTO_RESUME` 으로 이어 돌릴 수 있고, 아직 남의 것이라면 이 절의 재개 계획을 접어야 한다. registry 기준으로 세 런은 09-09 이후 재개된 적이 없다. **사용자·생각정리 확인 대상.**

> **왜 껐나**: bengio 를 후배에게 넘겨야 해서 user 지시로 **미리** 중단했다(09-09 04:18 KST, SIGTERM).
> **버린 것 없음** — 셋 다 `AUTO_RESUME: true` 이고 체크포인트가 `outputs/` 에 남아 있다. 같은 서버든
> 다른 서버든 **SAVE_DIR 의 산출물만 있으면 그 자리에서 이어 돌아간다.**
> 🔴 **bengio 의 `/SSDe/jemo_maeng/src/drone-MemorySAM-daily/outputs/` 를 지우지 마라.** 지우면 재개가 아니라 재시작이 된다.

| 런 | 중단 시점 | 재개 지점(ckpt) | 산출물 | config | GPU |
|---|---|---|---|---|---|
| **E4b** 혼동쌍 명시(RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain) | ep14 의 79% | `epoch10_62.83_top1` / `last_checkpoint.pth` | 8.7G | `bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_E4b.yaml` | 3장 |
| **B0s2** 기준선 시드2 | ep10 의 19% | `epoch5_59.71_top1` / `last_checkpoint.pth` | 5.3G | `bengio-…_seed20260902_screen40_B0s2.yaml` | 2장 |
| **E1s2** 4탭 시드2 | ep6 의 37% | `epoch5_61.75_top1` / `last_checkpoint.pth` | 5.4G | `bengio-…_seed20260902_screen40_E1s2.yaml` | 2장 |

**중단 시점의 트레이너 val**: E4b ep5 59.66 · ep10 62.83 / B0s2 ep5 59.71 / E1s2 ep5 **61.75**
(E1s2 의 ep5 61.75 는 E1 본판 60.98 보다 +0.77 — 시드 판별의 첫 표본)

**재개 방법**
1. 같은 서버(bengio 반환 후)면 원래 명령 그대로 다시 띄우면 `AUTO_RESUME` 이 `last_checkpoint.pth` 를 잡는다.
2. **다른 서버로 옮기면** ① `outputs/ReliaDINO/<SAVE_DIR명>/` 을 그 서버로 복사 ② config 사본에서 `SAVE_DIR`·`DATASET.ROOT`·`TEST.FILE` 을 그 서버 경로로 고침 ③ 같은 명령으로 기동. hpca100 E1M 을 이 방식으로 되살린 전례가 있다(2026-09-08, `/tmp` 우회).
3. 기동 검증: 로그에 `Resumed weights from … (epoch N) missing=0 unexpected=0` 과 `[SEED] fix_seeds(...)` 가 찍히는지 확인.

**남겨 둔 것**: E4 ep15 legal test 재채점(GPU0, PID 2506380)은 **끄지 않고 계속 돌린다**(04:14 시점 69%, 04:45 전후 완료 예정). 이것만 GPU 한 장을 쓴다.

## 📋 대기열 (우선순위 순) — 2026-08-24 전면 재설계 (논문-가치 필터)
> 2026-09-26부터 대기열·완료 판정의 정본은 `lab-plan`(`.claude_logs/plan/plan.yaml`)이다. 아래 생성 블록은 10분마다 자동 갱신된다. 수정은 `lab-plan add/edit/status/verdict`로 한다.
<!-- lab:plan-queue:begin -->
| id | 우선 | 제목 | 바꾼 변수 | 상태 | runs | 예상 비용 |
| --- | --- | --- | --- | --- | --- | --- |
| DRN-260929-01 | P20 | P56-A · 모달 충돌 학습(한 센서 영역에 틀린 내용을 넣고 나머지 센서와 정답을 따르게) | Q2 의 열화 패스에 모달 충돌 증강 추가: 한 센서의 임의 영역을 다른 장면 내용·포화 얼룩으로 바꾸고 정답은 그대로(나머지 센서가 본 장면) | running | - | A100 1장 × 약 3.5일(에폭당 약 2.1시간 × 40) × 2시드(통과 시 +1시드), 강건 평가 시드당 약 10 GPU-시간 |
| DRN-260929-02 | P21 | P56-B · 거리 조건화 attention(depth·LiDAR 입력의 거리값을 융합 attention 의 위치 정보로) | CrossModalAttentionLayer 로짓에 −λ_h·|log r_i − log r_j|/σ_h 편향 추가(λ 초기 0). r = depth 토큰 평균 → 없으면 LiDAR 유효 화소 평균 → 둘 다 없으면 0 | running | - | A100 1장 × 약 3.5일 × 2시드(+ B-2 거리 위치 임베딩 ablation 1시드) |
| DRN-260929-03 | P22 | P56-C · 환경 조건 LoRA 전문가 혼합(공유 LoRA 와 센서별 LoRA 를 얕은 블록 통계로 토큰마다 섞음) | SharedLoRAQKV 확장: ΔW x = α(x)·shared_r8(x) + (1−α)·sensor_r8[m](x), α = sigmoid(MLP(LN(block6 특징 stop-grad) + 센서 임베딩)), 블록 7~24 에만 적용, 라우터 약 7만 파라미터 | running | - | A100 1장 × 약 3.5일 × 2시드 |
| DRN-261001-02 | P30 | N-RGBX-T · E1 레시피 2모달 교사 3런(RGB+Depth → RGB+LiDAR → RGB+Event, 시드821 40ep 스크린) — N-RGBX 학생의 교사이자 MM SAM-adapter 와 같은 센서 조합의 1차 비교선 | 센서 집합만 축소(DATASET.MODALS: [img,depth] / [img,lidar] / [img,event]). 그 외는 bengio E1 시드821 40ep 스크린 config(bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_E1.yaml) 그대로. QAF 없음(교사 자체가 clean 학습). 새 config 3개 = bengio-deliver_rgbd|rgbl|rgbe_P46_c3only_seed20260821_screen40_E1.yaml(감시 세션이 파생·스모크) | running | - | 런당 약 1.2 GPU-일(bengio 3090, 2모달이라 4모달 E1 40ep 1.7일보다 짧음) × 3 = 약 3.6 GPU-일. bengio 에서 P56-C 시드902 다음 순서로 RGB-D 부터 기동(GPU 유휴 방지) |
| DRN-261001-03 | P32 | Q0-기준선-공식 · DGFusion 공식 DELIVER 가중치(게시 val 66.51/test 56.71)로 clean 재측정 + 같은 강건 프로토콜 61케이스 — (a) 재현본(test 55.68) 대신 발표 가중치를 기준선으로 | 학습 없음. dgfusion 세션이 보존한 공식 가중치(NAS ckpts/dgfusion_official_20261001/ DELIVER clde)로 ①README 명령 그대로 clean val·test 재측정(약 15분) ②tools/baseline_failure/robust_bench_eval.py 로 61케이스 측정(약 7 GPU-시간). 등가검증은 DRN-260930-02 와 같은 세 케이스 | queued | - | bengio GPU7(현재 유휴) 1장, clean 15분 + 61케이스 약 7시간 |
| DRN-260926-58 | P40 | CAFuser (b) · 열화 커리큘럼 재학습 재개(iter 90k→267k, detectron2) + 같은 강건 프로토콜 61케이스 측정 — 두 번째 융합 기준선 | 학습: CAFuser 발표 레시피 + DGFusion (b) 와 같은 열화 커리큘럼(iter 90k 체크포인트에서 재개, 약 177k iter 잔여). 평가: tools/baseline_failure/robust_bench_eval.py 로 test 1897장 61케이스(clean·EMM 14·RMM 14×3·NM 3+Gaussian) 측정, 시드0 | queued | - | 학습 약 177k iter × 0.6 s/it ≈ 30 GPU-시간(3090 4장 DDP, OOM 여부 미실측 — 기동 전 스모크 필수) + 평가 약 7 GPU-시간. bengio 빈 GPU 가 P56-C 시드902 뒤에도 남을 때만 배치(P56 계열·N-RGBX 교사보다 후순위) |
| DRN-260926-54 | P101 | Q0 · E1 확정 강건성 전 프로토콜(EMM 15조합·RMM 4부분집합×3비율·NM) | E1 확정 시드821·902·903 강건성 측정(학습 0) | running | - | 미정 |
| DRN-260926-55 | P102 | Q2→Q3 강건성 전 프로토콜(EMM15·RMM4×3·NM3, Q2 먼저) | Q2·Q3 체크포인트 강건성 측정 | running | - | 미정 |
| DRN-260926-56 | P103 | Q2noDeg · 원천분리: DEGRADE_P=0, KD 유지(증류만의 순기여) | DEGRADE_P=0 (KD 유지) | running | - | 미정 |
| DRN-260926-233 | P369 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed2026090{2,3}_screen40_Q2 | Q2 시드 재현(교사 고정, 학생 시드만 변경) | running | - | 미정 |

최근 14일 종결
| id | 결론 | 판정 요약 | 분석 경로 |
| --- | --- | --- | --- |
| DRN-260926-01 | adopt | E1(중간층 4탭 읽기)·E13(4탭 + 센서별 클래스 prototype) 확정 3쌍 채택: legal v2 로 C3-only 기준선 대비 E1 Δ25 +1.21±0.81 / Δ24 +0.92±0.22, E13 +1.45±0.88 / +0.80±0.38 → 사전 게이트(Δ25 ≥ +1.0, Δ24 ≥ +0.5) 두 카드 통과. 헤드라인 카드 = E1, E13 병기. 단 SOTA 거리 게이트(우리 최고 단일 런 56.99·DGFusion 56.71)는 미달. 판정 대장 2026-09-20 행(34행) 확인 | .claude_logs/experiments/judgment-ledger.md:34 |
| DRN-260926-02 | reject | MUSES 200에폭 풀 런(E1M·E7·E13M 시드3407, 시드902 짝) 공식 재채점: E13M 3407 81.95 · E13M 902 81.68 · E1M 3407 81.70 · E1M 902 81.93 · E7(기준선 카드) 81.88 → 2시드 평균 E13M 81.81 ≈ E1M 81.82, 기준선이 그 사이. 40에폭 스크린에서 보였던 E13M 이득이 풀 런에서 사라져 MUSES 에서 4탭·prototype 카드는 이득 없음. 제출본 교체 근거 없음. 판정 대장 2026-09-20 행(36행) 확인 | .claude_logs/experiments/judgment-ledger.md:36 |
| DRN-260926-03 | inconclusive | E-LoRA 3갈래(A 센서별 r16 / B 완전공유 r16 / C 공유 r8 + 센서별 잔차 r8): 축별로 결론이 다르다. clean·모달 결측에서는 구조 무관, depth 절반 열화(RMM r=0.5)에서는 B 가 손해를 절반 이하(−6.82 vs A −15.77)로 줄이고 시드 분산도 1/5, clean 대가는 legal v2 val 약 −1. C 는 둘을 합치지 못했다. 단독 채택·기각 없이 후속 카드 E1-shared(4탭 위 완전공유 LoRA) 로 넘김 → 그 결과는 DRN-260926-57 에서 판정. 판정 대장 2026-09-21 행(49행) 확인 | .claude_logs/experiments/judgment-ledger.md:49 |
| DRN-260926-04 | reject | E17(E1 레시피 + 고해상도 세부 가지 DETAIL_BRANCH) 40에폭 스크린 시드821: legal v2 Δtest +0.88(사전 게이트 +1.0 미달), Δval +1.47. 확정 200에폭 2런은 user 지시로 취소. 이후 얇은·원거리 객체 결함을 겨냥한 R1·R2 도 실패해 세 축 모두 기각. 판정 대장 2026-09-20 행(35행) 확인 | .claude_logs/experiments/judgment-ledger.md:35 |
| DRN-260926-10 | reject | 확정 레시피의 MCubeS 이식 3시드(200에폭): 4탭 단독 3짝 평균 +0.10, 4탭 + prototype 3시드 평균 −0.63 으로 둘 다 게이트(+0.5) 미달. 짝 간 폭 0.64 가 평균 효과의 여섯 배라 시드를 더 늘려도 판정이 바뀌지 않는다고 사전 종결 조건대로 종결. MCubeS 헤드라인은 C3-off 58.07±0.49 유지. plan.md N-MC 행(2026-09-16) 확인 | .claude_logs/experiments/plan.md (N-MC 행) |
| DRN-260926-100 | reject | P39-DPC DELIVER hpca100 런(V1~V5 토글): registry 의 '학습 중' 표기는 낡은 것. 이후 P39.1-rank → P46 → E1 계보가 대체해 승계되지 않음 | - |
| DRN-260926-101 | reject | P39-DPC MUSES jarvis 런: 완주 ep146 val 81.52 / 공식 test 78.881(DRN-260926-103 분석 대상). 이후 P39.1-rank(공식 test 79.788)가 대체해 승계되지 않음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-102 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P39 조기 즉검 + DELIVER 3시점 + 모듈 시각 리포트(yeon, 2026-07-20). P39 V1~V5 토글 기여 분해의 근거 | - |
| DRN-260926-103 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P39-DPC MUSES 3모달 표준분석(yeon, 2026-07-21), ep146 val 81.52 / 공식 test 78.881. lidar 어댑터 기여와 조건별 약점의 근거 | - |
| DRN-260926-104 | reject | SAM3-RBMA(DELIVER 25클래스, SAM3 이식) 계열: '학습/디버깅 중'으로 남았고 완주·판정 기록 없음. B200 상실과 ReliaDINO(DINOv3) 전환으로 중단. 승계되지 않음 | - |
| DRN-260926-105 | reject | P38(MaskQueryLite 쿼리 헤드) DELIVER hpca100 런: val 조기 포화·test plateau 로 ep78+ 에서 중단, 표준분석(DRN-260926-106)에서 쿼리 헤드 추론 기여 no-op. MUSES 쪽 P38-m2f 는 공식 test 79.025 로 7월 기록. DELIVER 는 승계되지 않음 | - |
| DRN-260926-106 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P38-m2f DELIVER 표준분석(yeon, 2026-07-19). D1 평균 53.66(P36 −1.63·P34 −1.99), 쿼리 헤드 추론 기여 no-op → P38 DELIVER 기각의 근거 | - |
| DRN-260926-107 | adopt | P39.1-rank(라우터 랭크 변형) MUSES 3모달 시드2(jarvis): val-best epoch208 82.62, 공식 test 79.788 = 우리 MUSES 공식 최고 단일 런. 헤드라인(best 단일 런 병기)으로 채택. 2시드 평균 79.29 는 DGFusion 79.5 미만이라 '융합 계열 1위'는 단일 런 한정 | - |
| DRN-260926-108 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank MUSES 4모달(radar 추가) 시드2(hpca100): 공식 test 79.571 로 3모달 79.788 대비 −0.22. radar 추가는 이득 없음. 4모달 기록으로 보존 | - |
| DRN-260926-109 | reject | P47-MUB D-1(LiDAR 투영 밀도화, DGFusion 식 projected_to_rgb 7×7 + 모션 보정) MUSES 4모달: val 82.58(+0.23)이지만 공식 test 78.790 으로 4모달 base 79.571 대비 −0.78, 3모달 최고 대비 −1.00. val→test 낙차 3.79 = 밀도화 val 과적합 → 폐기(user 제출 2026-08-17) | - |
| DRN-260926-11 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 분산. 공식 val 3점 {82.13, 81.79, 81.47} 폭 0.66 으로 val 은 안정하나, 공식 test 2점 {79.788, 78.786} 폭 1.00 으로 val 폭이 test 에서 커진다. 결론: 단일 제출 방어 근거로 못 쓰며 MUSES 도 시드 평균 병기 필요. 원문 = analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md | .claude_logs/experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md |
| DRN-260926-110 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank MUSES 3모달 시드3(yeon) 완주 val 81.89@ep204. 5시드 val 분산 자료의 일부(시드1 82.03 등). 공식 test 미제출 | - |
| DRN-260926-111 | reject | P43-PanopticDual MUSES 3모달(hpca100): val 82.51 → 공식 test 79.351(우리 2위, 시드2 79.788 대비 −0.44). PQ 판정 불가·test 미달로 승계되지 않음. DRN-260926-52 와 같은 런 | - |
| DRN-260926-113 | reject | P44-BMR MUSES 3모달(hpca100): val-best 80.71@ep156(외부 선점으로 중단), 공식 test 는 P39.1-rank 미달. 승계되지 않음 | - |
| DRN-260926-114 | - | - | - |
| DRN-260926-115 | reject | P46-CTR C1(희소 클래스 재샘플 RCS)+C3(prototype) DELIVER: ep40 대조에서 C3-only(RailTrack 64.13, 전체 55.64)가 C1+C3(59.10, 54.92) 를 앞서 C1 은 순유해 → 2026-07-30 kill. C1 축 폐기 | - |
| DRN-260926-116 | adopt | P46-CTR C3-only(λ0.1, 탭 없음) DELIVER 본 런(jarvis, 시드 고정 기능 없던 무작위 시드): val-best ep70 → 1024 평가 val 69.44 / test 56.99 = 우리 최고 단일 런(구기록). DGFusion 66.51 / 56.71 동시 상회, MM SAM-adapter(69.60 / 57.35)에는 미달. 체크포인트 정본 NAS ckpts/P46_c3only_base_ep70_test5699_20260730/(md5 d340e3fe). 헤드라인은 이후 E1 3시드 legal v2 로 교체됐고 이 값은 단일 런 최고로 병기 | experiments/judgment-ledger.md 행 16,17 |
| DRN-260926-117 | - | - | - |
| DRN-260926-118 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P46-CTR C3-only 재현성 검증 시드2: val-best ep62 → 1024 평가 val 68.90 / test 56.40(본 런 대비 test −0.59). 방향은 재현되나 단일 런 편차 0.6 존재 → 이후 3시드 평균 병기 규칙의 근거 | - |
| DRN-260926-119 | reject | P46 C3-only 를 1024 해상도로 학습한 시드B: legal test 54.85(학습 로그 test-best 57.43 은 비legal). 768 학습 + 1024 평가(56.99)보다 낮아 1024 학습은 폐기 | - |
| DRN-260926-12 | reject | MLE-SAM 식 평균 융합 기준선(믹서를 평균으로 교체): legal test 55.45 로 우리 gated-MLP trunk(54.2~55.4)와 동급 이상. 'trunk 가 우위'라는 주장은 철회하고 믹서 3점(평균 ≈ gated-MLP > 교차 attention 만) 을 기록. 기각 = 우리 trunk 의 우위 가설 기각. 원문 = analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md | .claude_logs/experiments/analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md |
| DRN-260926-120 | reject | P46 C3-only 1024 해상도 학습 시드C: legal test 54.55. 시드B 와 함께 1024 학습 폐기 확정 | - |
| DRN-260926-121 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): C3-only 768 대표 구성(λ0.1) 진짜 시드 2런(20260815·16, jarvis). registry '학습 중' 표기는 낡은 것 — N6(DRN-260926-16) 5시드 legal-val 재선택에 포함돼 완주·평가됨(5시드 평균 54.39±0.76, 최고 시드816 55.29) | - |
| DRN-260926-122 | - | - | - |
| DRN-260926-123 | reject | P46-CTR C3-only λ=0.2 DELIVER: 완주했으나 SOTA 미달(2026-08-04 정정). λ0.1 이 대표 구성으로 확정 | - |
| DRN-260926-124 | reject | P46-CTR C3-only λ=0.2 MUSES 3모달(hpca100): val 81.65@ep136 로 base(P39.1-rank 시드2 82.62) 대비 −0.97. 손해가 clear/day 에 집중(Δclear −1.72), fog/rain 은 +0.2. DELIVER 기제의 MUSES 이식 실패 → P47 D-2(단일 모달 균형) 근거로 전환 | - |
| DRN-260926-125 | reject | P46-CTR C3-only λ=0.3 DELIVER: λ 스윕 상단 탐색, λ0.1 대표 구성을 넘지 못함(registry 핵심수치 칸). λ 스윕 종결 | - |
| DRN-260926-126 | reject | P46-CTR C3-only λ=0.2 시드2 DELIVER: 재현성 검증 실패, SOTA 미달 확정(2026-08-05). λ0.2 폐기 | - |
| DRN-260926-127 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P47-2 UniBal(모달별 독립 보조 헤드 + 단일 모달 CE) MUSES 4모달, yeon 300에폭 완주 val-best 82.06@ep164, 하네스 가드 재평가 81.72 → P52 게이트 G2 = 81.42 로 확정. 공식 test 미제출. 캘리브레이션 문서 analysis/2026-09-04-p52-unibal-calibration.md | - |
| DRN-260926-128 | reject | P46 C3-only 1024 해상도 본 런(elice B200): registry '학습 중' 표기는 낡은 것. 1024 학습은 시드B·C(DRN-260926-119·120, legal test 54.85·54.55)로 폐기 확정, 768 학습 + 1024 평가가 대표 구성 | - |
| DRN-260926-129 | reject | P46 C3-only 1024 해상도 런(jarvis GPU6,7): 위 128 과 같은 계열. 1024 학습 폐기 확정, registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-130 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P39.1-rank 1024 해상도 대조군(C3 미적용, ep126 조기 종료): 해상도 순효과 val +1.15 / test +2.01 확정 → 1024 평가 규약(legal)의 근거 | - |
| DRN-260926-131 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(P39.1-rank − event) val 82.00@ep136. DRN-260926-53 과 같은 런 | - |
| DRN-260926-132 | reject | cross-attention trunk A/B(#12, XAttnTrunk 대칭 모달 간 attention vs gated-MLP trunk, 나머지 동일): 믹서 3점 비교에서 평균 ≈ gated-MLP > 교차 attention trunk → XAttnTrunk 기각(DRN-260926-12 N2 판정과 같은 근거) | - |
| DRN-260926-133 | adopt | MCubeS 첫 진입 파일럿(P39.1-rank 통일 레시피, C3 off, yeon): 3시드 58.07±0.49 의 첫 런. DRN-260926-14 와 같은 계열, 헤드라인 채택 | - |
| DRN-260926-134 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES P39.1-rank 진짜 시드 20260824: 공식 val 81.79(시드2 82.13 대비 −0.34). N1 시드 분산 자료. 체크포인트 부재로 test 제출 불가(DRN-260926-51 참조) | - |
| DRN-260926-135 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 20260825 공식 test 78.786(2026-09-18). DRN-260926-51 과 같은 항목. 2시드 평균 79.29±0.71 | experiments/judgment-ledger.md 행 15 |
| DRN-260926-136 | adopt | P50-MAP 정렬 사전학습(동결 DINOv3-L + LoRA + trunk 의 cross-modal masked reconstruction, Places365 200k 의사 모달, yeon): 완주해 Phase1(프로브) init 이 P52 에 채택됨(P50 legal test 54.95 = P52 게이트 G1). 확장 Phase2 는 DRN-260926-32 에서 기각. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-137 | adopt | 기록 항목(채택 = 완료 확정): P50 의사 모달 생성(depth = Depth-Anything-V2-Small, lidar·event 합성, Places365 200k). P50 사전학습 입력으로 사용 완료 | - |
| DRN-260926-138 | inconclusive | P49.1-AIR(비대칭 주입, γ init 0.1) MUSES 4모달 런(yeon): 기동 검증만 기록되고 완주·판정 기록이 없다. DELIVER 쪽 P49.1 이 '57.68 SOTA' 철회 후 폐기돼 MUSES 팔도 승계되지 않음. 판정 불가로 종결 | - |
| DRN-260926-139 | inconclusive | P49.1-AIR MUSES 3모달 본 런(yeon): 기동 검증만 기록되고 완주·판정 기록이 없다. DELIVER 쪽 폐기로 승계되지 않음. 판정 불가로 종결 | - |
| DRN-260926-14 | adopt | MCubeS 첫 진입 3시드(P39.1-rank 통일 레시피, C3 off): test {57.93, 57.67, 58.62} = 58.07±0.49, published 최고 54.65 대비 +3.42, StitchFusion 55.9 대비 +2.17 → MCubeS 헤드라인으로 채택(3시드 평균 병기). 원문 = analysis/2026-08-25-n4-mcubes-first-entry-verdict.md, plan.md N4 행 | .claude_logs/experiments/analysis/2026-08-25-n4-mcubes-first-entry-verdict.md |
| DRN-260926-140 | reject | C2-MCC(masked consistency)+C3 vs C3-only A/B(hpca100 재기동, ep134 조기 종료): 1024 평가 55.32 / 768 54.32 vs 본 런 56.99 / 55.91 → Δ −1.67 / −1.59 로 C2 는 유해 확정(가설 대장 H15). C2 축 폐기. 원문 = analysis/2026-08-16-c2-mcc-ab-verdict.md | - |
| DRN-260926-141 | reject | P49.1-AIR DELIVER 768 본 런(yeon, ep150 완주): 학습 로그 test-best 58.30 은 비legal, fair-eval(val.py) test@768 55.66 으로 '57.68 SOTA' 주장 철회(2026-08-14). γ 가 0.10→0.019 로 감쇠해 주입 기제가 아니라 RGB full-FT 가 이득 원천일 가능성. 폐기 | - |
| DRN-260926-142 | reject | P49-AIR DELIVER 768 본 런(yeon): ep30 게이트① 위반(γ 정체 0.0006, val 63.83@50 열세)으로 ep54 중단(가설 대장 H13). 폐기 | - |
| DRN-260926-143 | reject | DELIVER RGB+Depth 2모달(P46 C3-only λ0.05, ep66 val-best) 1024 공정 평가: val 65.79 / test 54.14 로 SOTA RGB-D 대비 −3.81 / −3.21, 우리 4모달 대비 −2.85 → '2모달로 충분' 가설 기각. 원문 = analysis/2026-08-10-rgbd-2modal-fair-eval.md | - |
| DRN-260926-144 | reject | P48(쿼리 경로 인스턴스 감독: 클래스 단위 타깃을 연결 성분 인스턴스 단위로 전환) MUSES 4모달 제안: 게이트 미달로 폐기(2026-08-06). 이후 R2(연결 성분 손실)도 같은 축에서 실패 | - |
| DRN-260926-145 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(event 제거) P39.1-rank, 2026-08-10 미기록 상태로 발견된 완주 런. val 82.00@ep136. DRN-260926-53·131 과 같은 런 | - |
| DRN-260926-146 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): RGB+Depth 2모달 완주 체크포인트의 1024 재평가 작업(학습 0). 결과 val 65.79 / test 54.14 는 DRN-260926-143 에 기록. registry '착수' 표기는 낡은 것 | - |
| DRN-260926-147 | reject | P46 C3-only + P50-EXT Phase2 init 채택 게이트 파인튠(시드821, hpca100): ep30 legal test 53.10 < 게이트 55.25 → 기각(user 결정 2026-09-07). DRN-260926-32 와 같은 항목 | - |
| DRN-260926-148 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A(센서별 LoRA r16, 시드821 짝, yeon) 완주. 3갈래 비교의 A 팔이며 결론은 DRN-260926-03(축별로 다름, 후속 E1-shared)에 기록. registry '착수' 표기는 낡은 것 | - |
| DRN-260926-15 | adopt | MCubeS C3(클래스 prototype 손실) on 짝: 사전 등록한 예측 2/2 적중 — rubber 클래스 +9.76(18.80→28.56, published 대역 복귀), 전체 mIoU Δ −0.10(허용 범위 안) → C3 의 용량-반응 3점(DELIVER·MUSES·MCubeS) 성립. 채택 = 기제 가설 확인. 헤드라인은 C3 off 3시드 유지. 원문 = analysis/2026-08-27-n4b-dose-response-confirmed.md | .claude_logs/experiments/analysis/2026-08-27-n4b-dose-response-confirmed.md |
| DRN-260926-150 | reject | P52 RxDINO(적응형 C3·UniBal 단일 config) DELIVER 본 런 시드20260901(yeon): legal test 53.87 / val 66.17 / 24클래스 53.61. 5시드 기준선 평균 54.39 대비 미달 → P52 는 DELIVER 에서 이득 없음. 원문 = cards/2026-W37-verdicts.md ③ | - |
| DRN-260926-151 | reject | P52 RxDINO DELIVER 본 런 시드20260902(yeon): legal test 53.07 / val 66.15 / 24클래스 53.16. 2시드 평균 53.47 로 기준선 54.39 대비 −0.92 → P52 DELIVER 기각. 원문 = cards/2026-W37-verdicts.md ③ | - |
| DRN-260926-152 | reject | P52 RxDINO MCubeS 본 런 시드20260901(yeon, 200에폭 완주): val-best 58.18@174(final 57.96), C3-off 3시드 정본 58.07±0.49 대역 안 = 동률. 이득 없음 → MCubeS 헤드라인 교체 없음 | - |
| DRN-260926-154 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): N7 VICReg-off 격리(P46 C3-only λ0.1, 시드821 짝, yeon 200에폭): val-best 66.56@ep32(정본 본 런 67.79@70 대비 −1.23 트레이너 val). test-best 56.31 은 인용 금지, val-best 시점 legal test 미측정. VICReg 의 기여는 val 축에서만 확인됨 | - |
| DRN-260926-155 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 일일 카드 B0 기준선(P46 C3-only λ0.1 레시피, 40에폭 스크린 시드821, bengio): legal test 53.78 / val 64.97(체크포인트 epoch40_65.4_top1). 카드 통과선 = test ≥ 54.78. 원문 = registry 행, cards/2026-W37-verdicts.md | - |
| DRN-260926-156 | adopt | 일일 카드 E1(중간층 4탭 읽기, MODEL.TAPS per_modal [6,12,18,24]) 40에폭 스크린 시드821(bengio): legal test 54.85(B0 +1.07) 로 첫 통과 카드. 이후 확정 3쌍 채택(DRN-260926-01). 체크포인트 epoch35_67.25_top1(md5 e69d7deb) | - |
| DRN-260926-157 | reject | E0 특징 정보 프로브(원 4탭 vs 어댑터 후 vs 융합, 선형 프로브, 시드821 ckpt epoch90): test 27.8 / 35.6 / 45.1 로 게이트(i) 미달 → '어댑터가 정보를 버린다' 가설 기각. DRN-260926-25 와 같은 항목. 원문 = analysis/2026-09-08-daily-cards-E0-feature-probe.md | - |
| DRN-260926-158 | reject | E9 검증셋 사전 로짓 보정 τ∈{0, 0.5, 1.0}(학습 0): test 53.57 / 53.45 / 51.89 로 τ 가 커질수록 단조 하락, 게이트 +0.5 미달 → 폐기. τ=0 은 하네스 재동결용 대표 재채점으로 활용(636e490) | - |
| DRN-260926-159 | inconclusive | 일일 카드 E2(전 선형층 LoRA, qkv_full 등으로 확장) 40에폭 스크린 시드821(hpca100): legal val 68.38 / test 54.50, B0 대비 val +3.41 / test +0.72(회색지대 +0.5~+1.0). 이득이 RailTrack·TrafficLight 두 클래스에 몰리고 Wall·Water·Static 이 되받음. 시드 판별 대신 E12(E1+E2 결합) 가 test 54.39 로 가산성 기각되어 E2 축 종결(§5-9). 단독 판정은 회색지대로 미확정 | - |
| DRN-260926-16 | adopt | DELIVER 체크포인트 선택을 legal-val 기준으로 재선택(5시드): 평균 53.82 → 54.39±0.76, 선택 아티팩트가 2/5런에서 실재(816 +2.21, 821 +0.63), 최고 단일 런 = 시드816 55.29. 채택 = 프로토콜 확정: 이후 legal 수치는 하네스 가드 --check 필수, 체크포인트 = 학습기 val-best top1 규칙과 병기. 원문 = plan.md N6 행(2026-08-31) | .claude_logs/experiments/plan.md (N6 행) |
| DRN-260926-160 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 일일 카드 E7 = MUSES PhysAug-off 기준선(P39.1-rank 시드2, 40에폭 스크린, hpca100): 학습기 val 80.29@ep40, 공식 native val 80.08. MUSES 40에폭 카드의 짝 기준선 | - |
| DRN-260926-161 | reject | 일일 카드 E7c = PhysAug-on 대조군(E7 과 PHYSAUG 만 다름): 학습기 val 80.18@ep40, 공식 native val 79.94 로 E7(80.08) 대비 −0.13, 작은 객체 클래스에서 손해 → PhysAug-on 폐기. 이후 MUSES 공정선은 PhysAug-off 로 통일 | - |
| DRN-260926-162 | adopt | 일일 카드 E3(센서별 클래스 prototype 손실, P46.C3_PROTO.SRC permodal) 40에폭 스크린 시드821(bengio): legal test 55.18(B0 +1.40, 넷 중 최고, val 보다 test 에서 더 버는 유일한 카드) → 스크린 통과. 후속 E13(E1+E3 결합) 확정 3쌍 채택(DRN-260926-01)으로 이어짐 | - |
| DRN-260926-163 | reject | 일일 카드 E4(혼동 쌍 margin, auto_val_k5·m 0.5·λ 0.05) 40에폭 스크린 시드821(bengio): legal val 65.86 / test 53.75(B0 대비 val +0.89 / test −0.03). RailTrack 만 오르고 24클래스로 넘어오지 않음 → E4 계열 종결(2026-09-11, §5-12). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-164 | reject | 일일 카드 E4b(혼동 쌍 명시: RailTrack→Sky/Static/Terrain, Wall→Building, Water→Terrain) 40에폭 스크린 시드821(bengio): 35에폭 기준 legal test 55.14 로 한때 통과였으나 이득이 RailTrack 에만 있고 24클래스 Δ 는 게이트 미달, MARGIN 0.5 는 후기 Adam 불안정으로 죽음 → E4 계열 종결(§5-12). 후속 E4c 도 기각(DRN-260926-65) | - |
| DRN-260926-165 | reject | 일일 카드 E12 = E1(4탭) + E2(전 선형층 LoRA r32) 결합 40에폭 스크린 시드821(hpca100): legal test 54.39 로 E1 단독(54.85)보다 낮아 가산 가설 기각, E2 축 종결(§5-9). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-166 | adopt | E13(4탭 + 센서별 클래스 prototype) 확정 200에폭 시드1(jarvis): val-best epoch100 67.84(이후 100에폭 무갱신). 확정 3쌍 채택(DRN-260926-01)의 첫 시드, legal v2 재채점 완료. 체크포인트 NAS 보존(E13_s1_epoch100, md5 1517c12a) | - |
| DRN-260926-167 | adopt | E13 확정 200에폭 시드2(jarvis): val-best epoch160 68.0. 확정 3쌍 채택(DRN-260926-01)의 둘째 시드 | - |
| DRN-260926-168 | adopt | E13 확정 200에폭 시드3(jarvis): val-best epoch190 68.45. 확정 3쌍 채택(DRN-260926-01)의 셋째 시드, 판정 대장 22행 | experiments/judgment-ledger.md 행 22 |
| DRN-260926-169 | adopt | E1(중간층 4탭 읽기) 확정 200에폭 시드1(jarvis): val-best epoch140(ep138 부터 급등 68.83→68.90). DELIVER 헤드라인 3시드(test 56.24±0.42 / val 69.51±0.15)의 첫 시드이자 P56-A 의 동결 교사(E1_s1_epoch140, md5 c648f925). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-17 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): N7 VICReg-off 격리(P46 C3-only λ0.1, 시드821 짝, yeon 200에폭): val-best 66.56@ep32 로 본 런(67.79@70) 대비 학습기 val −1.23. legal test 는 val-best 시점 미측정(test-best 56.31 은 인용 금지). VICReg 기여는 val 축에서만 확인(DRN-260926-154 과 같은 런) | - |
| DRN-260926-170 | adopt | E1 확정 200에폭 시드2(jarvis, 09-15 완주): legal v2 재채점 완료, 헤드라인 3시드의 둘째(test 최고 시드 56.73). 체크포인트 NAS 보존(E1_s2_epoch134, md5 79a3ff39) | - |
| DRN-260926-171 | adopt | E1 확정 200에폭 시드3(jarvis, 09-16 완주): legal v2 재채점 완료, 헤드라인 3시드의 셋째(val 최고 시드 69.68). 체크포인트 NAS 보존(E1_s3_epoch70, md5 d8e3e8ba) | - |
| DRN-260926-172 | adopt | E13 시드2 판별 40에폭 스크린(hpca100): 완주 + legal 재채점 완료, E13 통과(§5-13 '24클래스 안정성 네 번째 확증') → 확정 200에폭 3시드 기동의 근거 | - |
| DRN-260926-173 | adopt | E13 시드3 판별 40에폭 스크린(hpca100, 09-12 완주): 세 시드 평균 +0.91(최상값 +1.29 는 대표값에서 제외) → E13 확정 런의 근거(§5-16). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-174 | reject | E14(탭 깊이 어블레이션: E13 레시피 + TAPS.LAYERS [6,12,18,24]→[4,8,12,24]) 40에폭 스크린 시드821(hpca100): legal test 54.95 / val 66.43 으로 미달 → 얕은 탭 방향을 닫음(§5-18). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-175 | reject | E15(탭 투영 어블레이션: TAPS.MODE per_modal→mean, 투영 16개→4개) 40에폭 스크린 시드821(hpca100): 미달(§5-19). 단 mean 은 '투영 수'와 '센서별 유지' 두 요인을 묶어 해석에 한계(§5-15). per_modal 유지 확정 | - |
| DRN-260926-176 | adopt | E1M = E1(4탭)의 MUSES 이식(E7 PhysAug-off 기준선 대비 TAPS 만 on) 40에폭 스크린 시드3407(hpca100): E7 대비 3짝 +0.34 / +0.80 / +0.41, 평균 +0.52 로 스크린 통과(09-15). 다만 200에폭 풀 런에서는 이득 소실(DRN-260926-02) → 40에폭 통과 기록으로만 채택 | - |
| DRN-260926-177 | adopt | E13M = E13 의 MUSES 이식(E1M + 센서별 prototype) 40에폭 스크린 시드3407(hpca100): 공식 재채점으로 게이트 ①② 통과(§5-17), 야간 악조건 손실(fog/night −1.14, rain/night −1.84 vs E1M) 관찰. 200에폭 풀 런에서는 이득 소실(DRN-260926-02) → 40에폭 통과 기록으로만 채택 | - |
| DRN-260926-178 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13M 시드2 40에폭 스크린(hpca100): 공식 native val 81.132. 시드1 의 야간 손실이 재현돼(두 시드 같은 방향, 시드2 가 더 나쁨) 센서별 prototype 손실이 MUSES 야간을 일관되게 깎는다는 관찰 확정 | - |
| DRN-260926-179 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1M 시드2 40에폭 스크린(hpca100, 09-13 기동): E13M 의 야간 손실이 prototype 탓인지 가르는 대조군. E1M 3짝 평균 +0.52 판정(DRN-260926-176)에 사용됨. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-18 | adopt | P50(정렬 사전학습 init) 런 6개 체크포인트 legal-val 스윕: ep30(학습기 top1) = legal-val-best 로 동일, 재선택 변화 없음 → P50 legal test 54.95 를 최종 확정하고 P52 게이트 G1 기준으로 채택. 원문 = plan.md N8 행(2026-09-01) | .claude_logs/experiments/plan.md (N8 행) |
| DRN-260926-180 | reject | E13Mc = E13 의 MCubeS 이식 시드3407(hpca100, 200에폭 완주): final 57.82(짝 C3-off 대비 −0.11), val-best 58.73 은 val=test 라 판정에 미사용. 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-181 | reject | E13Mc(E13 의 MCubeS 이식) 시드20260827(hpca100, 200에폭): final 56.55(짝 C3-off 대비 −1.12). 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-182 | reject | E13Mc 시드20260828(hpca100, 200에폭): final 57.99(짝 C3-off 대비 −0.63). 3시드 평균 −0.63 으로 N-MC 종결(DRN-260926-10) | - |
| DRN-260926-183 | reject | E1Mc(E1 4탭의 MCubeS 이식) 시드3407(hpca100, 200에폭): final 58.68, 4탭 단독 3짝 평균 +0.10 으로 게이트(+0.5) 미달 → N-MC 종결(DRN-260926-10). MCubeS 헤드라인은 C3-off 58.07±0.49 유지 | - |
| DRN-260926-184 | - | - | - |
| DRN-260926-185 | - | - | - |
| DRN-260926-186 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B(완전 공유 LoRA r16, 1.57M) 시드821(jarvis, 200에폭 완주). clean·결측은 A 와 같고 depth 절반 열화에서 손해가 절반 이하(−6.82 vs A −15.77) → DRN-260926-03 재판정의 근거. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-187 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C(공유 r8 + 센서별 잔차 r8, 3.93M) 시드821(yeon, 200에폭 완주 + legal 재채점): depth 열화 −11.94 로 A 와 B 사이, 둘의 장점을 합치지 못함 → DRN-260926-03 의 근거 | - |
| DRN-260926-188 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 확정 런 매칭 분모 P46 C3-only 200에폭 시드20260902(jarvis, OOM 사망 후 재기동 완주, 학습기 val 67.20@ep140). E1·E13 확정 3쌍 판정(DRN-260926-01)의 시드2 짝. 판정 대장 25행 | experiments/judgment-ledger.md 행 25 |
| DRN-260926-189 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): 확정 런 매칭 분모 P46 C3-only 200에폭 시드20260903(jarvis, 09-17 완주). E1·E13 확정 3쌍 판정의 시드3 짝. 판정 대장 25행 | experiments/judgment-ledger.md 행 25 |
| DRN-260926-190 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7(MUSES PhysAug-off 기준선) 시드20260902 40에폭 스크린(hpca100, 공식 재채점 lecun 09-15). E1M·E13M 3짝 판정의 시드2 짝 | - |
| DRN-260926-191 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7 시드20260903 40에폭 스크린(hpca100, 공식 재채점 09-15). E1M·E13M 3짝 판정의 시드3 짝 | - |
| DRN-260926-192 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1M 시드20260903 40에폭 스크린(hpca100): ep35 평가 직후 디스크 가득(Errno 28)으로 사망, 판정은 ep35 val-best 80.86 으로(생각정리 09-15, E4b 전례). 3짝 평균 +0.52 판정(DRN-260926-176)에 포함 | - |
| DRN-260926-193 | reject | E1M(E1 4탭의 MUSES 이식) 200에폭 풀 런 시드3407(hpca100, Errno 28 후 ep38 부터 이어 완주): 공식 val 81.70 으로 E7 풀 런(81.88)을 넘지 못함 → 40에폭 이득이 풀 런에서 소실(DRN-260926-02). test 제출 후보에서 제외 | - |
| DRN-260926-194 | reject | E13M(E13 의 MUSES 이식) 200에폭 풀 런 시드3407(hpca100): 공식 val 81.95 로 5런 중 최고이나 시드2 81.68 과 평균 81.81 ≈ E1M 81.82, E7 기준선(81.88)이 그 사이 → 이득 없음(DRN-260926-02). 제출본 교체 근거 없음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-195 | reject | E13M 200에폭 풀 런 시드20260902(hpca100): 공식 val 81.68. 2시드 평균 81.81 로 기준선과 동급 → 이득 없음(DRN-260926-02) | - |
| DRN-260926-196 | reject | E1M 200에폭 풀 런 시드20260902(hpca100): 공식 val 81.93. 2시드 평균 81.82 로 기준선(81.88)과 동급 → 이득 없음(DRN-260926-02) | - |
| DRN-260926-197 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): E7 기준선 200에폭 풀 런 시드3407(jarvis, 09-18 완주): 공식 val 81.88. E1M·E13M 풀 런과 같은 길이의 짝이며 MUSES 풀 런 판정(DRN-260926-02)의 분모 | - |
| DRN-260926-198 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13M 시드20260903 40에폭 스크린(hpca100, 09-16 완주·공식 재채점). E13M 3짝 판정의 셋째 짝. 40에폭 통과는 풀 런에서 소실(DRN-260926-02) | - |
| DRN-260926-199 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드3407 = MCubeS 같은 코드 기준선(TAPS off·C3 off, hpca100 200에폭). E1Mc·E13Mc 재판정(N-MC)의 분모 | - |
| DRN-260926-200 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드20260827(hpca100 200에폭). N-MC 판정의 분모 | - |
| DRN-260926-201 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0Mc 시드20260828(hpca100 200에폭). N-MC 판정의 분모 | - |
| DRN-260926-202 | reject | E1Mc 시드20260827(hpca100, yeon 무효 시드의 재실행, 200에폭 완주): 4탭 단독 3짝 평균 +0.10 으로 게이트 미달 → N-MC 종결(DRN-260926-10) | - |
| DRN-260926-203 | reject | E1Mc 시드20260828(hpca100, 200에폭 완주): 4탭 단독 3짝 평균 +0.10 으로 게이트(+0.5) 미달 → N-MC 종결(DRN-260926-10) | - |
| DRN-260926-204 | - | - | - |
| DRN-260926-205 | - | - | - |
| DRN-260926-206 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A(센서별 r16) 시드902(bengio, 200에폭). 3갈래×3시드 9칸 판정(DRN-260926-03)의 A 팔 시드2. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-207 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B(완전 공유 r16) 시드902(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 B 팔 시드2: depth 열화 손해 −6.82±0.57 의 자료 | - |
| DRN-260926-208 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm A 시드903(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 A 팔 시드3: depth 열화 손해 −15.77±2.73 의 자료 | - |
| DRN-260926-209 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm B 시드903(bengio, 200에폭). 9칸 판정(DRN-260926-03)의 B 팔 시드3 | - |
| DRN-260926-210 | - | - | - |
| DRN-260926-211 | - | - | - |
| DRN-260926-212 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1 확정 시드4 v2(yeon 새 체크아웃, 20260904): 완주·재채점 완료. DELIVER 넷째 짝이며 헤드라인은 3시드 유지(4시드 평균은 감시 세션 std 회신 후 갱신 — 판정 대장 39행). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-213 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): C3-only 200에폭 시드4(yeon 새 체크아웃, 20260904): E1·E13 시드4 짝의 같은 코드 분모. 완주·재채점 완료 | - |
| DRN-260926-214 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E13 확정 시드4(yeon, 20260904): legal test 55.67 / val 69.40(ep110) → E13 4시드 test {56.95, 55.21, 57.29, 55.67}. 헤드라인은 3시드(56.48) 유지, 4시드 평균은 std 회신 후 갱신. 판정 대장 39행 | experiments/judgment-ledger.md 행 39 |
| DRN-260926-215 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C(공유 r8 + 잔차 r8) 시드902(yeon, 옛 코드 버전 각주). 9칸 판정(DRN-260926-03)의 C 팔 시드2: depth 열화 −11.94±2.79 의 자료. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-216 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E-LoRA arm C 시드903(jarvis, 09-18 완주). 9칸 판정(DRN-260926-03)의 C 팔 시드3 | - |
| DRN-260926-217 | reject | E2(전 선형층 LoRA) 시드2 짝 40에폭 스크린(hpca100, B0s2 짝): 회색지대 판별용이었으나 E12(E1+E2 결합) 가산성 기각으로 E2 축이 09-11 종결(§5-9) → 단독 채택 없음. registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-220 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E3 legal test 재채점(ckpt epoch25_65.97_top1, bengio): test 55.18(B0 +1.40, 카드 최고), val 은 bengio 양도로 미측정. E3 스크린 통과(DRN-260926-162)의 원자료 | - |
| DRN-260926-221 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): E1 legal 재채점(ckpt epoch35_67.25_top1, bengio): val 66.90 / test 54.85(B0 +1.07), 전이율 55%. E1 스크린 통과(DRN-260926-156)의 원자료 | - |
| DRN-260926-222 | reject | E4 legal 재채점(ckpt epoch15_65.92_top1, bengio): val 65.86 / test 53.75(B0 −0.03) → E4 폐기(DRN-260926-163)의 원자료 | - |
| DRN-260926-223 | - | - | - |
| DRN-260926-226 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): B0 legal 재채점(ckpt epoch40_65.4_top1, bengio): val 64.97 / test 53.78 → 40에폭 카드 통과선 test ≥ 54.78 확정. DRN-260926-155 의 원자료 | - |
| DRN-260926-227 | - | - | - |
| DRN-260926-228 | reject | E17(4탭 + 고해상도 세부 가지) 40에폭 스크린 시드821 hpca100 사본(09-16 기동): jarvis 사본과 같은 카드. Δtest +0.88 로 게이트 미달, 확정 런 취소(DRN-260926-04). registry '학습 중' 표기는 낡은 것 | - |
| DRN-260926-229 | - | - | - |
| DRN-260926-23 | adopt | 측정 항목(채택 = 공정선 기록으로 확정): MUSES 최선 레시피(P39.1-rank 3모달)에서 PhysAug 만 끈 시드 3짝(824·825·826): 공식 val 82.463 / 82.19 / 82.256 = 82.30±0.14. PhysAug-on 같은 레시피 시드2 82.62 대비 −0.3 안. 공정선 헤드라인 후보이며 test 제출은 user 결정으로 보류(2026-09-29, 새 4모달 모델 전까지). 판정 대장 2026-09-24·25 행 | experiments/judgment-ledger.md 행 74,75,76 |
| DRN-260926-230 | - | - | - |
| DRN-260926-231 | adopt | Q2(두 패스 = clean CE + 열화 CE + 동결 E1 교사 증류) 를 이후 카드의 바탕 레시피로 채택: clean 축은 3시드 24클래스 평균 Δ +0.41 로 사전 기준(+0.6) 미달이라 clean 개선 주장은 하지 않음(손해도 없음). 강건 축은 3시드 재현: 모달 결측 15조합 평균 48.06 vs E1 스크린 40.40(+7.7), depth 부분 결측 저하 비율 0.26, S&P 노이즈 d.2 저하 약 3.8 감소. RGB 부분 결측에는 이득 없음. 헤드라인 교체 없음. 판정 대장 2026-09-28·09-29 행 | experiments/judgment-ledger.md 행 71 |
| DRN-260926-232 | reject | Q3(품질 헤드 QAF 본 카드) 종료: 시드821 legal v2 에서 Q2(품질 헤드 없는 두 패스+교사 증류) 대비 24클래스 −1.31·얇은 객체 4클래스 −5.67, 강건 축도 모달 결측 15조합 평균 47.09 vs Q2 48.68 로 Q2 이하. 사전 규칙(강건에서 Q2 를 못 넘으면 종료) 적용. 판정 대장 2026-09-24·09-29 행 | experiments/judgment-ledger.md 행 51,53,59,73 |
| DRN-260926-235 | adopt | det(poongsan, 국책 목표 mAP50 0.85) P29-Det egofill 데이터(train 11,799, bengio 50에폭): best mAP50 0.8501@ep9 로 목표 달성(2026-07-05, 동일 스택에서 데이터만으로 0.4455 → 0.850). det 트랙 목표 달성 기록으로 채택 | - |
| DRN-260926-236 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P29-Det 모달 ablation(img/event/thermal, egofill 11,799, bengio) 완주. 결론은 registry 모달 ablation 행과 메모리 det-final-ann-modality-ablation(RGB 단독 ≥ 3모달 on mAP50)에 기록 | - |
| DRN-260926-237 | inconclusive | P29-Det 최종 annotation 풀 런(det_P29_final_full): 2026-07-08 이후 상태 미갱신·결과 기록 없음. det 목표는 egofill 런(DRN-260926-235)과 D1 계열로 달성됐으므로 재실행 없이 미확정 종결 | - |
| DRN-260926-238 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): P29-Det poongsan_v2(clean label) 재학습(jarvis): 라벨 정정 후 스택 기준선. 원문 = det/p29det-data-fix.md | - |
| DRN-260926-239 | inconclusive | P31.1-Det(캘리브레이션 신뢰도 + decisive 라우터 + FCOS) poongsan v3clip 비공식 분할: 완주했으나 v2 분할과 직접 비교 불가(재평가 과제 D2 미수행). det 목표는 P29-Det egofill·D1 계열로 달성돼 재평가 없이 미확정 종결 | - |
| DRN-260926-240 | reject | P30-Det(라우터 + 쿼리 디코더) poongsan_v2: 소물체 붕괴로 mAP50 0.228 vs P29-Det 0.446 → dead. 비교 리포트 /mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/ | - |
| DRN-260926-241 | adopt | 측정 항목(채택 = 외부 기준점 기록으로 확정): YOLO11m RGB 단독(외부 헤드, hinton) poongsan label-v3 mAP50 0.864. 우리 det 계열의 외부 비교점 | - |
| DRN-260926-242 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): ProbeA2 백본 스케일링(동결 DINOv3 S+/B/L/H+/7B + 공용 경량 헤드, MUSES RGB 단독) 5점 완료. 축 분리 게이트 §⑧ 정식 판정 완료(백본 크기 축과 융합 축 분리 근거). 캐시 55.9G 는 2026-09-30 삭제(재생성 가능) | - |
| DRN-260926-243 | adopt | P47-2 UniBal(모달별 독립 보조 헤드 + 단일 모달 CE, MUSES 4모달): val-best 82.06, 하네스 가드 재평가 81.72 → P52 게이트 G2 기준(81.42)으로 채택. 공식 test 미제출. 기법 단위 노벨티는 아님(선행 UMT·OGM-GE) | - |
| DRN-260926-244 | adopt | P46 CTR(클래스 전이 복구: C1 희소 클래스 재샘플 + C2 masked consistency + C3 클래스 prototype): C1 은 순유해, C2 는 유해(Δ −1.67), C3 만 유효해 C3-only(λ0.1)가 단일 런 최고 test 56.99 → C3 prototype 손실을 현행 레시피의 일부로 채택(부분 채택). SOTA·재현성 검증은 미달 | - |
| DRN-260926-245 | reject | P43 PanopticDual(독립 주손실 mask-classification 헤드, MUSES): val 82.51 → 공식 test 79.351 로 P39.1(79.788) 미달, PQ 는 정답 부재로 판정 불가 → 승계되지 않음(DRN-260926-52·111) | - |
| DRN-260926-246 | reject | P44 BMR(균형 멀티모달 신뢰도) + P45 FogStyle: BMR 은 완전 실패 + 가설의 test 반증, FogStyle 은 미실행. 세대 종결 | - |
| DRN-260926-247 | adopt | P38 MaskQueryLite(Mask2Former-lite 쿼리 헤드): DELIVER 에서 모듈 자체는 실패(D1 평균 53.66, 추론 기여 no-op)이나 부산물 두 개 성공 — MUSES 공식 test 79.025(7월 기록)와 쿼리 헤드 구조가 현행 레시피(M2F)에 남음. 부분 채택 | - |
| DRN-260926-248 | reject | P39 DPC(이중 경로 경쟁): 기제는 성공, 성능 전환은 실패. 부수 결과: gate/calibration 이 얇은 클래스에 유해(off 시 +35.9/+26.0)로 재판정 → GATE·CALIBRATION 영구 off | - |
| DRN-260926-249 | adopt | P39.1 Rank 수리(gated_mlp trunk + VICReg): 기제 실증 성공 + MUSES 내부 최고(공식 test 79.788) → 현행 기준선으로 채택. 경고: 시드 분산 0.92 가 대부분의 델타보다 커 0.1~0.3 차이 판정은 재검토 대상(시드 평균 병기 규칙의 출발) | - |
| DRN-260926-25 | reject | E0/S0 특징 프로브(학습 = 선형 헤드만, DINOv3 중간층 원 특징 vs 어댑터 후 특징 vs 융합 특징): 원 특징 27.8 < 어댑터 후 35.6 < 융합 45.1 로 '원 특징만으로 충분하다'는 가설 기각. depth 중간층에 Water 47.6·RailTrack 23.9 정보가 남아 있어 중간층 읽기(E1)·prototype(E3) 카드는 유지, E5·E6 은 우선순위 하향. 원문 = cards/2026-W37-verdicts.md E0 행, plan.md 09-07~08 절 | .claude_logs/experiments/plan.md (직전 완결 09-07~08 절) |
| DRN-260926-250 | inconclusive | P40 RCA-Fusion(신뢰도 조건부 감쇠): 미실행(보류). P39 분석에서 gate/calib 이 유해로 재판정돼 착수 근거가 사라짐. 판정 불가로 종결 | - |
| DRN-260926-251 | reject | P32 CoRB(입증 편향 메모리 attention, DELIVER): val 64.12(계보 최고) / test 55.00(P28 −0.27), attention bias 는 유의한 순손해(p=4.5e-22) → 실패이자 계보에서 가장 결정적인 반증(attention bias 축 재시도 금지) | - |
| DRN-260926-252 | inconclusive | P31(캘리브레이션 이중 신뢰도 RBMA + 다중 스케일 클래스 토큰 디코딩, DELIVER B200): 성능 정체, 기제 2건은 성공(arch-evolution ④). 수치 이득 없이 기제 자료만 남김. SAM2 계보 종료로 승계되지 않음 | - |
| DRN-260926-253 | reject | P30-Det(P30 백본의 검출 확장: 신뢰도 라우터 융합 + 객체 쿼리 디코더 + FCOS 보조): 소물체 붕괴로 mAP50 0.228 vs P29-Det 0.446 → 실패(DRN-260926-240 과 같은 런) | - |
| DRN-260926-254 | reject | P30(클래스 토큰 디코더 + 신뢰도 앵커 학습 라우터, DELIVER): val 49.76 / test 44.10 으로 P29 대비 −13.4 / −10.2 → 명백한 실패(원인 귀속은 뒤집힘: 디코더 아닌 라우터 문제). arch-evolution ④ | - |
| DRN-260926-255 | reject | P29 SDC(자기 유도 조건 라우팅, DELIVER B200): val 63.20@ep100 / test 54.34@ep146 로 목표(66.51 / 56.71) 대비 −3.31 / −2.37 → 정체(미달). 비교 기준선으로만 남음(DRN-260926-95) | - |
| DRN-260926-256 | inconclusive | P28 RBMA(신뢰도 편향 메모리 attention, DELIVER B200): val 57.87 / test 50.61@ep12 에서 ep16 사망(조기 종료) → 불완전 실패. 신호 정의는 계보 표준이 됐으나 성능 판정에 필요한 완주가 없음. 이후 P29·P31·P32 에서 attention bias 효과 ≈0 으로 반증됨 | - |
| DRN-260926-257 | reject | P8(ConfidenceHeadV2 + Sigmoid UAMM, MULTIAQUA): M-score 77.5~78.5(증강별) 로 P9(81.47)에 대체됨. 종료 트랙 | - |
| DRN-260926-258 | adopt | P9(CrossModalFusionHead + Max-Norm UAMM, MULTIAQUA): M-score 82.10(재제출 #16710, val 93.29 / test 70.91) = 챌린지 공동 1위 제출본. SAM2 계보의 최선으로 채택(종료 트랙, DRN-260926-87 과 같은 런) | - |
| DRN-260926-259 | - | - | - |
| DRN-260926-26 | - | - | - |
| DRN-260926-260 | - | - | - |
| DRN-260926-261 | reject | P12(입력 조건부 Soft MoE LoRA, MULTIAQUA): M-score 80.80(P9 대비 −0.67) → 기각. 동적 융합 계열 실패 패턴의 시작 | - |
| DRN-260926-262 | reject | P13(Energy Score 융합 + expert collapse 수정, MULTIAQUA): M-score 81.21(P9 대비 −0.26) → 기각 | - |
| DRN-260926-263 | reject | P14(모달별 독립 보조 디코더, MULTIAQUA): M-score 74.27(P9 대비 −7.20) → 기각. 원인 = 동결 백본의 보조 마스크 품질 부족(ISSUE-008) | - |
| DRN-260926-265 | reject | P16(캘리브레이션 공간 엔트로피 융합, P15 설계 구현판, MULTIAQUA): M-score 68.42(P9 대비 −13.05) → 기각. 픽셀 단위 융합이 보조 마스크 오차를 증폭 | - |
| DRN-260926-266 | reject | P17(다중 스케일 FPN 보조 디코더 + 공간 엔트로피 융합, MULTIAQUA): M-score 73.23(P9 대비 −8.24) → 기각. 'P14~P17 동적 융합의 실패' 종합 분석(log.md)으로 축 종결 | - |
| DRN-260926-267 | inconclusive | P18(학습 가능한 ResNet-18 보조 백본 + 구성 가능한 융합, MULTIAQUA): 설계 절만 있고 log·registry 에 완주 수치가 없다. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-268 | reject | P19(학습된 공간 교차 모달 융합 헤드, MULTIAQUA): hardaug5 ep36 M-score 69.63 로 P9 대비 크게 미달 → 기각 | - |
| DRN-260926-269 | inconclusive | P20(공유 MLP 게이트 + 고랭크 MoE, 실험 J-A, MULTIAQUA): log 에 '학습 대기'로만 남아 완주 수치 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-270 | reject | P21(DeBA-FP 변형 가능 병목 어댑터, 실험 K, MULTIAQUA): M-score 81.77(#16792) 로 P9·P22(82.10) 미달 → 최종 제출본 아님(DRN-260926-89) | - |
| DRN-260926-271 | adopt | P22(다중 스케일 DeBA-FP, 실험 L, MULTIAQUA): M-score 82.10(#16932, val 93.42 / test 70.77) = P9 와 공동 1위 제출본으로 채택(종료 트랙, DRN-260926-88) | - |
| DRN-260926-273 | inconclusive | P24(P9 + 품질 인식 메모리 게이팅, 모달별 디코더 증류, 실험 N, MULTIAQUA): ep36 시점 sigmoid 버그(ISSUE-013)로 교사 신호가 잘못돼 미제출, 수정 후 완주 기록 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-276 | inconclusive | P27(교차 모달 메모리 attention 의 가산 attention bias, RBMA 전구체, 2026-04-14): 기구로는 성립했으나 성능 판정 불가(arch-evolution ④ 판정). 이후 P28 RBMA 로 승계 | - |
| DRN-260926-277 | inconclusive | P33 CG-MoD(역량 게이트 hard 융합 + 모달 드롭아웃): P33.1 조기 중단(이득 0)·P33.2 ep52 무효과 뒤 P34 로 피벗, 완주 수치·판정 기록 없음 → 미완결(기록 소실). P33.3 은 구현 없는 placeholder | - |
| DRN-260926-278 | adopt | P34 ReliaDINO(동결 DINOv3-L + 센서별 LoRA, 계보 전환점, 2026-07-13): legal val 68.19 / test 56.62 로 계보 최대의 성공. 단 성공 원천은 백본 + LoRA 이지 제안 모듈(RBMA attention bias ≈0)이 아님. 이후 모든 세대의 바탕으로 채택 | - |
| DRN-260926-279 | adopt | P35 공정 레시피 동결(P34 − ATTN_BIAS − CONSISTENCY − PhysAug): 의도대로 공정선 확립, 성능은 −1.12. PhysAug 를 켠 수치는 헤드라인·게이트에 쓰지 않는다는 규칙(user 2026-07-20)의 근거로 채택 | - |
| DRN-260926-28 | - | - | - |
| DRN-260926-280 | adopt | P36 클래스별 신뢰도 앵커 라우터(= P35 + router): 유일하게 살아남은 모듈이나 증분은 잡음 대역(test +0.10). 현행 레시피의 ROUTER 로 유지(채택)하되 '새 모듈 + SOTA' 논문은 성립하지 않는다는 판정 기록 | - |
| DRN-260926-281 | reject | P37a CEFR-Head(클래스 기대 특징 라우팅)·P37b ClassToken-lite: 둘 다 실패(방식은 다름 — CEFR 은 라우팅 분화 실패 0/19, ClassToken 은 성능 미달). CEFR 라우팅은 재시도 금지 축 | - |
| DRN-260926-282 | reject | P41 FCR(융합 스펙트럼 붕괴 / 융합 클래스 정렬 규제): 결정적 기각(airtight falsification), 계보에서 가장 깨끗한 음성 결과 | - |
| DRN-260926-283 | reject | P42 lidar 강제(조건부 균형 img 마스킹): 전 FRAC 에서 미달, 단조 열화 → 실패 | - |
| DRN-260926-285 | reject | P47-1 LiDAR 투영 밀도화(구 D-1, MUSES 4모달): 공식 test 78.790 으로 base 79.571 대비 −0.78, val 만 오르고 test 역행 → 음성 결과 확정(효과 없음) | - |
| DRN-260926-286 | reject | P48 쿼리 경로 인스턴스 감독(제안 단계, MUSES 4모달): 게이트 미달로 폐기(2026-08-06, DRN-260926-144) | - |
| DRN-260926-290 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): det 인증 배포 계열 D1(ViT-S/S+/B 백본, poongsan). ViT-S+ 가 drone-jemo(RTX 5090) 인증 배포본이며 GPU 19.43 fps, 야간 mAP50 0.8292 검증. 국책 목표 mAP50 0.85 는 D1 스윕에서 달성(status 표 0.9321). 원문 = det/det-cert-D1-realtime.md, 브랜치 26-drone-certificate | - |
| DRN-260926-291 | adopt | P29-Det(RBMA 백본 + FPN/FCOS 검출, poongsan): egofill 데이터로 mAP50 0.8501@ep9 달성(2026-07-04, 국책 목표 0.85). det 트랙의 목표 달성 계열로 채택(DRN-260926-235) | - |
| DRN-260926-292 | inconclusive | det 백본 이식 계열(P34~P39-Det, RF-DETR, P9 base; poongsan): config 12개만 남아 있고 registry·판정 대장에 결과 기록이 없어 판정 불가. det 트랙은 D1 계열로 목표(mAP50 0.85)를 달성했으므로 이 계열은 재실행하지 않고 미확정으로 종결 | - |
| DRN-260926-293 | adopt | 기록 항목(채택 = 계보 기록으로 확정): MemorySAM 초기 기준선(SAM·LoRA-SAM·P4~P7·재구성). SAM2 계보(종료 트랙 MULTIAQUA)의 출발점. 상세 = models/arch-evolution.md '공통 기반' 절 | - |
| DRN-260926-294 | reject | 실험 I P9 Gamma TTA(MULTIAQUA, softmax 확률 평균): 실패 — 기준선 대비 이득 없음. 종료 트랙 | - |
| DRN-260926-295 | reject | 실험 II·III I2I 변환(test 야간 RGB 를 img2img-turbo 로 주간화 뒤 추론, MULTIAQUA): 실패. 종료 트랙 | - |
| DRN-260926-296 | reject | 실험 IV P9 CV 휴리스틱 보정(worst 56장 밝기·대비 보정 뒤 재추론, MULTIAQUA): test mIoU −4.76, M-score −2.39(79.08) → 실패(잘 맞던 이미지까지 보정됨). 종료 트랙 | - |
| DRN-260926-297 | - | - | - |
| DRN-260926-298 | adopt | 실험 VI P9 PhysAug(물리 기반 증강, MULTIAQUA): hardaug8+physaug 가 P9 최선 M-score 82.10 의 레시피가 됨 → 종료 트랙에서 채택. 단 논문 트랙(DELIVER·MUSES)에서는 PhysAug 를 헤드라인에 쓰지 않는다(P35 규칙) | - |
| DRN-260926-299 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): P9 증강 ablation 계열(hardaug2~8·night2·noCRM·tta, MULTIAQUA): hardaug8 이 최선(M 81.98, physaug 결합 시 82.10), Night Aug 는 포화. 종료 트랙 | - |
| DRN-260926-300 | inconclusive | Tiled Inference(원본 해상도 슬라이딩 윈도우 추론, MULTIAQUA, 2026-03-17): log 에 '대기 중 — 제출 후 비교 필요'로만 남아 결과 없음. 판정 불가로 종결(종료 트랙) | - |
| DRN-260926-301 | adopt | 측정 항목(채택 = 진단 기록으로 확정): P25 공간 품질 게이팅의 야간 예측 품질(Pred Q) 분석(2026-06-14). SAM2 계보 품질 게이팅 실패 원인 자료 | - |
| DRN-260926-302 | adopt | 측정 항목(채택 = 기록으로 확정): P26 DELIVER AMP on + gradient checkpoint off 단일 GPU 메모리 프로브(2026-04-10). 메모리 예산 자료 | - |
| DRN-260926-303 | adopt | 측정 항목(채택 = 분석 기록으로 확정): 아키텍처 심층 분석 A·B(UAMM 실효성, P9 vs MMSamBase, 2026-03-24). 'SAM2 메모리 attention 이 이미 암묵적 교차 모달 적응을 한다'는 P9 우위 해석의 근거 | - |
| DRN-260926-304 | inconclusive | P39.1-rank MUSES 3모달 시드4(yeon): monitor-log 에 val 81.92@ep198 정체·약 277/300 에폭에서 '저신뢰, 미완주'로만 남아 있고 config 도 레포에 없다. 완주 기록·체크포인트·재채점이 없어 판정 불가. MUSES 시드 통계는 시드2·824·825·826 으로 대체됨 | - |
| DRN-260926-305 | inconclusive | P44-BMR DELIVER 시드2(yeon): monitor-log 에 '미완주'로만 남아 있고 결과·체크포인트 기록 없음. 판정 불가. P44 계열은 이후 P46/E1 계보로 대체됨 | - |
| DRN-260926-306 | inconclusive | P34 MUSES 4모달 + DGFusion 투영 변형: monitor-log 에 이름만 1회 언급되고 결과 기록이 없어 판정 불가. 투영 정합 자체는 DRN-260926-47 에서 '성능 이득 0, 공정성만 확보'로 확정됨 | - |
| DRN-260926-307 | adopt | 기록 항목(채택 = 이력을 기록으로 확정): MUSES Codabench 공식 test 제출 11건의 이력. 최고 = P39.1-rank 시드2 3모달 79.788, 둘째 시드 20260825 78.786(2시드 평균 79.29). 제출 zip 규약 위치 = /ailab_mat2/.../submission/muses/. 새 제출은 user 만 하며 2026-09-29 user 결정으로 새 4모달 모델 전까지 보류 | - |
| DRN-260926-308 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): DGFusion 기준선 재현 평가. DELIVER 재학습 (a) val-best 80k val 66.54 / test 55.68(final-iter 65.62 / 55.56, 공개 66.51 / 56.71 대비 −1.15), MUSES 는 공개 가중치 val 재현이 감시 세션에서 진행 중. 우리 P39.1 과의 비교는 판정 대장 2026-09-19~23 행에 기록 | - |
| DRN-260926-32 | reject | P50-EXT(정렬 사전학습 코퍼스 500k·등-스텝 확장, Phase2 375,000 step 완주): EXT-init 파인튠 ep30 legal test 53.10 < 채택 게이트 55.25(= 프로브-init 54.95 + 0.3) → 기각, P52 init 은 Phase1(프로브) 로 확정. user 결정 2026-09-07. 원문 = registry hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 행, decisions/2026-08-31-p52-rxdino-adaptive-amendment.md | .claude_logs/experiments/registry.md (hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 행) |
| DRN-260926-39 | - | - | - |
| DRN-260926-40 | - | - | - |
| DRN-260926-41 | - | - | - |
| DRN-260926-42 | - | - | - |
| DRN-260926-43 | - | - | - |
| DRN-260926-44 | - | - | - |
| DRN-260926-45 | adopt | 측정 항목(채택 = 진단 결과를 기록으로 확정): MUSES 성능 저하 원인 A/B 격리. radar(또는 4모달 구조)가 원인이고 lidar 재투영·event dilation·eff batch 는 무관. Arm A ep24 best 73.85@ep18 은 대조군 ep10 74.24 에 앞서지 않아 DGFusion 식 투영은 중립. 원문 = plan.md ✅ 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-46 | reject | TTA(다중 스케일·플립 시험 시 증강)를 헤드라인에 쓸 수 있는가: 경쟁 논문 3종(CMNeXt·CAFuser·DGFusion) 모두 미사용(CMNeXt 는 논문에 single-scale 명시) → 헤드라인 사용 불가로 기각. 우리 MSF config 는 쓰인 적 없어 과거 수치 오염 없음. 원문 = plan.md 완료·판정 표, monitor-log.md TTA 조사 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-47 | adopt | 측정·정합 항목(채택 = 결과를 기록으로 확정): DGFusion 전처리 파라미터 재현 완료(공개 PIXEL_MEAN 을 오라클로 −0.1% 안에서 적중). 실제 남은 차이는 lidar 투영뿐(radar·event 30ms 는 이미 동일). 성능 이득은 0 이고 공정 비교 조건만 확보. 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-48 | reject | SAM2 계보 제안 모듈 ablation: ATTN_BIAS(RBMA 간판 기제) 포함 제안 모듈 전부 효과 ≈0, gate+calibration 만 test +0.26. 성능 출처 = DINOv3 백본 + 센서별 LoRA 로 확정 → 제안 모듈들의 기여 가설 기각(이후 재시도 금지 축의 근거). 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-49 | adopt | 측정 항목(채택 = 진단 결과를 기록으로 확정): det(poongsan) 학습 붕괴 원인 = BS1 의 gradient 잡음(양성 앵커 1~3개), LR 은 원인 아님. 처방 = 배치 증가 + LR 유지, warmup 5에폭 완주로 검증. 원문 = plan.md 완료·판정 표 | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-50 | inconclusive | seg P37a/P37b(bengio 분): bengio 노드 CUDA 전체 장애로 ep1~2 에서 종료된 런 소실. 결과 없음이라 판정 불가. jarvis 재기동분(P37a→P37b 체인)이 계보를 승계함. 원문 = plan.md 완료·판정 표, monitor-log.md | .claude_logs/experiments/plan.md (✅ 완료·판정 표) |
| DRN-260926-51 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES 시드 20260825 공식 test 78.786(2026-09-18 수신). 시드2 79.788 대비 −1.00, 2시드 평균 79.29±0.71 < DGFusion 79.5 → '융합 계열 1위' 서술은 시드2 단일 런 한정으로 제한. 시드 20260824 는 체크포인트 부재로 제출 불가. 원문 = analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md | .claude_logs/experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md |
| DRN-260926-52 | inconclusive | P43-PanopticDual(MUSES, 분할 헤드에 panoptic 이중 헤드 추가): 학습기 val 82.51@ep156 로 완주했으나 MUSES 에 panoptic 정답이 없어 PQ 판정이 불가하고 공식 test 제출도 하지 않았다. 이후 MUSES 헤드라인은 P39.1-rank(공식 test 79.788)로 확정돼 P43 은 승계되지 않음. 판정 근거 부족으로 미확정 종결 | - |
| DRN-260926-53 | adopt | 측정 항목(채택 = 결과를 기록으로 확정): MUSES RGB+LiDAR 2모달(P39.1-rank) 완주 val 82.00@ep136. 같은 레시피 3모달(RGB+Event+LiDAR) val 82.22 와 0.22 차이로, MUSES 에서 event 의 기여가 작다는 근거. 공식 test 제출은 하지 않았고 재실행 금지. 원문 = plan.md 2026-08-10 절, registry 행 | - |
| DRN-260926-57 | inconclusive | E1-shared(E1 4탭 레시피 + 전 센서 공유 LoRA r16) 40에폭 스크린 3시드: 게이트 1차(clean 24클래스 Δ ≥ −0.3) 평균 −0.24 로 통과(여유 0.06, 얇은 객체 4클래스 −2.10), 게이트 2차(depth 부분 열화 저하가 E1 의 절반 이하) 비율 0.47 로 통과. 그러나 clean 분산이 커지고(s902 붕괴형 54.46) 효과가 depth 한 모달에 한정돼 단독 채택하지 않음. P56-C(공유 + 센서별 LoRA 전문가 혼합)의 재료로 넘김. 판정 대장 2026-09-23·24 행 | experiments/judgment-ledger.md 행 56,66,68,69,77 |
| DRN-260926-59 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): 기준선(DGFusion·CAFuser)과 우리 모델의 실패 분석 D4/D6. 핵심 결과: 우리 depth 의존은 기준선의 절반(모달 zero-out −17.6 vs −31~−33), 기준선은 event·LiDAR 를 어떤 조건에서도 쓰지 않음(제거 시 ±0.4), 원거리 얇은 객체 결함(찾고도 못 그리는 비율 거리3 0.44 vs 기준선 0.25)이 우리 주 손실 요인. 이 결과가 Q2(두 패스 학습)·P56 설계의 출발점. 원문 = 판정 대장 2026-09-19~20 행 8개, analysis/2026-09-17-baseline-failure-analysis-plan.md | experiments/judgment-ledger.md 행 28,29,30,31,32,33,37,38 |
| DRN-260926-60 | reject | Q1(품질 헤드 실현성 프로브) 기각: Q1b-2 에서 RGB 광학 열화(흐림·감마·색 틀어짐) AUROC 0.67~0.73 으로 인식 실패, Q1d 에서 합성 열화로 학습한 헤드가 실제 열화로 전이되지 않음(다중 블록 혼합 헤드의 실제 야간 AUROC 0.002). 합성 열화 라벨로 감독하는 품질 헤드 계열은 재시도 금지. 판정 대장 2026-09-23·09-28·09-29 행 | experiments/judgment-ledger.md 행 43,50,61 |
| DRN-260926-61 | reject | R1(depth 경계 prior refinement) 종료: 40ep 스크린 3시드 legal v2 test 평균 56.61(E1 스크린 짝 56.58), 24클래스 평균 Δ −0.09, 얇은 객체 4클래스 Δ 시드별 +2.11/−1.57/−1.80 으로 부호 불일치. 기제 게이트(찾고도 못 그린 비율)도 미달. val +1.18 은 test 에서 사라짐. 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 52,54,57,60,65 |
| DRN-260926-62 | reject | R2(연결 성분 단위 soft-IoU 손실, 원거리 가중) 40에폭 스크린: 시드821 두 사본 legal v2 24클래스 Δ −0.03 / +0.06(효과 없음), 기제 게이트 미달(원거리 못 그림 비율은 줄었으나 찾음률도 하락, mIoU 이득은 근거리 구간에만). 시드902·903 은 재채점했다고 기록됐으나 수치가 대장·NAS 에 보존되지 않아 3시드 평균은 미확정 → 시드821 근거와 기제 게이트 미달로 종료. 체크포인트는 NAS ckpts/screen40_bengio_20260923/ 에 있어 필요 시 재채점 가능. 원문 = 판정 대장 2026-09-23·24 행(55·67·72행) | experiments/judgment-ledger.md 행 55,67,72 |
| DRN-260926-63 | adopt | E1 40ep 스크린 3시드(821·902·903) 를 40ep 스크린 카드의 짝 기준선으로 확정: legal v2 test 56.58±0.55, 24클래스 56.65±0.23, 얇은 객체 4클래스 49.60, val 67.51±0.91. 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 64 |
| DRN-260926-64 | reject | E3b(E3 센서별 클래스 prototype + 센서 간 prototype 일치 항, AGREE_LAMBDA 0.1) 40에폭 스크린: legal test 54.10 으로 E3 보다 낮음 → 센서 간 일치를 강제하면 해롭다는 결론으로 폐기. 원문 = cards/2026-W37-verdicts.md | - |
| DRN-260926-65 | reject | E4c(혼동 쌍 margin 을 RailTrack 3쌍으로 좁히고 MARGIN 0.25) 40에폭 스크린: legal test 55.32, 24클래스 Δ +0.13 으로 게이트(Δ25 ≥ +1.0, Δ24 ≥ +0.5) 둘 다 미달 → 폐기. 원문 = cards/2026-W37-verdicts.md §5-12 | - |
| DRN-260926-78 | reject | P49 AIR 비대칭 주입(제안 2026-08-10): DELIVER 에서 ep30 게이트 위반(P49) 및 fair-eval 로 '57.68 SOTA' 철회(P49.1), MUSES 팔은 완주 기록 없음 → 계열 종결(2026-08-16, 양 벤치 패배, 가설 대장 H14) | - |
| DRN-260926-79 | adopt | P50 MAP 모달 정렬 사전학습(제안 2026-08-17): Phase1(프로브, 200k) init 파인튠 게이트 통과(+0.74, H22 확인) → P52 init 으로 채택. Phase2 확장(EXT)은 기각(DRN-260926-32). 단 P52 본 런이 DELIVER 에서 기준선 미달이라 최종 레시피에는 남지 않음 | - |
| DRN-260926-80 | reject | P51 CMLC 교차 모달 LoRA 결합(제안 2026-08-21): 페어1 에서 가설 대장 H19 반증으로 2026-08-26 종결 | - |
| DRN-260926-81 | inconclusive | condexpert 어댑터 프로브 제안(2026-08-08): registry·monitor-log 에 실행 기록이 없어 실행 여부와 결과를 확인하지 못함. 판정 불가로 종결. 조건 전문가 어댑터 아이디어는 P56-C(환경 조건 LoRA 전문가 혼합)로 이어짐 | - |
| DRN-260926-82 | inconclusive | H10 재판정 실험 요청(2026-08-08, 런 이름 jarvis_muses_h10_instsup_mini): 실행 기록·결과가 없어 판정 불가로 종결. 인스턴스 감독 축은 P48·R2 에서 실패로 확정됨 | - |
| DRN-260926-83 | adopt | 공간 모달 오라클 프로브 제안(2026-08-18, H16): 2026-09-28 tools/oracle_headroom_probe.py 로 실행됨 — 창마다 최선 모달 부분집합을 고르는 오라클의 순 여유는 +1.0~2.2 mIoU(조건별 1.1~1.8), 64px 창의 약 70% 는 전 모달이 최선. 이 결과를 채택해 곱셈 게이트를 주 후보에서 제외(판정 대장 2026-09-28·29 행) | - |
| DRN-260926-84 | adopt | P36 노벨티 비판 검토(2026-07-16): 'attention bias 계열은 4세대·2백본에서 결정을 바꾸지 못했다'는 체계적 반증을 논문 ablation 재료로 삼고, '새 모듈 제안 + SOTA' 노선을 접는다는 결론을 채택. 이후 학습 설계 노선(Q2·P56)의 출발점 | - |
| DRN-260926-85 | - | - | - |
| DRN-260926-86 | - | - | - |
| DRN-260926-87 | adopt | MULTIAQUA 챌린지(RGB+Thermal+LiDAR, 종료 트랙) P9(교차 모달 융합 헤드) hardaug8+PhysAug ep131: M-score 82.10(val 93.29 / test 70.91, 재제출 #16710) = 챌린지 공동 1위 제출본. 종료 트랙의 확정 기록으로 채택 | - |
| DRN-260926-88 | adopt | MULTIAQUA 챌린지 P22 hardaug8+PhysAug ep120: M-score 82.10(val 93.42 / test 70.77, #16932) = P9 와 공동 1위 제출본. 종료 트랙의 확정 기록으로 채택 | - |
| DRN-260926-89 | reject | MULTIAQUA 챌린지 P21(DEBA-FP) hardaug8+PhysAug ep94: M-score 81.77(val 93.17 / test 70.36, #16792) 로 P9·P22(82.10) 에 미달 → 최종 제출본으로 채택되지 않음. 종료 트랙 | - |
| DRN-260926-90 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): DGFusion Swin-T 재학습 (a), 발표 config 재현, DELIVER 4모달 200k(yeon). final-iter(공식 프로토콜) val 65.62 / test 55.56, val-best(80k) val 66.54 / test 55.68, 공개 발표값 66.51 / 56.71 대비 −0.89 / −1.15. test-best(100k 55.92)는 인용 금지. 이 값이 우리 모델과의 공정 비교 기준선 | - |
| DRN-260926-91 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): DGFusion Swin-T 재학습 (b) = (a) + 우리와 같은 모달 드롭 p=0.2 + 열화 커리큘럼(jarvis). val-best 90k val 61.74 / test 54.77 → (a) 대비 val −4.80 / test −0.91. 열화 커리큘럼이 DGFusion clean 을 깎는다는 근거(우리 Q2 는 clean 손해 없음). 판정 대장 2026-09-23 행 | experiments/judgment-ledger.md 행 40,58,63 |
| DRN-260926-92 | adopt | 측정 항목(채택 = 기준선 기록으로 확정): CAFuser Swin-T 공식 학습 코드 재학습, DELIVER 4모달 267k(lecun, 2026-09-15 완주). 공식 test 55.38(final), 학습 로그 최종 val 66.04. 체크포인트 스윕 27개의 test 예측이 lecun 에 남아 있음(lecun 은 배치 금지 서버, 파일 회수만 가능). DGFusion (a) 와 함께 공정 비교 기준선 | - |
| DRN-260926-93 | reject | P31(SAM2 계보, PhysAug) DELIVER B200 런: 당시 최선이었으나 B200 상실(2026-07-15 마감) 뒤 P34~P36(ReliaDINO)·P46/E1 계보가 대체. registry 의 '학습 중' 표기는 낡은 것. 계보에서 승계되지 않음 | - |
| DRN-260926-94 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P29·P31·P32·P34 표준분석(lecun, 2026-07-12~13). SAM2 계보 실패-키 문서의 근거. 산출물 NAS analysis_logs | - |
| DRN-260926-95 | reject | P29(SDC) DELIVER B200 런: val 63.20@ep100 / test 54.34@ep146 로 목표(66.51 / 56.71) 대비 −3.31 / −2.37. SAM2 계보 비교 기준선으로만 남고 승계되지 않음 | - |
| DRN-260926-96 | reject | P30 DELIVER B200 런: val 49.76 / test 44.10 으로 P29 대비 −13.4 / −10.2 회귀 확정(dead). 승계되지 않음 | - |
| DRN-260926-97 | - | - | - |
| DRN-260926-98 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P37a-CEFR(클래스 기대 특징 라우팅) MUSES 출력 분석(yeon, 2026-07-18). 라우팅 분화 실패(19클래스 중 0개 committed) → CEFR 라우팅은 반증 축으로 등재 | - |
| DRN-260926-99 | adopt | 측정 항목(채택 = 분석 결과를 기록으로 확정): P37a MUSES 표준분석 + 종합 실패-키(yeon, 2026-07-20). fog 62.7 최약, event·lidar 어댑터 기여 확인. 산출물 NAS analysis_logs/P37a_muses_std_20260719/ | - |
| DRN-260930-02 | adopt | 측정 항목(채택 = 결과를 기록으로 확정) + 가설 판정: DGFusion (a) 발표 재현 vs (b) 열화 재학습, 같은 프로토콜(test 1,897장·시드0·61케이스). EMM 15조합 평균 31.53 → 45.52 로 가설 전반 채택(열화 학습이 결측 강건성을 줌). 후반(depth 의존 유지)은 (b) 에서 반증(depth 통째 결측 −33.3 → −4.7). 우리 Q2 3시드는 (b) 대비 clean +1.8, EMM +2.5(48.06), depth 부분 저하 약 절반(0.98/2.04/2.64 vs 2.51/2.88/3.47), 노이즈 저하 1/2~1/4 로 모든 축 우위 → 강건 우위 주장 성립(단일 런·백본 차이 각주). 원본 NAS analysis_logs/baseline_robust_dgfusion_{a,b}_20261001, 판정 대장 2026-10-01 행 | - |
| DRN-261001-01 | adopt | 측정 확정: DGFusion 공식 MUSES 가중치 val 250장 재현 79.7183(README 79.72, 채점 축 동일) → 재학습 불필요. 우리 PhysAug-off 3시드(82.30±0.14) 대비 val +2.58 이나 공식 test 는 −0.21 이라 이 대조는 val 만 설명. 열세는 얇고 작은 4클래스(pole −6.20·person −5.37·traffic sign −4.34·traffic light −3.16, 3/3시드)와 야간 조건(clear/night −2.98·rain/night −3.01)에 국소 → '전 클래스 균일 열세' 반증 조건 불성립, 가설 채택. 거리 구간 표(LiDAR 희소 화소 한정)는 하지 않음. 다음 = test 측 증거(Codabench 클래스별 IoU vs DGFusion 논문 test 클래스별 표, GPU 불필요). 원본 NAS analysis_logs/muses_baseline_dgfusion_official_20261001/, 판정 대장 2026-10-01 행 | .claude_logs/experiments/analysis/2026-10-01-muses-dgfusion-official-vs-ours-val.md |
<!-- lab:plan-queue:end -->

> **재설계 기준**: ①논문(accept) 기여 — A(P51 확장)·B(진단-프레임워크) 어느 분기에서도 쓰이는가 ②24GB(yeon 3090/jarvis 4090)에서 도는가 ③원장 반증 경로가 아닌가. 옛 대기열 대부분은 계보 사망·중복으로 종결 처리(하단 🗑).

### 🔵 진행 중 (2026-09-17 재정리 — 옛 3트랙 표는 전부 종결돼 내림) — (2026-09-26 lab-plan으로 이관, 이하 표는 동결 사본)

| 트랙 | 다음 이벤트 |
|---|---|
| **DELIVER 확정 3쌍 판정** | jarvis C3-only 902·903 완주(09-17 16:25·20:20) → legal 재채점 → E1·E13 3쌍 판정(생각정리). 판정표에 카드 §0-2 두 항목(우리 최고 단일 런 56.99 대비·모달리티 정합 SOTA 거리) 병기 |
| **MUSES 풀 런 4종** | E1M 3407(09-17 08:20) · E7 3407(09-18 11:20) · E13M 3407(09-18 23:35) · 시드 20260902 페어(09-19 03:30~03:45) → 공식 재채점 → test 제출 후보 확정 |
| **E-LoRA 3갈래 판정** | arm C 903(09-18 03:45) · A·B 902·903(09-20) 완주 → A/B/C 3시드 판정 |
| **E17 세부 가지 스크린** | hpca100 GPU0 대기 기동(09-17 08:20 예정) → 40ep 스크린 → G1~G4 판정 |

> 🗑 내린 것: ~~P51-CMLC 페어1~~(08-26 H19 반증으로 종결) · ~~P50-MAP finetune~~(H22 +0.74 로 판정 완료) · ~~시드 n=5~~(N6 5/5 로 완결, 54.39±0.76).

### 🎯 신규 대기열 (논문-가치 순, 여유 GPU 투입 대상) — (2026-09-26 lab-plan으로 이관, 이하 표는 동결 사본)
| # | 실험 | 자원 적합 | 논문 가치 (A/B 분기별) | 상태 |
|---|---|---|---|---|
| **N-E18** | **E18 = E1 레시피 + Lovász-Softmax 보조손실**(OHEM CE 유지, 픽셀 로짓, 배치 내 존재 클래스만, 가중 0.5) 40ep 스크린 시드 821 — 얇은 객체 손실 직격(문헌 pole +2.2·TrafficLight +3.2). 제안 = decisions/2026-09-17-strategy-exploration-moe-lora-fusion-training.md §2 | 빈 슬롯 1~2장 | 게이트 G1 24클래스 Δ ≥ +0.5 · G2 얇은 객체 4클래스 Δ ≥ +2.0 · G3 · G4. 레포 미시도(losses.py에 Lovász 없음) | 🟡 후보 등재(2026-09-17), labcode 구현 대기 |
| **N-E19** | **E19 = E1 레시피 + 가중치 EMA(0.9995) val 평가·저장** — 시드 분산 축소 + 소물체 이득(GOOSE +1.12). 헤드라인 선택 규칙에 "EMA val-best" 병기 조항 필요(카드 §0) | 빈 슬롯 1~2장 | 게이트 G1 · 3시드 std 감소 여부 병기 | 🟡 후보 등재(2026-09-17), 규칙 개정 후 구현 |
| **N-D0** | **D0 = LoRA 진단 2종(학습 0)**: (i) 4모달 LoRA A 행렬 층별 코사인 (ii) 조건별 미니배치 그래디언트 코사인 — E20(공유 A)·E22(조건 게이트) 근거 확보용 | GPU 1장 수십 분 | (i) ≥0.9 → E20 진행, (ii) ≤0 조건쌍 존재 → 조건 전문가 근거 | 🟡 후보 등재(2026-09-17) |
| **N-E23** | **E23 = E1 레시피 + DGFusion식 depth 보조 감독(학습 시 depth 헤드) + 로컬 depth 토큰(융합 cross-attn K/V)** — 🔴 user 허용(2026-09-17) 단 **+α 필수**(E23 단독은 SOTA 주장 불가). +α는 기준선 실패 분석 D4(가설 H1~H4, decisions/2026-09-17-strategy-exploration §2)에서 도출 | 빈 슬롯 2장 | 게이트 G1 24클래스 Δ vs E1 ≥ +0.5(재현 근거 +1.0~1.3) · G3 · G4 | 🟡 설계 대기 — D4 결과 후 E23(기반)과 +α 카드 동시 설계 |
| **N-USER** | **사용자 결정 대기(잔여)**: 토큰 밀도 상향(실측 반대) · 마스크 분류 주 헤드(비용 4×) | — | — | ⏸ user 결정 대기 |
| **N-P53** | **E17 = E1 레시피 + 고해상도 세부 가지(DETAIL_BRANCH)** 40ep 스크린, 시드 821 — 얇은 객체 병목 직격(제안서 decisions/2026-09-17-p53-detail-branch-proposal.md). config `configs/{jarvis,hpca100}-deliver_rgbdel_P46_c3only_seed20260821_screen40_E17.yaml`(구현 후 등재) | jarvis 2장 또는 hpca100 1장 | user 렌즈(전 모달 융합 계열 1위)의 DELIVER 시드 평균 약점 해소 후보. 게이트 G1 24클래스 Δ vs E1 ≥ +0.5 · G2 얇은 객체 4클래스 Δ vs E1 ≥ +2.0 · G3 악조건 · G4 SOTA 거리 병기 | ✅ **완료·종결로 표에서 제거(09-22 로그 확인)** — 상세는 judgment-ledger.md, 재기동 대상 아님(09-24 자동배치 조사에서 상태 상충 발견해 정정) |
| **N-MC** | **확정 레시피의 MCubeS 이식 3시드(200ep)** — DELIVER 확정 판정(09-14~15, E13·E1 각 3페어)에서 채택된 레시피를 MCubeS 통일 레시피 3시드(3407·20260827·20260828)에 매칭해 돌린다. E13 채택 → `configs/{hpca100,yeon}-mcubes_rgbadn_P39_1_rank_E13Mc_seed{3407,20260827,20260828}.yaml`, E1 채택 → `…_E1Mc_seed…yaml` | 서버당 2장×3런(yeon 3090×2 ≈16h/런, hpca100도 가능) | 단일 아키텍처 원칙상 세 벤치 공통 적용 증거(DELIVER·MUSES·MCubeS). 게이트 = 3시드 평균 Δ vs 매칭 C3-off(58.07±0.49) ≥ +0.5, **final-epoch 기준**(MCubeS는 val-best = test-best 동치라 보수치), val-best 병기 | ❌ **종결(2026-09-16)** — 기다리지 않고 먼저 돌린 같은 코드 3페어가 결론을 냈다. 4탭 단독 3페어 평균 **+0.10**, 4탭+prototype 3시드 평균 **−0.63** 으로 둘 다 게이트(+0.5) 미달이고 페어 폭 0.64 가 평균 효과의 여섯 배다. 사전 등록 종결 조건대로 **MCubeS 시드를 더 늘리지 않는다**(카드 §5-30 ② · §5-21 말미). config 12벌은 보관만 한다 |
| ~~N1~~ | ~~MUSES 시드 분산 ×2~~ | — | ✅ **완결(2026-08-27)** | 공식 val 3점 {82.13, 81.79(s824), 81.47(s825)} spread **0.66** = MUSES val 시드 안정 확정 → test 단일제출 방어 근거. ⚠️ **09-18 정정: 공식 test 2점 {79.788(s2), 78.786(s825)} 격차 1.00 = val spread가 test에서 증폭 → 단일제출 방어 근거로 못 쓴다**(`experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md`) | registry 참조 |
| ~~N2~~ | ~~MLE-SAM 평균융합 baseline~~ | — | ✅ **완료·판정(2026-08-31, H21)** | legal test **55.45** — gated-MLP(54.2~55.4)와 동급 이상 = **우리 트렁크 우위 주장 철회**, 믹서 3점 완성(mean≈gated-MLP>xattn). 소거 논지 완결 재료 | 판정 [analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md](analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md) |
| **N3** | **C3 진단-구동 검출기 (분석, 학습 0)** — 기존 ckpt들의 val confusion에서 class-transfer 붕괴 지표(비대각 집중도) 정량화 → C3 on/off 효과와 상관 검증 (DELIVER 붕괴有/MUSES 無) | **GPU ~0**(캐시 confusion 재집계, 필요시 1 GPU eval) | **B 헤드라인 기둥·A여도 통일 서사 필수** — "벤치별 C3 on/off"를 원칙적 자동설정으로 전환하는 근거. 검출기가 두 벤치의 경험적 C3 효과와 일치하면 통일 아키텍처 주장 성립 | 🟡 분석 설계 = discussion 세션 직접 — **즉시 가능** |
| ~~N4~~ | ~~MCubeS 이식 파일럿~~ | — | ✅ **완결 + 3-seed(2026-08-31)** | {57.93, 57.67, 58.62} = **mean 58.07±0.49, published 최고(54.65) +3.42 / min +3.02 = MCubeS 1등 통계 확보**. N4b(C3-on)는 dose-response 예측 2/2 적중(H20✓) | [analysis/2026-08-25-n4-mcubes-first-entry-verdict.md] · [analysis/2026-08-27-n4b-dose-response-confirmed.md] |
| ~~N4b~~ | ~~MCubeS C3-on 페어~~ | — | ✅ **완료·판정(2026-08-27): 사전등록 예측 2/2 적중** | rubber **+9.76**(18.80→28.56, published 대역 복귀) + overall Δ−0.10(범위 내) → **dose-response 3점 성립(H20 ✓)** — 진단-구동 프레임워크 예측력 실증, 논문 헤드라인 기둥. 판정 [analysis/2026-08-27-n4b-dose-response-confirmed.md](analysis/2026-08-27-n4b-dose-response-confirmed.md) |
| ~~N6~~ | ~~DELIVER legal-val ckpt 재선택~~ | — | ✅ **완결(2026-08-31)** | 5/5: mean **53.82→54.39±0.76** (816 +2.21·821 +0.63, 선택 아티팩트 2/5런 실재). **최고 단일런 = seed816 55.29**(base 아님 — outlier 서사 완전 해소). 이후 모든 legal 수치는 하네스 가드 `--check` 필수 | registry·current.md 반영 |
| **N7** | **VICReg-off 격리 토글 1런** (seed821 매칭, VICReg만 off) — 컴포넌트 기여표 마지막 미지 행 | yeon 2장 ×~15h | 🔵 **착수(2026-08-31, user 승인)** — config `yeon-...seed20260821_vicregoff.yaml`(MODEL.P39.VICREG.ENABLE만 off, diff검증), yeon GPU0,1 200ep, 부팅검증 PASS(SEED 일치·카운트 일치·RANDOM INIT 0). ETA ~15h — N2(H21)로 트렁크-타입 기여가 죽어서, 우리 설계 기여 중 VICReg(rank 복원)의 단독 격리가 필요해짐. MUSES +0.76이 트렁크+VICReg 묶음이었음 | 게이트: Δ(VICReg-on − off) — 격리 수치가 곧 논문 ablation 행 |
| ~~N8~~ | ~~P50 런 legal-val 재선택~~ | — | ✅ **완료(2026-09-01)** | 6개 ckpt 스윕 — ep30(trainer top1)=legal-val-best 동일, 재선택 변화 없음 → **P50 legal test 54.95 최종 확정 = P52 게이트 G1 기준**. 선택 아티팩트 없음(이 런은 트레이너 지표가 정확했던 케이스) | H22 수치 최종화 |
| **N9** | **정성 증명 패키지** — ①VICReg 전후 유효랭크 스펙트럼 ②C3 전후 confusion(RailTrack·rubber) ③per-class 예측맵 패널 | 1 GPU 간헐 (기존 seg-analysis 도구) | 🟡 **스코프 설계 중** — 대상 ckpt가 3서버 분산(jarvis/yeon/hpca100)이라 실행 위치 설계 후 착수(모니터링 세션) — user 요구 "정량+정성 개선 증명"의 정성 절반. 논문 그림 직결 | 산출 = analysis_logs/ + 논문 figure 후보 |
| **P52-감사** | 🔴 **P52 기동 전 재설계(P52.1) — [decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md](../decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md)**: ① 신호 즉검 — 진행 중 P52 DELIVER s1/s2 train.log의 `[C3-ADPT]`(ep≥5)에서 RailTrack λ_c ≥0.5·λ_max 확인, 미발화면 신호 교체 후 재기동 ② 게이트 재등록(동등성 [−1.0,∞) 3페어 + G5 고정가중 스윕 −0.5 이내 + λ 궤적) ③ 병리 조작 1런 ④ MUSES 3모달 기준선 병기 | ①=로그 grep(학습 0) · 시드 +1 3런 | 🟡 **user 검토 대기(2026-09-07)** — 마진 0.3 < 시드 σ 0.76~0.92라 현 게이트로는 판정 불가. 본런 5개는 이미 진행 중(d8f6e81)이므로 **런 유지 + 병행 수정**; DELIVER ep5~10 `[C3-ADPT]` 판독이 첫 행동 |
| **P50-재현** | P50 Phase1 init 유/무 매칭 페어 +2(총 3페어) — H22(+0.74)가 페어 1개 결과라 노이즈 안; P52 init 근거 확정용 | yeon 2장 × 4런 | 🟡 등록(2026-09-07, 위 감사 문서 축 3-0) — P52.1 본런보다 먼저 |
| **E-BB A1** | 백본 적응 깊이 ablation 1순위: LoRA 전 선형층(QKVO+MLP) r32·α=64, 3페어 매칭 (A2 마지막 6블록 LP-FT unfreeze·A3 전체 FT+feature-align은 A1 판정 후) | yeon 2장 × 3런 | 🟡 등록(2026-09-07, 감사 문서 축 2) — E-LoRA arm A(=A0 기준치) 완주 후. 게이트 ≥A0+1.0 채택 / ep30 RGB-only −0.5 kill |
| **MUSES-PhysAug-off** | 🔴 **공정성 정정(user 2026-09-07)**: MUSES 최선(P39.1 seed2 3모달) 레시피에서 PhysAug만 off, 시드 매칭 3페어 → MUSES 헤드라인·SOTA 표 교체. P52 MUSES 팔도 시드 3부터 PhysAug off | yeon/hpca100 2장 × 3런 | 🟡 등록(2026-09-07, 감사 문서 §3.5-1) — DELIVER는 07-20부터 off였으나 MUSES 계보는 on이었음(P43/P44만 off, 효과 분리 불가) 🔵 **착수(2026-09-21 16:2x, 모니터링 세션)** — user 요청 "yeon 에서 plan.md 기반으로 다음 작업 얹어줄 수 있어?"에 이 대기열에서 구현이 끝나 있어 바로 돌릴 수 있는 항목으로 고른 것이다(생각정리 세션이 등재·승인한 카드가 아니다). 기준 = MUSES 최선 레시피 P39.1-rank(공식 test 79.788) 시드 명시판, 바꾼 키는 DATASET.PHYSAUG.ENABLE true→false · TRAIN.SEED · SAVE_DIR 셋뿐. yeon GPU5,6 = 시드 20260824, GPU2,3 = 시드 20260825(둘 다 300ep, 페어당 하루 이상). 셋째 시드 20260826 은 config 만 올려 두고 자동 배치 대기열(`scripts/autoplace/queue.tsv` id `muphys_826`, 우선순위 1)에 있다 — **세 페어를 다 돌릴지는 user 결정 대기**. 근거 = user 2026-09-07 공정성 정정(DELIVER 는 07-20 부터 PhysAug off 였는데 MUSES 계보만 on 이라 두 벤치 비교가 불공정). 결과가 나오면 MUSES 헤드라인·SOTA 표 교체 여부를 생각정리 세션이 판정한다. ✅ **소급 승인(판정 세션 2026-09-21)**: 시드 824·825 는 이미 ep1 이상 진행 중이라 그대로 완주시킨다. MUSES 공정선 헤드라인을 "최선 레시피 P39.1-rank PhysAug-off, 시드 명시 2시드 mean±std" 로 채울 수 있다(기존 79.29 는 PhysAug-on). **시드 826(세 번째 페어)은 user 결정 전까지 자동 배치 대기열에서 뺐다**(`scripts/autoplace/queue.tsv` 에서 `muphys_826` 행 삭제, 300ep×페어 하루 이상의 비용은 user 승인 사항). 우선순위 규칙: yeon 의 GPU 는 MUSES 300ep 가 점유하므로 P54 카드(E1-shared·R1/R2 시드·Q2/Q3)는 bengio·jarvis 에서만 배치한다. |
| **DELIVER-perclass-3way** | 클래스별 3자 비교표(학습 0): 우리 P46 C3 5-seed / DGFusion 재현 / MM-SA(공개 시) → 앞섬·동급·뒤짐 + 혼동형/공통붕괴형 재분류 + C3 on/off 클래스별 비용 | GPU 1장 재채점 | 🟡 등록(2026-09-07, 감사 문서 §3.5-2) — '강점 유지·혼동 개선' 서사의 실증 표 |
| ~~**Router-oracle-probe**~~ (S0-feature-probe로 흡수) | 오라클 지도 라우터 프로브: 고정 특징 위 소형 라우터를 H16 오라클 정답 부분집합으로 지도학습, val 정확도 − 우연 ≥ +10%p and 합성 예측 legal val ≥ +0.5 → 학습형 라우팅 재개 검토 / 미달 → 영구 폐쇄(H26) | GPU 1장 수 시간 | 🟡 등록(2026-09-07, 감사 문서 §3.5-3) — user 제안(공유 LoRA + 패치·클래스별 가중)의 재개 가능성 판정 |
| **DAILY-CARDS** | 🔴 **일일 사이클 실험 카드**([decisions/2026-09-07-daily-cycle-experiment-cards.md](../decisions/2026-09-07-daily-cycle-experiment-cards.md)): 40ep 스크린(seed821 매칭, Δtest≥+1.0 통과) → 200ep 3페어 확정. 순서 B0 기준선 → E0 특징 프로브(=S0) → E9 로짓 보정 → E7 MUSES PhysAug-off → E1 4탭(S1) → E2 전층 LoRA(S2) → E3 센서별 prototype → E4 혼동쌍 margin → E8 표적 copy-paste → E5 deformable 픽셀 디코더 → E6 블록 교환(S3) → E10 상위절반 FT(S4). **EPOCHS 40** | 카드당 yeon 2장 20h / hpca100 4장 6h | 🔵 **1일차 착수(2026-09-07 22:xx)** — bengio: B0(GPU0-3)·E1(GPU6,7) 학습 중, E0(GPU4)·E9(GPU5) 실행 중. **E2 hpca100 GPU1,3 착수**(tmux `hpca100_E2`, 로그 `logs/hpca100_E2_launch.log`, 착수 2026-09-07 hpca100 UTC 14:07, EPOCHS 40·EVAL_INTERVAL 5, iter속도 1.49it/s×1991iter≈22분/ep → ETA ≈14.8h, 완주 예상 2026-09-08 13:55 KST경). **E7 hpca100 GPU2 착수**(단일 GPU, DDP 아님, tmux `hpca100_E7`, 로그 `logs/hpca100_E7_launch.log`, 착수 2026-09-07 hpca100 UTC 14:07, EPOCHS 40·EVAL_INTERVAL 5, iter속도 1.72it/s×1500iter≈14.5분/ep → ETA ≈9.7h, 완주 예상 2026-09-08 08:49 KST경). hpca100 GPU0은 user 예약으로 비움(E2·E7 착수로 확보한 GPU1,2,3 슬롯 = P52 MUSES seed1(GPU0,2)·seed2(GPU1,3) 보류로 반납된 것, [decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md](../decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md) §1.5). E3 코드 완료/구현 중, GPU 해방 시 순차 기동 |
| **S0-feature-probe** | 🔴 **사다리 출발점(학습 = 선형 헤드만)**: DINOv3 중간층 원 특징 vs 어댑터 후 특징에서 오라클 센서 선택·클래스 예측 정확도 비교 → 어댑터가 정보를 버리는지 판정(감사 문서 §3.6 S0; Router-oracle-probe 행을 흡수) | GPU 1장 반나절 | 🟡 등록(2026-09-07) — 결과가 S1~S3 착수 여부와 논문 포지션(방법 vs 분석)을 가름 |
| **S1-4tap** | 4탭 읽기(블록 6/12/18/24)+SimpleFPN, 3페어 매칭 — 문헌 frozen 백본 +2~4(기하 모달) | yeon 2장 × 3런 | 🟡 등록(2026-09-07, §3.6) — S0 양성 시 즉시 |
| **S3-block-exchange** | 매 블록 모달 간 교환 어댑터(StitchFusion-MoA/CrossWeaver-MIB식), 백본 frozen — ⚠️ P51-CMLC(−0.82)와의 차이를 제안서에 선행 기술 | 구현 1주 + 3런 | 🟡 등록(2026-09-07, §3.6) — S1·S2(E-BB A1) 판정 후 |
| **P50-진단** | 학습0 진단 D1~D5(LogME·모달 Fisher·intruder-dim·CKA ID/OOD·WiSE-FT α=0.5 보간)로 Phase2 역전 원인 판정 → P50.1 처방 1개 등록 | GPU 1장 간헐 | 🟡 등록(2026-09-07, 감사 문서 축 3) |
| **P52** | **RxDINO 개정(2026-08-31 user 승인): 단일-config 자기-적응 처방** — C3-adaptive(per-class λ=f(온라인 붕괴지표)) + UniBal-adaptive(per-modal λ_u=f(laziness지표)) + P50 init. 벤치별 config 차이 0, 차이는 λ 궤적으로 창발 | 3벤치 × 단일 config × 시드2+ | 🔵 **컨트롤러 구현 위임 중(GLM)** → 검수·스모크 → UniBal 고정런 완주(~2일, G2 기준 확정) 후 본런 | **게이트 사전등록**: G1/G2/G3 = 각 벤치 최선 고정팔 −0.3 이내(단일 config) + **G4 창발**(λ 궤적이 RailTrack↑/MUSES≈0/rubber↑를 스스로 재현). 실패 시 1회 수정 재시도 후 철회·정직 보고. 개정문서 [decisions/2026-08-31-p52-rxdino-adaptive-amendment.md](../decisions/2026-08-31-p52-rxdino-adaptive-amendment.md) |
| **P50-EXT** | P50 사전학습 확장(코퍼스/에폭 스케일업) — 제안서 "통과 시 확장" 조항 | 대규모(수일) | ⏸ 등록 + 슬롯 후보 = hpca100 0,1(현재 타유저 점유, 해방 감시) — 🟢 **user 승인(2026-08-31, 디스커션 B)** — **Phase1 설계 확정(2026-08-31)**: 편광 = 새 모델 0 — **기존 depth gradient→normal→Fresnel(n=1.5) 유도**(비용 2배→1배), NIR = luminance+excess-green 결정론. 출력 = MCubeS 로더 원자규격(aolp_sin/cos·dolp npy + nir png, 로더 실측 스펙 기반). **코퍼스 = 500k, 사전학습 = 등-스텝(500k×12ep ≈ 200k×30ep = 6M샘플)로 다양성 효과 격리**, 통과 시 에폭 확장. **생성기 병합 완료(2026-08-31, 검수 PASS)** → **(a) 200k 증분 완주·검증(2026-08-31)** — 4모달×200k 각 전량, 기존 파일 무변경(mtime), meta 병합 정상, uint8 PNG 포맷. **✅ 500k 코퍼스 완성(2026-09-01)** — (b) 300k×8모달 전량(실패 0, DA 백엔드 4워커 일관, 254GB/SSDc), 200k@SSDe + 300k@SSDc(Phase2 multi-root). SHA256SUMS 기준선 생성 중. **🔵 Phase2 DELIVER-팔 기동(2026-09-02, hpca100 GPU1,3)** — 2×bs4×accum2=eff16·448²·등-스텝 375k, 검증 5항 PASS(root_samples 200k+300k·19.6GB/rank·0.848s/step), **ETA ~3.7일**. 워치독 커버. 게이트: EXT-init 파인튠 vs 프로브-init 54.95, Δ≥+0.3→EXT 채택. 사고 이력 3건 문서화(post-Adam ill-conditioned 검사→grad-레벨 교체 / A100 TF32→스모크 CPU고정 / img-size 448 누락→검증항목 추가). MCubeS-팔은 이후. Phase2 사전학습(등-스텝 500k×12ep)은 hpca100 해방 대기 → Phase2=hpca100 해방 시 본 사전학습. 게이트 사전등록 예정(프로브 +0.74 대비 스케일 이득). 확장 시 **편광/NIR proxy 추가 생성**으로 3벤치 공통 init화가 설계 포인트 | H22 이득(+0.74)이 스케일에 비례하는지 |
| **E-LoRA** | **LoRA 구조 ablation** — A. per-modal r16(현행=기존 seed821 재사용, 추가런 0) vs **B. 완전공유 r16** vs **C. 공유 r8 + per-modal 잔차 r8** — 시드 20260821 매칭, DELIVER 프로브, 파라미터 수 보고 | 신규 2런(B·C) × yeon 2장 × ~15h | 🟢 **등재(2026-09-01 user 승인)** — 근거: H22(P50 +0.74)가 "독립 어댑터 미정렬" 증거 → C는 정렬을 아키텍처에 내장하는 대안. ⚠️ **반증 가족 아님 확인**: 라우터/게이트/입력-의존 가중 없음(모달 정체성 정적 배정) — H1/H16(선택 계열)과 무관. 입력-조건부(D형)는 금지 유지 | **게이트(사전등록)**: C ≥ A(54.21) − 0.3 → 채택 검토(+P50 상호작용 팔 후속: C+P50 vs A+P50으로 정렬 기제 상보/중복 판정) / C·B 모두 A 미달 → 현행 A가 실증 정당화(논문 ablation 행). 슬롯 = N7 완주 후 yeon 해방분, P52 본런보다 후순위 |
| **N5** | TTA-on 실측 (구 #4, 참고용 ablation 행) | yeon/jarvis 1장 ×7h | 낮음 — 헤드라인 불가 확정, ablation 완결성용 | ⏸ 위 소진 후 필러 |

### 🅰️ A100 대기열 (⚠️ 제목의 "P51 완주 후" 조건은 2026-08-26 P51 종결로 소멸 — 현재 A100 4장은 MUSES 풀 런 4종이 09-17~19 까지 점유) — (2026-09-26 lab-plan으로 이관, 이하 표는 동결 사본)
| 순위 | 실험 | 근거 |
|---|---|---|
| ① | **P47-2 UniBal** (MUSES 4모달 역전 유일 레버, 구현·스모크 완료) | A100 필요(보조 head 메모리). P51 판정 후 슬롯 |
| ② | P51 후속 (F 추가 재프로브 or MUSES 비회귀) — P51 Δ 판정에 따라 | 게이트 분기 결과 대기 |

### 🗑 종결 처리 (2026-08-24 재설계에서 제거 — 재등재 금지 사유 명시) — (2026-09-26 lab-plan으로 이관, 이하 표는 동결 사본)
- ~~#1 P40 RCA-Fusion~~: 계보 사망(P39.1 게이트 자체가 P46/P51로 승계) + C-2 감쇠는 적응계열(H1~H4 폐쇄) 인접 — **원장 저촉**.
- ~~#2 P39 radar-fix 재실험~~: 이미 충족 — fixed-decoder drop-radar ablation(2026-07-30)이 radar 무익(+0.13)을 재확정. 별도 학습 불요.
- ~~#5 P47-MUB D-1~~: **이미 실행·폐기**(P47-D1 공식 test 78.790, val 과적합, 08-17). D-2는 위 A100 ①로 승계.
- ~~#10 P49-AIR~~: 계열 종결(08-16, 양 벤치 패배·H14). plan 갱신 누락분 정리.
- ~~A100 ③ P49 @1024 대조~~: P49 계열 종결로 무의미.
- ~~ProbeA2-7B 추가 측정~~: H12 폐쇄(7B +0.18)로 종결.

## ✅ 완료·판정 (재실행 금지) — (2026-09-26 lab-plan으로 이관, 이하 표는 동결 사본)

> 🔎 **2026-08-10 발견**: `jarvis_muses_rgbl_P39_1_rank_2modal`(MUSES RGB-L 2모달)이 **이미 2026-08-06 완주**돼 있었음(val 82.00@136, 서버 로컬 미기록 실행) — 재실행 금지, test 제출 여부만 판단 대기. registry 행 참조.

| 실험 | 결론 |
|---|---|
| **A/B 격리 (Arm A/B)** | **radar(또는 4모달 구조)가 범인. lidar 재투영·event dilation·eff batch 전부 무죄.** Arm A ep24 best 73.85@ep18 — 대조군(ep10 74.24)에 앞서지 않음 → **DGF 투영 = 중립** |
| **TTA 판정** | **경쟁자 3종 전부 미사용** → 헤드라인 사용 불가. CMNeXt 논문 명시(*"single-scale test strategy"*). 우리 MSF는 **dead config**라 과거 수치 무오염 |
| **투영 정합** | DGFusion 파라미터 재현 완료(공개 PIXEL_MEAN 오라클로 −0.1% 적중). **실제 차이는 lidar뿐**(radar·event 30ms는 이미 동일). **성능 이득 0, 공정성만 확보** |
| **module ablation** | **제안 모듈 전부 ≈0**(ATTN_BIAS=RBMA 간판 포함). gate+calib만 test +0.26. **성능 출처 = DINOv3 백본 + per-modal LoRA** |
| **det 붕괴 진단** | 원인 = **BS1의 gradient 노이즈**(n_pos 1~3), LR 아님. 처방 = 배치↑ + **LR 유지**. warmup 5ep 완주로 검증 |
| **seg-P37a/b (bengio분)** | **사망 확정** — bengio 노드 CUDA 전체 장애(GPU5 HW 고장, 재부팅 후 SSH 미복귀)로 ep1~2에서 종료. jarvis 재기동분(P37a→P37b 체인)이 계보 승계 — 남 세션 소관이라 수치 갱신하지 않음 |
| **MUSES 시드 20260825 test 제출본** | **제출 완료·공식 test 78.786 수신(2026-09-18)** — 시드2 79.788 대비 −1.00, 2점 mean 79.29(std 0.71) < DGFusion 79.5 → 융합 계보 1위 서술은 시드2 단일 런 한정, 판독 = `experiments/analysis/2026-09-18-muses-official-test-p39_1-seed20260825.md`. 이전 기록: 준비 완료·업로드 전(2026-09-17) — MUSES 1위 수치가 시드2 단일 제출이라 시드 평균을 만들기 위한 둘째 제출 후보. 공식 val 재채점 **81.4653** 으로 기록값 81.47 재현(로드 clean, `total_trainable=50,801,334`), test 750장 추론 완료, zip 이 헤드라인 제출본과 파일명 집합 완전 일치. 산출 = `/ailab_mat2/.../submission/muses/muses_P39_1_seed20260825_3modal_ep168_submission.zip`(md5 e4f02f3fc6599d31db4a9a5bf9b3b95f). 🔴 **시드 20260824 는 체크포인트가 어느 서버에도 남아 있지 않아 재학습 없이는 제출 불가** — 생각정리 결정으로 재학습은 보류 |
| **P43-PanopticDual (MUSES, hpca100)** | **완주** — best val 82.51@ep156 (seed2 82.62 −0.11 / P38 82.22 +0.29). val로는 seed2 미돌파. PQ 축(설계 헤드라인)은 MUSES panoptic GT 부재로 val PQ 미측정 → PQ 판정 보류. ckpt `outputs/ReliaDINO/hpca100_muses_rgbel_P43_pdual/epoch156_82.51_top1_checkpoint.pth`. test 제출 후보(mIoU 82.5대). Total Training Time 01:37:24는 로깅 아티팩트 |

## ⚠️ 사고 기록 (반복 금지)

- **2026-09-09 — DGFusion 재학습 감시 공백으로 jarvis GPU5·7 상실**: 발산 후 1h45m 방치되는 사이 타 사용자(openpi) 점유. 베이스라인 재학습에도 워치독 등록 필수. 같은 날 발견한 평가 함정: DGFusion 세만틱 평가는 `DATASETS.TEST_SEMANTIC`을 읽는다(`DATASETS.TEST`만 바꾸면 val이 그대로 돌아감 — val 2005/test 1897 장수로 구별).

- **2026-07-27 — jarvis SSH 불통 (connection refused)**: 내부 172.27.183.201:22 즉시 거부 — bengio(timeout)와 달리 호스트는 살아있고 sshd 중단/포트 변경 가능성. jarvis 상주 학습·ckpt 생존 여부 미확인 — **jarvis 사용 세션은 접근 복구 확인 후 진행할 것.** (p33-impl 세션 감시 중 감지)


- **2026-07-21 ISSUE-026 — ColorAugSSD brightness uint8 클램프 버그**: 07-16 이후 `DGFUSION_AUG:true` DELIVER 학습(jarvis P37a/b, hpca100 P38-DELIVER 완주분·P39-DPC resume 진행중, yeon 스모크) 전부 RGB가 발화 샘플(p=0.5)에서 백색 상수로 붕괴(RGB-dropout 0.5 효과) — MUSES 계보는 무영향. **P38-DELIVER 게이트 미달 판정 및 P39-DELIVER thin-class 퇴행 판정 모두 보류**(교란변수), P39.1부터 픽스 적용 클린 학습(상세: `issues/issues-and-fixes.md` ISSUE-026).

- **2026-07-21 ISSUE-025 — MUSES radar 디코딩 3중 버그**: `_open_radar` 폴스루+디스패치 오배선+`RADAR_RANGE_MAX` 미정의로 100m 클립(포화 2.76%) + height 채널 오염, develop에서 수정 완료 — 3모달 전 계보 무영향, 4모달(P34 등)만 오염(상세: `issues/issues-and-fixes.md` ISSUE-025).

- **2026-07-16 14:3x — bengio SSH 접속 불가 (port 400 Connection refused ×3)**: seg-P37a/b launch(~13:53) 약 30분 후 발생. 게이트웨이(210.125.85.207)는 정상(yeon 포트 600 OK) → bengio sshd 또는 호스트 자체 다운. 내부망 확인 일부 시도(미확정). **학습 생존 여부 불명** — (a) 호스트 다운이면 seg-P37a/b 사망(ep1~2라 손실 미미, 재기동 필요), (b) sshd만 죽었으면 학습 생존. **콘솔/관리자 확인 필요.** 복구 시: `pgrep -f "P37a_cefr|P37b_classtoken"` → 생존이면 지속, 사망이면 `/SSDb/jemo_maeng/src/p37_train`에서 재launch(스냅샷·데이터 무손실). 교훈: 서버 단일점 의존 — **ckpt 주기 NAS 백업**(B200 상실 전례) 체계화 필요.

- **2026-07-16 04:30 — 진단 런에 EPOCHS=300 방치**: Arm A(3모달+DGF)는 **진단 목적**이었는데 EPOCHS를 300(≈36h)으로 둔 채 방치 → **8배치 대기 잡을 36시간 막을 뻔함.** 진단은 ep14에 이미 끝났음(lidar 무죄). **교훈: 진단 런은 EPOCHS를 판정에 필요한 만큼만(예: 10~16) 설정하고, 계획에 ETA를 명시할 것.** 04:40 종료해 0-3 해제.
- **2026-07-15 lecun 상실**: 위 GPU 점유 원칙 참조.
- **2026-07-18 hpca100 로컬 WIP 보존**: seg-P38 본학습 launch 전, 타 세션이 hpca100에 남겨둔 미커밋 MUSES 작업을 wip 커밋(3e7fd68) 후 브랜치 `hpca100-wip-20260718`로 보존해 GitHub에 push(릴레이). hpca100 체크아웃은 develop @c3d1184로 전환됨 — 원 세션이 회수 가능.

## 🔗 관련

- [registry.md](registry.md) 실험 한눈표 · [monitor-log.md](monitor-log.md) 실시간 · [log.md](log.md) 결과 canonical · [../status/current.md](../status/current.md) 현재 스냅샷
- 회수 산출물: `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/` · 분석: `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/`


## 🔬 Modality Ablation 실험설계 규약 (user 지정 2026-07-17)

**모달 추가/제거 실험은 경쟁 논문 ablation과 대조해 설계·해석한다.** 문헌 조사 결과(2026-07-17):

**문헌 modality ablation 실태:**
| 논문 | ablation | metric | 핵심 수치 |
|---|---|---|---|
| **CAFuser** (MUSES) Table IX | 누적(RGB→+L→+R→+E) | **PQ** | RGB 55.7 → +L **+3.0** → +R **+0.6** → +E +0.4 |
| **MUSES** 원논문 Table 3 | 카메라 대비 개별 | **PQ** | +E +2.6 / **+R +4.4** / +L +5.8 (단독). **night +5.4, snow +3.9(lidar 열화조건 radar 대체)** |
| **CMNeXt** (DELIVER) | 누적 | **mIoU** | RGB 57.20 → +D **+6.38** → +E +0.86 → +L +1.86 (radar 없음) |
| **DGFusion** | **per-sensor ablation 부재** | — | 아키텍처/loss ablation만(Table IV/V). DELIVER는 CLE 51.6 → CLDE(+depth) **+5.1 mIoU** |

**🔴 규약 (모달 실험 설계·해석 시 필수):**
1. **비교 기준선을 정확히**: radar는 **"카메라 대비"(+4.4)가 아니라 "lidar 위에 추가"(CAFuser +0.6≈0)**로 봐야 함. 우리 P34는 lidar 있으므로 radar 기대 천장 ≈0. **우리 MUSES radar val −0.09/test −0.72는 이 잉여성과 정합 — "우리 실패"로 단정 금지.**
2. **누적 ablation 필수**: 한 점(4종 vs 3종)이 아니라 **RGB→+L→+L+R→+L+R+E 누적을 우리 metric(mIoU)으로**. CAFuser Table IX 미러링. 우리 +R이 ≈0이면 "정상(잉여)", 크게 음수면 "우리 융합 문제".
3. **per-condition breakdown 필수**: radar는 **night/snow(lidar 열화조건)에서 대체재**로 이득(MUSES night +5.4/snow +3.9). **aggregate가 조건별 이득을 가릴 수 있음** → 조건별로 3모달 vs 4모달 대조.
4. **DGFusion 대조**: DGFusion은 per-sensor ablation이 **아예 없다** → **mIoU 기준 modality ablation은 문헌 전무**(전부 PQ). 우리가 내면 문헌 공백 메우는 기여. 우리 노벨티(신뢰도 라우팅) = "radar를 lidar 보조가 아니라 lidar-degraded 조건 대체 range 신호로 라우팅"이 살 자리.
5. **PQ vs mIoU 구분**: 문헌 modality ablation은 전부 **PQ**(우리는 semantic-only=mIoU). PQ↔mIoU 직접 비교 금지. panoptic head 없이 PQ 산출 불가([[seg-report-sota-gap]]).

**적용 예**: MUSES/DELIVER 모달 실험을 짤 때 위 표를 baseline으로 붙이고, 누적 ablation + per-condition을 기본 산출로. 근거 = 조사보고(2026-07-17, DGFusion 2509.09828 v3 / CAFuser 2410.10791 v2 Table IX / MUSES 2401.12761 v4 Table 3 / CMNeXt 2303.01480).

### ~~[대기 트리거] jarvis P37b-DELIVER 완주 시 → P34 per-class 비교 분석~~ (등록 2026-07-18 · 🗑 **2026-09-17 종결** — 대상 런이 7월에 끝났고 감시 ID·PID 가 모두 사라진 죽은 트리거다. 계보도 P39.1 로 승계됐다)

- **트리거**: jarvis `train_reliadino ... jarvis-deliver_rgbdel_P37b_classtoken` 프로세스 종료(Monitor blf49vkbc 감시 중). ETA 07-19 02:20경, ep200 완주.
- **할 일**: P37b-best(top-1 val ckpt)와 P34-best를 **동일 프로토콜로 DELIVER val per-class IoU 산출 → 클래스별 Δ(P37b−P34) 테이블**. 실행=sonnet(tools/ 표준 분석 스위트), 비교 판정=opus.
- **P34 baseline ckpt**: `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/P34_final_20260713/`. P34 DELIVER val 68.19.
- **맥락**: P37b는 P37a(CEFR) 붕괴를 피했으나 val best 62.99로 P34 −5.2pt 미달. per-class로 "어느 클래스에서 classtoken이 P34 대비 이득/손해인지" 규명 = DELIVER analysis 목적(user 지정). P37a는 ep24 고착(실패), P37b는 중립(무해무익) 잠정 판정.
- 참고 그래프: P37a `jarvis_p37a_valtest.png`, P37b `jarvis_p37b_valtest.png`.

### ~~[대기 트리거] yeon P37b-det 완주 → P38-det 자동 기동~~ (등록 2026-07-19 · 🗑 **2026-09-17 종결** — 체인 래퍼와 PID 가 사라진 죽은 트리거다. det 트랙은 D1 인증 배포로 이관됐다)
- **체인**: yeon tmux jemo/p38_chain wrapper가 P37b-det(torchrun PID 3733229) 종료 감시 → 종료 시 P38-det 자동 기동. 세션 독립(서버측 실행).
- **P38-det**: 워크트리 `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38` (브랜치 worktree-p38-det, f775687, `ReliaDINOM2FDetector`=M2F query head를 detector로). config `configs/det/det_P38_m2f_yeon.yaml`(M2F on, CEFR/CLASS_TOKEN/ROUTER off, grad-ckpt false, GRAD_CLIP 0.1).
- **기동 설정**: env openmmlab, DET_GRAD_CLIP=0.1, port 29713, **4-GPU 고정**(eff-batch를 P37a/b-det와 일치). ETA 07-20 14:30경.
- **검증 필요**: P38 기동 후 rank0 util>0·iteration 전진·NaN 없음 확인(opus). yeon P37b-det 완주 run은 DRONE-NAS ckpts/로 회수.
