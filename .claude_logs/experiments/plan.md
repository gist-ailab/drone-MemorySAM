---
created: 2026-07-16
updated: 2026-07-21 (ISSUE-025 MUSES radar 디코딩 버그 수정 반영 → 대기열 #3 "P39-4모달 radar-fix 재실험" 신설 + 사고 기록 1줄, 이하 대기열 번호 +1) ; ISSUE-026 ColorAugSSD RGB 붕괴 버그 반영 → hpca100 P39-DPC resume 오염 표기 + 사고 기록 1줄 + 대기열 #1 클린런 표기 ; 2026-07-23 대기열 #1 "P39.1 Rank 수리" MUSES-jarvis 분기 착수 → 실행중 표에 행 추가(jarvis 2,3,4,5, 기동검증 통과) ; 2026-07-26 P43-MUSES 완주(val 82.51@ep156, seed2 미돌파) → 대기열 #11 P44-BMR을 hpca100 GPU2,3에 착수(develop 678c493, 기동검증 통과); 2026-07-27 seed4 완주(81.92)→해방 GPU에 첫 4-modal(P39.1+radar) 착수(yeon 0,1,5, 305b030); seed2 분석 완료(trunk+2~7·VICReg lidar rank 78~100 검증); 2026-07-27 jarvis 리부트(드라이버 595.84 복구)→DELIVER 2실험 착수(P39.1-rank GPU0-3 / P44-BMR GPU4-7, BS1, develop be2603c) — DELIVER 첫 캠페인 실험; 2026-07-27 4-modal ep2 eval OOM→EVAL BS1+expandable_segments 수정 재기동(9f199be), ep4 eval 통과 확인; 2026-07-28 P44-MUSES 완주(80.71)→해방 A100에 2번째 4-modal(P44-BMR+radar) 착수(hpca100 0-3, 1cf1e66, BS1 OOM수정); 2026-07-28 hpca100 4모달 HF 백본 이중고장(offline=RANDOM INIT/online=hang) 확진 → RELIADINO_LOCAL_BACKBONE env fix(encoder.py 697a10a) → P39.1+radar seed2 클린 기동(ep2 47.61); 2026-07-28 seed3 완주(81.89@204, 5-seed variance 완결) → P44-DELIVER seed2 yeon6,7 수동기동; P44-MUSES(80.71) test staging+분석 lecun; 2026-07-28 P46-CTR 제안 등재(DELIVER SOTA class-transfer, 내부신호 RCS+MIC+prototype) ; 2026-08-03 P46 C3-only λ0.2 DELIVER 완주(200/200, test-best 57.05@ep108) = **DELIVER test SOTA 돌파 확정**(DGFusion 56.71 대비 +0.34, @768 동일 프로토콜) → λ 스윕 상단탐색 λ0.3을 jarvis GPU4-7(회수됨)에 착수, 기동검증 PASS ; 2026-08-03 λ0.2 SOTA 재현성 검증을 위해 seed2를 jarvis GPU1-3(4090×3, GPU0=user 예약)에 착수, config `jarvis-deliver_rgbdel_P46_ctr_c3only_lam02_seed2.yaml`(develop b925c90), 기동검증 PASS ; 2026-08-04 **정정**: 57.05는 test-best 체크포인트 값으로 규약상 무효 확인됨 — legal 재계산(val-best/final-iter) 결과 최고 test 55.62~55.69, DGFusion 56.71 대비 −1.0로 **SOTA 미달**(base 대비 실제 이득은 test +1.35~1.74/val +0.97로 견고, λ 최적 0.05~0.2 평탄). 상세 [experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md](analysis/2026-08-03-p46-c3only-lambda-sweep.md) ; 2026-08-06 MUSES val PQ 첫 측정(P47-MUB D-1 ep172, native, tools/eval_pq.py b6d3da0) → things PQ 22.87 ≤ 30 = P48(쿼리 경로 인스턴스 감독) 사전등록 게이트 미달 → **설계 폐기** (analysis/2026-08-06-pq-first-measurement-p48-gate.md) ; 2026-08-08 대기열·예약표 청소(완주 4건 제거, bengio 잔재 제거, CEA 프로브 등재)
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

## 🖥 GPU 예약 현황 (2026-08-08 갱신 — 07-18 이후 미갱신 상태였음; 아래 실행 중 표와 함께 재정리)

| 서버 | GPU | 상태 | ETA |
|---|---|---|---|
| **lecun** | — | 🔴 타인(openvla) 점유 | — |
| ~~B200~~ | — | 🔴 상실(07-15 마감) | — |

## 🔬 실행 중

| 실험 | 서버/GPU | EPOCHS | ETA | 우선순위 | 목적 |
|---|---|---|---|---|---|
| **P49-AIR @768 본런** (DELIVER 4모달) | yeon GPU2,6,7 (3090×3) | 150 (ep30 게이트) | ep30 ~08-11 새벽 · 완주 ~08-12 | **P0** | 비대칭 주입 구조 전환 본검증. 기동 19:10, 기동검증 PASS(iter 9→270 전진·3GPU 99%·19.3GB·RANDOM INIT 0·에러 0). tmux jemo:p49_air_main, 워치독 등록. 게이트 = 제안서 §4 (γ성장·RGB-clean≥−0.3·test≥57.35·G-4M) |
| **hpca100 P39-DPC resume** (DELIVER 4모달) | hpca100 GPU 2,3 | 200 | 07-22 09:00 | — | resume 후 val 44ep·test 64ep 무갱신 정체(val 66.14@ep96/test 55.50@ep76, P38 대비 +0.95/+0.45 계보최고 유지). 🔴 **ISSUE-026 오염 상태로 학습 중 — 지속/중단 user 판단 필요**(07-16 이후 DGFUSION_AUG:true 런이라 ColorAugSSD RGB-dropout 오염 해당, 상세 `issues/issues-and-fixes.md` ISSUE-026) |
| **P39.1-rank MUSES** (jarvis, 대기열 #1 착수) | jarvis 2,3,4,5 (4090×4) | 300 (ep30 조기게이트) | ep30 게이트 ~07-23 18:35 · 완주(300ep) ~07-24 09:30 (추정, eval 오버헤드 별도) | **P0** | R-1(gated_mlp trunk, γ=0.1 init) + R-2(VICReg var+cov, lidar×1.0/기타×0.25) + M-2(gate/calib/veto off) — P39-MUSES 표준분석이 지목한 lidar effective-rank 붕괴(4.7)·fog_night 붕괴(62.68) 수리. config `configs/jarvis-muses_rgbel_P39_1_rank.yaml`, develop @a06b666(≥ac5c7fe). 07-23 16:5x 기동, tmux `jemo:p39_1_rank`, log `logs/jarvis_muses_rgbel_P39_1_rank_*.log`. **기동검증 통과**: iter 0→160/375(epoch1) 전진 확인·GPU 2,3,4,5 전부 99-100%util·~19GB/24GB(활성화 수준)·에러 0·wandb만 미설정(비치명, no-API-key). **판정 게이트(사전등록, ep30)** = lidar effective-rank ≥15 & fog_night drop-lidar ≥4.0 (미달 시 R-3: r8→16+rsLoRA 재기동) |
| **P39.1-rank DELIVER resume** (yeon, RCA 대조군) | yeon GPU 5,6,7 (3090×3, 원 6장에서 축소) | 200 (07-22 16:10 SIGTERM 사망 @ep39 iter198/663, 원인 미상) | ~07-24 오전 (다음 eval ep40 기준, 1327it/ep·~1.5-2s/it 추정) | — | `outputs/ReliaDINO/yeon_deliver_rgbdel_P39_1_rank/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth`(ep38)에서 AUTO_RESUME. SAVE_DIR을 `/SSDe/jemo_maeng/outputs/yeon-deliver_P39_1_rank_resume`로 이전(원본 20GB 복사, `/SSDb` 8G뿐이라 그대로 못 씀), config `configs/yeon-deliver_rgbdel_P39_1_rank_resume.yaml`(develop @1ba7f87→hub merge 7e655de). 실행 `openmmlab` env(timm 1.0.20) + `PYTHONPATH=<repo>/semseg/models/sam2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(🔴 MMSS_SAM env timm이 0.4.12로 회귀해 DINOv3 못 빌드 — 07-22 07:03 로그는 정상 빌드했었음, 원인 미상 회귀). **기동검증 통과**: Epoch[39/200] 재개 확인(처음부터 아님)·iter 179→192 전진·GPU5,6,7 86-99%util·~20.9GB·에러 0. 🔴 **ep30 rank 게이트 raw 수치는 통과하나(`p391/rank lidar`=532.1≥15) 실제 인과기여(drop-lidar dMIoU, night 대리조건)=−0.78로 여전히 ~0/음** — 판정 보류, 상위 세션 검토 필요 |
| **P42-MaskImg-f07 (FRAC 0.7)** (yeon, FRAC 스윕 완성) | yeon GPU 3,4 (3090×2 DDP) | 300 (ep30 조기게이트) | ep30 ~07-24 오전 (750it/ep·~1.2-2s/it 추정, 15-25min/ep) | **P0** | FRAC 스윕 0.3(jarvis)/0.5(hpca100)/**0.7(이 런)** 완성. base=`configs/hpca100-muses_rgbel_P42_maskimg.yaml`(FRAC 0.5) 복사 → `configs/yeon-muses_rgbel_P42_maskimg_f07.yaml`(FRAC 0.7, SAVE_DIR `/SSDe/jemo_maeng/outputs/yeon-muses_rgbel_P42_f07`, DATASET.ROOT yeon 경로, BATCH_SIZE 1). develop @bda6341→hub merge 8072101. 실행 = **정식 문서화된 방법**(`experiments/launch-runbook.md`) MMSS_SAM env + `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:.`(timm 1.0.24로 우회, sam2 불필요 — `train_reliadino.py`가 sam2 미import) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **기동검증 통과**: Epoch[1/300] iter 26→42/750 전진·GPU3,4 88-100%util·~17GB·loss 유한(10.67→9.22)·에러 0·config dump로 FRAC=0.7 확인. `p42_mask_rate` 로그는 EVAL_INTERVAL=2라 첫 eval(ep2) 전까지 미출력(다음 조회 시 확인) |
| **P39.1-rank 4모달(+radar)** | yeon GPU0,1,5(3090×3) | 300(ep30 게이트) | ~07-29/30 완주 추정 | **P0** | **첫 4-modal SOTA 시도**(user 4모달-우선 07-27). base=seed2 아키(P39.1-rank, val 82.62/test 79.788) + radar 추가=1변수. config `configs/yeon-muses_rgbelr_P39_1_rank_4modal.yaml`(develop 305b030), ISSUE-025 radar-fix 후 첫 클린 4모달. tmux `jemo:p39_1_4modal`, log `/SSDe/jemo_maeng/yeon_muses_P39_1_4modal_run.log`. 🔴 1차 기동 ep2 eval OOM 사망(3090 24GB 4모달 EVAL BS4 과대) → **EVAL BS4→1 + expandable_segments 수정 재기동**(develop 9f199be, config `yeon-muses_rgbelr_P39_1_rank_4modal.yaml`, log `_run2.log`). AUTO_RESUME으로 ep2 이어받아 **ep4 eval 63.46 OOM없이 통과·ep5 진입 = 사망지점 통과 확인**. ⚠️ GPU mem 22GB/24GB(~94%)로 여유 작음 — OOM 재발 계속 관찰. ✅ **ep30 게이트 PASS(재보정, 2026-07-27 20:34)**: val 78.82@ep30. 사전등록 임계 'val≥80'은 오설정이었음 — 3모달 P39.1-rank seed 4개의 ep30 = 78.45~78.78(seed2 78.73/seed4 78.78/seed3 78.59/seed5 78.45, 최종 81.7~82.62)이라 80을 넘은 seed 없음. 4-modal 78.82는 그 범위 최상단(소폭 위) = **radar가 수렴 지연 없이 seed2 궤적 추종**. 붕괴/OOM 없음. → 계속. radar 순이득 여부(3모달 초과 vs 동급)는 완주 val(≥82.62?) + drop-radar dMIoU로 최종 판정. 게이트: ep30 val≥80&붕괴없음 / 완주 val≥82.62(3모달 초과) 또는 drop-radar dMIoU>0@fog. seed4 완주(81.92)로 GPU 해방분에 배치 |
| **P39.1-rank DELIVER** | jarvis GPU0-3(4090×4) | 200 | ~07-30/31 완주 추정(995it/ep, 첫 ep 후 재확정) | **P1** | DELIVER 검증레시피 클린런(큐 #1 DELIVER 분기 미착수분). 4모달(img/depth/event/lidar). config `configs/jarvis-deliver_rgbdel_P39_1_rank.yaml`(develop be2603c, BS1 4090). tmux `jemo:deliver_p39_1`, log `_run2.log`. 기동검증 PASS(iter 570/995, GPU0-3 74~100%/~15GB, loss 7.72 유한, BS2→1로 OOM 해소). 게이트: P36 fair(val 67.74/test 56.62) 대비 + thin-class 유지. 목표 = DELIVER val 68.79(CAFuser-CAA, 격차 −1.05) 좁힘 |
| **4-modal P39.1+radar seed2** | hpca100 GPU0-3(A100×4) | 300 | ~07-30 완주 추정 | **P0** | yeon P39.1+radar 4모달의 seed2(A100 = 더 빠른 완주 + variance). config `configs/hpca100-muses_rgbelr_P39_1_rank_4modal_seed2.yaml`(eec590d), **RELIADINO_LOCAL_BACKBONE env로 로컬 백본 로드**(hpca100 HF 이중고장 우회, encoder.py 697a10a). BS1. tmux `jemo:p39_1_4modal_v7`, log `hpca_P39_1_4modal_v7.log`. 기동검증 PASS(백본 로컬로드·RANDOM INIT 없음·ep0 클린·**ep2 47.61 정상궤도**·GPU 24.6GB·OOM 없음). ⚠️ hpca100 churn — 재개 시에도 RELIADINO_LOCAL_BACKBONE env 필수. |
| **P39.1-rank 4모달(+radar) seed3** | jarvis GPU0-3(4090×4) | 300(ep30 게이트) | 산정 전(첫 ep 후 재확정) | **P0** | jarvis P39.1-rank DELIVER 완주(200/200, val 67.60/test 55.56, Total 07:19:25)로 해방된 GPU0-3에 배치. yeon seed1(4090×3 base)·hpca100 seed2에 이은 4-modal **3번째 시드**(variance). config `configs/jarvis-muses_rgbelr_P39_1_rank_4modal_seed3.yaml`(develop 017d712, yeon 4-modal config에서 SAVE_DIR/ROOT만 이식, BS1 그대로). radar 데이터 사전확인(jarvis `/SSDb/jemo_maeng/dset/MUSES/projected_to_rgb/radar` = train 1500/val 250/test 750 = 2500장, lidar와 동일 — 정상). 🔴 1차 기동 시 tensorboard/protobuf `TypeError: Descriptors cannot be created directly` 충돌로 즉시 크래시(WANDB.ENABLE=true가 tensorboard import 유발) → `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 추가해 재기동. tmux `jemo:p39_1_4modal_seed3`, log `/SSDb/jemo_maeng/jarvis_muses_P39_1_4modal_seed3_run.log`. **jarvis는 HF 온라인 정상이라 RELIADINO_LOCAL_BACKBONE 불필요**(hpca100 전용 우회). **기동검증 PASS**(백본 DINOv3 온라인 로드·RANDOM INIT 없음·radar+MODALS=4 확인·GPU0-3 89-100%util/~22.3GB·에러 0·**ep2 mIoU 46.93(정상 궤도, hpca100 seed2 ep2=47.61과 정합)**) — ⚠️ jarvis GPU4-7(P44-BMR DELIVER)은 89-100%util/~15.7GB로 무영향 유지 확인. |
| **P46-CTR (DELIVER class-transfer recovery)** | jarvis GPU4-7(4090×4) | 200(ep30 조기게이트) | 산정 전(첫 ep 후 재확정, ~995it/ep·2.6it/s 추정 21h+/ep200) | **P0** | P39.1-rank(val 67.60/test 55.56) 위에 class-transfer 3종 all-on: **C-1 RCS**(DAFormer, 희소클래스 우선샘플)+**C-2 MCC**(MIC EMA-teacher 패치마스킹 consistency)+**C-3 PROTO**(per-class prototype + cross-view). 근거 = per-class 붕괴맵(Wall/Water/Bridge test IoU 0~8) + DGFusion 대조 + R1024 재-eval — Wall/Water/Bridge는 DGFusion도 동반붕괴(복구불가·1차타깃 제외), **RailTrack만 진짜 격차**(우리 test 4.02 vs DGFusion 64.47, 해상도 무관) → §9 재타깃. develop 병합 32458f4(코드)+781cdaa(jarvis config). config `configs/jarvis-deliver_rgbdel_P46_ctr.yaml`(all-on, **BS1** — BS2는 스모크에서 backward OOM 실증, BS1은 rank당 15.2GB로 여유). tmux `jemo:p46_ctr_train`, log `logs/jarvis-deliver_rgbdel_P46_ctr/run.log`. **기동검증 PASS**(DINOv3 온라인 로드·RANDOM INIT 없음, C1/C2/C3 기동 3줄 전부 확인, iter 44→76/995 전진(25s)·loss 10.23→9.73 유한, GPU4-7 70-93%util/~15.2GB). **판정 게이트(사전등록, §9)** = ep30 조기: collapse 클래스(RailTrack) test IoU 무변화면 kill. 완주: **RailTrack test 4→≥40**(DGFusion 64.47 실증) & overall test≥56.62(+@1024 병기). |
| **P46-C1C3-DELIVER (all-on OOM 대체안)** | jarvis GPU0-3(4090×4) | 200(ep30 조기게이트) | 산정 전(2.6-2.7it/s, ep200까지 base 21h+ 추정) | **P0** | **all-on(C1+C2+C3) BS1이 4090 24GB에서 warmup 계단(WARMUP_EP=5) 도달 시 OOM 확정**(P46-MEMPROBE 실측: base 14.00GiB → 보조 branch 켜지자마자 23.5GiB 초과, 4-rank 전부 즉시 OOM) → **user 결정: C1+C3만 유지, C2_MCC.ENABLE=false + C3_PROTO.CROSS_VIEW=false**(보조 2-forward branch 완전 제거, C3 prototype은 주 forward aux로만 유지) — RailTrack 재타깃 핵심 기제(C1 RCS + C3 prototype) 보존. all-on(C2 포함)은 A100 슬롯 대기. config `configs/jarvis-deliver_rgbdel_P46_ctr_c1c3.yaml`(develop ecab531, all-on config 대비 2곳만 변경).
  - 🔴 **1차 기동 오염 발견 → user 판정 → FIX**: 1차 기동이 all-on 크래시런과 **SAVE_DIR 공유**(`jarvis_deliver_rgbdel_P46_ctr`, "2곳만 변경" 지시의 자연스러운 결과)로 AUTO_RESUME이 그 크래시런의 epoch5 체크포인트에서 재개하는 오염이 발생 — user 판정: "AUTO_RESUME 오염 확정 → kill + 전용 SAVE_DIR로 fresh 재시작". **조치**: 1차 런 kill(GPU0-3 확인 idle) → config에 전용 SAVE_DIR(`jarvis_deliver_rgbdel_P46_ctr_c1c3`) 추가(commit **bd0e098**, develop push 완료, jarvis merge 완료) → 신규 SAVE_DIR 비어있음 확인(진짜 fresh 보장) → fresh 재기동.
  - **재기동 중 발견/해결한 환경 이슈 2건(향후 jarvis P46/ReliaDINO 계열 기동 시 필수 반영)**: ①protobuf/tensorboard `Descriptors cannot be created directly` — `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 필요(P39.1 4modal seed3에서도 겪은 동일 이슈, plan.md L89 참조). ②conda env `MMSS_SAM` 기본 timm=**0.4.12**(DINOv3 미지원) — dinov3/dinov2 이름 둘 다 "Unknown model"로 실패 후 RANDOM INIT 폴백하다 최종 크래시. **`PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34` 필수**(timm 1.0.24 포함, DINOv3 레지스트리 보유) — 이 PYTHONPATH 없이 기동한 시도는 이번에 실제로 RANDOM INIT+크래시로 재현됨(중간 시도 로그 `run_fresh.log`).
  - tmux `jemo:p46_c1c3_train2`, log `logs/jarvis-deliver_rgbdel_P46_ctr_c1c3/run_fresh2.log`. **최종 fresh 기동검증 PASS**: `Resumed` 로그 0건(진짜 fresh, Epoch[1/200]부터 시작), 백본 DINOv3 온라인 로드(safetensors, RANDOM INIT 없음), `[P46-C1] RCS on`+`[P46-C3] prototype consistency on ... cross_view=False` 확인, C2 로그 0건, iter 22/995 loss 11.60→11.04 유한 감소, GPU0-3 15.6GB/rank·70-100%util, GPU4-7 미사용(무영향). **판정 게이트**: ep30 조기 RailTrack test 무변화 kill / 완주 RailTrack test 4→≥40 & overall test≥56.62. |
  - 🔴 **2026-07-30 12:xx kill (user 판정: C1 RCS 유해 확정)**: 게이트(RailTrack test 4→≥40)는 ep40에서 59.10/60.14(@768/@1024)로 압도적 통과했으나, **동일 ep40 C3-only(C1 off) 대조 결과가 C1+C3보다 오히려 높음**(RailTrack 64.13 > 59.10, overall test 55.64 > 54.92, @768 기준) — **C1(RCS)이 순유해**로 판정, C1+C3 완주는 저가치 판단. GPU0-3 kill 확인(프로세스 0, 전 GPU 0%util), epoch40 best ckpt(val 67.36) 보존 확인. GPU0은 user 별도 용도로 비워둠, GPU1-3은 C3 주력 후속 배치 대기(코디네이터 다음 지시).
| **P46-C3only-DELIVER (C3 단독기여 ablation)** | jarvis GPU4-7(4090×4) | 200(ep30 조기게이트) | 산정 전(2.6-2.7it/s 추정) | **P0** | **user 지시**: c1c3(C1+C3) 결과에서 C3 prototype의 단독기여를 분리하기 위한 ablation — C1_RCS off, C2_MCC off(기존과 동일), C3_PROTO만 유지(cross_view off). config `configs/jarvis-deliver_rgbdel_P46_ctr_c3only.yaml`(develop 576f1a9, c1c3 대비 3곳: 주석+SAVE_DIR+C1_RCS.ENABLE만 변경). 코드 확인(`train_reliadino.py` L188-233): `C1_RCS.ENABLE=false`면 `rcs_sampler=None` 유지 → DDP 분기가 표준 `DistributedSampler`로 정상 폴백, 이상 없음. 전용 SAVE_DIR(`jarvis_deliver_rgbdel_P46_ctr_c3only`)로 all-on/c1c3와 완전 분리(AUTO_RESUME 오염 재발 방지). tmux `jemo:p46_c3only_train`, log `logs/jarvis-deliver_rgbdel_P46_ctr_c3only/run.log`. 기동 시 c1c3에서 배운 필수 env(`PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) 처음부터 반영해 재시행 없이 1회 성공. **기동검증 PASS**: `Resumed` 0건(진짜 fresh), 백본 DINOv3 온라인 로드(safetensors, RANDOM INIT 없음), `[P46-C3] prototype consistency on ... cross_view=False` 확인·`[P46-C1]`/`[P46-C2]` 로그 0건(둘 다 off 정상), iter 21/995 loss 13.64→11.35 유한 감소, GPU4-7 15.5-15.6GB/rank·65-100%util, GPU0-3(c1c3) 무간섭 확인(15.8-15.9GB 그대로 유지). **판정 게이트**: c1c3와 동일(ep30 조기 RailTrack test 무변화 kill / 완주 RailTrack test 4→≥40 & overall test≥56.62) — **c1c3 대비 결과 차이가 C1 RCS의 순기여분**. |
| **P46-C2C3-DELIVER (C2 기여 격리)** | hpca100 GPU2,3(A100×2, 40GB) | 200(ep30 조기게이트) | 산정 전(2.06-2.08it/s, 1991it/ep) | **P0** | user 지시: C3-only가 C1+C3 전 지표(RailTrack/overall, @768/@1024)를 상회해 C1 RCS 유해 확정 -> C1 off + C2 MCC on + C3 PROTO on(cross_view=true)으로 C2의 순기여를 격리. all-on(C1+C2+C3)이 4090 24GB서 OOM났던 바로 그 보조 branch(WARMUP_EP=5 계단)를 A100 40GB로 재시도 - 여유 확인이 핵심 목적 중 하나. config `configs/hpca100-deliver_rgbdel_P46_ctr_c2c3.yaml`(develop f0fd231, all-on 대비 5곳: 주석+SAVE_DIR+C1_RCS.ENABLE+TRAIN.BATCH_SIZE(2->1)+EVAL.BATCH_SIZE(2->1)). tmux `jemo:p46_c2c3_train`, log `logs/hpca100-deliver_rgbdel_P46_ctr_c2c3/run.log`. env: hpca100 cuDNN 규약(LD_LIBRARY_PATH venv 번들 cuDNN) + RELIADINO_LOCAL_BACKBONE(hpca100 HF 우회, seed2 학습때 쓴 local safetensors 경로 그대로) + PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python. **기동검증 PASS**: Resumed 0건(진짜 fresh), 백본 DINOv3 온라인 로드(RANDOM INIT 0건), `[P46-C2] MIC-style masked consistency on` + `[P46-C3] prototype consistency on ... cross_view=True` 확인, `[P46-C1]` 로그 0건(off 정상), iter 132/1991 loss 13.63->9.89 유한 감소, **GPU2,3 16.7GB/rank·78-83%util(40GB 중 여유 큼 - 4090 OOM 원인이던 보조 branch가 A100서는 충분히 수용됨을 확인)**, hpca100 GPU0,1(타테넌트) 무간섭(37.8GB 그대로). **판정 게이트**: C2C3 대비 c1c3/C3-only 삼자 비교로 C2 순기여 산출. |
| **P46-C3only-seed2-DELIVER (재현성 검증)** | jarvis GPU1-3(4090x3) | 200(ep30 조기게이트) | 산정 전(2.6-2.7it/s 추정) | **P0** | user 지시: C3-only 결과(RailTrack test-SOTA 예비 도달)의 재현성 검증. config `configs/jarvis-deliver_rgbdel_P46_ctr_c3only_seed2.yaml`(develop 73479fa, c3only 대비 SAVE_DIR + 주석 + C1_RCS.SEED(0->1, C1 off라 inert) 변경). 코드에 전역 SEED 키 없음(train_reliadino.py grep 확인 - C1_RCS.SEED만 존재, RCS 자체가 off라 명목상 변경) - 학습은 원래 unseeded RNG라 SAVE_DIR 분리 자체로 이미 독립 2nd run. GPU0은 user 예약이라 1,2,3만 사용(3-proc). tmux `jemo:p46_c3only_seed2`, log `logs/jarvis-deliver_rgbdel_P46_ctr_c3only_seed2/run.log`. **기동검증 PASS**: Resumed 0건(진짜 fresh), 백본 DINOv3 온라인 로드(RANDOM INIT 0건), `[P46-C3] prototype consistency on ... cross_view=False` 확인, `[P46-C1]`/`[P46-C2]` 로그 0건(둘 다 off 정상), iter 206/1327 loss 13.6->9.32 유한 감소, GPU1-3 15.6-15.7GB/rank·83-90%util, GPU0(user 예약) 무변화(4062MiB 그대로)·GPU4-7(원본 c3only) 무간섭 확인. **판정**: 원본 C3-only 결과(RailTrack test-SOTA 예비)의 재현 여부. |
| **P46 C3-only 본run @1024²** | elice-b200 | — | 08-09 새벽 | **P0** | 판정 = CVPR/RA-L 분기 게이트. 커밋 c6efc8c/df753e8/a65e9cf(develop). BS8, eff-batch16. 🔵 학습 중(launch ~2026-08-07) |
| **P46 C3-only @1024²** | jarvis GPU6,7 | — | 산정 전 | **P0** | 커밋 c6efc8c/df753e8/a65e9cf(develop). 🔵 학습 중(launch ~2026-08-07) |
| **P39.1-rank @1024² 대조(해상도 교란 검증)** | jarvis GPU1-5 | — | 산정 전 | **P0** | 커밋 c6efc8c/df753e8/a65e9cf(develop). 🔵 학습 중(launch ~2026-08-07) |

## ⚡ 2026-07-25 즉시 배치 지시 (모니터링/학습 세션 필독 — P43/P44 준비 중)

**서버 실측 (2026-07-25 recon)**: hpca100 **GPU2,3 유휴**(A100 40GB×2, 우리 프로세스 0) · jarvis = P39.1-MUSES **base 완주(val 82.03@ep224)** + seed2(GPU2-5)/seed5(GPU0-1) 진행 · yeon = P42-f07 진행(ep98, best 79.13@ep96) + P39.1 seed3/4 진행. ⚠️ **P42-f05(hpca100) 완주: best 80.85@ep124 = P38 82.22 −1.37** — 전역 마스킹 val 손실 신호(판정은 상위 세션 몫).

1. 🟢 **P43/P44/P45 develop 병합 완료 = launch GO (07-25, commit 35ddbe0)**: 합성 스모크 P43 전건 PASS + P44 86 assert PASS + **P43+P44 동시-on 상호작용 검사 PASS**(aux 5종 공존·grad 유한·eval 결정론·panoptic_inference 동작). 서버는 `git fetch local && git merge develop`으로 수령.
2. **첫 슬롯 = P43-MUSES on hpca100 GPU2,3** (`configs/hpca100-muses_rgbel_P43_pdual.yaml`, A100 BS2·accum4=eff16). ✅ 검증 2건 완료·회수됨(판정 = [analysis/2026-07-25-router-coverage-verification.md](analysis/2026-07-25-router-coverage-verification.md) — V-1 필수 실증, P44 config 기본 on 전환). ⚠️ **GPU2,3을 타 세션 `hpca100-muses_P42_seed2_hold`(nohup, ep37/300, PID 3993220/1)가 점유 중 — `_hold` 런이므로 홀드 규칙 §1("진짜 실험 준비되면 즉시 죽이고 양보")에 따라 P43에 양보**(ckpt 보존되므로 재개 가능, 이 파일 상단 홀드 절 참조). **순서 = ① 실데이터 2ep 스모크를 같은 GPU에서 먼저**(EPOCHS만 2로 오버라이드 또는 로그 초반 2ep 검증으로 갈음 — p43_mask_loss 유한·mask_rate 로그·OOM 없음 확인) **→ ② 본학습 200ep**. 기동검증 기준 = iteration 전진·전 GPU util>0·에러 0·`train/p43_mask_loss` 유한. ep30 게이트(사전등록) = PQ_thing>0 & thin-class −1pt 이내 & 쿼리 non-empty(제안 §2).
3. **P44-BMR**(`configs/jarvis-muses_rgbel_P44_bmr.yaml`)은 jarvis P39.1 seed 런 종료 슬롯 또는 P42-f07 판정 후. ⚠️ **yeon GPU3,4는 user 예약(2026-07-25, 학습 배치 금지 — [[yeon-gpu34-reserved]], gpu-never-idle보다 우선)** — P42-f07(기존 런)이 쓰는 중이나 신규 배치는 금지.
4. ⚠️ P43 MUSES **PQ 실측은 panoptic GT 부재로 아직 불가**(gt_semantic만 보유) — 학습은 semantic mask-cls 모드로 진행 가능, PQ는 MUSES 공식 panoptic GT 다운로드 + 로더 후속(TODO, 별도 등재). mIoU 게이트(val ≥82.22 유지)는 기존 도구로 즉시 측정 가능.
   - **2026-08-04 갱신 — 코드 쪽 블로커는 해소**: `tools/eval_pq.py`(+`pq_format.py`, 스모크 `tools/smoke_pq.py`) 배선 완료로 **P43뿐 아니라 M2F(P39.1-rank 실전 계보)에서도** `panoptic_inference` → AUPQ 포맷 → PQ/SQ/RQ(thing/stuff 분해)가 돌아간다. 재학습 불필요. 남은 것은 **`gt_panoptic` + `gt_uncertainty` 다운로드**뿐이며, 받는 즉시 `--build-gt-json`(또는 동봉 json)으로 val PQ를 산출할 수 있다. test는 GT 비공개라 로컬 산출 불가(도구가 거부).

## 📋 대기열 (우선순위 순)

| # | 실험 | 필요 자원 | 언제 | 근거 |
|---|---|---|---|---|
| **1** | **P40 RCA-Fusion 본학습** (DELIVER + MUSES) | P39.1과 동일 자원, 완주 후 이어서 | **P39.1 rank 게이트(lidar effective-rank ≥15) 통과 확인 후** 투입 — rank가 죽은 채면 C-3 lidar readout이 헛돎 | **구현 완료(develop ac5c7fe)** — P39.1 위에 Reliability-Conditioned Attenuation 추가. C-1: lidar 리턴 유효성(입력 유도 내부 신호) → 가드/분석. C-2: 자기추정 rel(img) 배치 하위 분위(30%) 샘플의 img feature soft 감쇠(α 0.1~0.5, hard-zero 금지, p_max 0.5, warmup 20ep, 학습 전용). C-3: 감쇠 샘플 한정 lidar readout 보조 CE(w 0.5, gradient 출구). **판정 게이트(사전 등록)** = MUSES test ≥79.025 & fog_night ≥74(P38 복원 우선) · DELIVER = P36 fair + thin-class 유지. configs `jarvis-muses_rgbel_P40_rca.yaml`/`hpca100-deliver_rgbdel_P40_rca.yaml`/`yeon-deliver_rgbdel_P40_rca_smoke.yaml`(스모크). 합성 스모크 PASS(RCA pick 발생, C-1 가드 동작, 손실 유한, grad 흐름). 상세 [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md) / [models/arch-evolution.md](../models/arch-evolution.md) P40 |
| **2** | **P39-4모달 radar-fix 재실험** | hpca100/jarvis 4모달 슬롯 | P39.1/P40 완료 후 | ISSUE-025(MUSES radar 디코딩 버그) 픽스 후 radar 기여 재측정 — P34 4모달 test −0.72 판정이 broken-radar 상태 기반이라 보류 중 |
| **3** | **시드 복제 (2~3 seed)** | 4 GPU × N | GPU 여유 시 | 세션 내내 "+0.13/+0.10은 노이즈"라 말했으나 **분산 데이터 없음**. ablation 표에 ± 를 달 수 있음 |
| **4** | **TTA-on 실측** (참고용) | 1 GPU × ~7h(4090) | 여유 시 | **헤드라인 사용 불가 확정**(경쟁자 미사용) → ablation 행 전용. 준비물 배치 완료(hinton/jarvis). TTA-off는 G0a가 이미 확보(val 68.20/test 56.64) |
| **5** | **P47-MUB (D-1 lidar 투영 밀도화 → D-2 uni-modal balance aux)** | MUSES 슬롯 (D-1 먼저, 학습0 선행확인 후) | **D-1 최우선**(비용0, 데이터 기존 존재) — `muses.py` config knob 추가 후 즉시 착수 가능. D-2는 D-1 결과 후 labcode 위임 구현+코드검수 | **제안 등재(2026-08-03)** — 진단 재정의: 내부 프레임("주야격차 5.14")은 SOTA 추월 관점에선 오도, Codabench 원본 대조로 병목=clear/day(−4.4~−5.9) 확인, 모달↑=순위↓ 역상관 실측. D-1: `projected_to_rgb`(유효 6.7%)→`projected_to_rgb_dgf`((7,7)+motion comp, 32.6%=4.99×, 오라클 검증 완료, 7500 PNG 기존재)로 교체만. D-2: modality-laziness(2305.01233 UMT 등) 억제용 per-modal aux CE head(학습시만, 추론 불변). **게이트**: val≥82.62(seed2 base 초과) & Codabench test≥79.788(우리 최고) 1회 제출. D-1 falsifiable=drop-lidar day dMIoU 4.24→≥6. D-2 falsifiable=val day≥81.5(야간만 오르면 반증). ep30 조기kill=base 궤적 대비 −1.0. DELIVER 무영향(추론 불변+MUSES 전용 데이터+토글). 상세 [decisions/2026-08-03-p47-mub-muses-proposal.md](../decisions/2026-08-03-p47-mub-muses-proposal.md) |
| ~~6~~ | ~~CEA oracle 프로브~~ | — | — | 🔴 **완료 + 폐기 확정(2026-08-08)** — 7런 완주, G-P1 5배 미달(oracle Δ +0.21 < +1.0). 적응 가설 계열 폐쇄. [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md) §6·§7, 원장 H4 |
| ~~7~~ | ~~RGB-D 2모달 fair-eval~~ | — | — | 🔵 **착수(2026-08-10)** — yeon GPU2 val 모드 eval 중(tmux jemo:rgbd_eval1024), test 모드 후속. registry 행 참조 |
| ~~8~~ | ~~H10 재판정 미니 실험~~ | — | 🔴 취소(2026-08-10 user — PQ 비경쟁축) | **의뢰서 = [decisions/2026-08-08-h10-readjudication-experiment-request.md](../decisions/2026-08-08-h10-readjudication-experiment-request.md)** — 이 문서만 읽고 실행 가능. 게이트(학습 후 things PQ>33.6) 사전 등록, 결과 기록처 명시 |
| **9** | ~~ProbeA2 — 백본 스케일링 프로브~~ | jarvis GPU1, ~1h(캐시+head 학습 4종) | ✅ **완료(2026-08-09)** | 결과: S+ 59.85 / B 62.82 / L 68.67 / **H+ 69.19**. G-A2 **완결(08-12)**: 7B 69.37(+0.18)·축분리 음성 → 표현력 축 소진 확정(원장 H12 ✗) / G-A2-하한 Δ(L−S+)=+8.82(>3.0, 대형백본 전제 공개 필요). 상세 = [analysis/2026-08-09-probea2-backbone-scaling.md](analysis/2026-08-09-probea2-backbone-scaling.md), 원장 = [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) H12/H12′. **미결**: 7B 추가 측정(hpca100 A100 필요 — 24GB OOM 위험) 여부는 코디네이터 판단 대기 |
| **10** | **P49-AIR** (비대칭 주입 구조 전환) | Phase0=1 GPU 2h(학습0) → 구현(labcode) → jarvis/yeon 2~4장, EPOCHS 100~200 | 🟢 **승인 + 구현 병합(be. 검수 PASS) — 24GB 실측 스모크 진행 중, 통과 시 본런 즉시 기동(user 사전 승인 2026-08-10, jarvis 1,2,5 예정, 워치독 등록 포함)** | [decisions/2026-08-10-p49-air-asymmetric-injection-proposal.md](../decisions/2026-08-10-p49-air-asymmetric-injection-proposal.md) — 대칭 융합 폐지, RGB 주경로 FT + 인코더-내부 zero-init 주입. ep30 게이트(γ성장·RGB-easy 무손실)·DELIVER test ≥57.35·falsifiable A/B 사전 등록 |

## 🅰️ A100/B200 대기열 (슬롯 감시 = `scripts/gpu_slot_watch.sh`, cron 10분 — 2026-08-10 도입)

> hpca100(A100 40GB×4)·elice-b200 전부 타인 점유 중(0/4·0/8 실측). 빈 슬롯 전이 시 alerts.log + notify-send. **슬롯이 나면 아래 순서로 즉시 투입**:

| 순위 | 실험 | 필요 | 근거 |
|---|---|---|---|
| ① | **C2-MCC 순기여** (c2c3 config) | A100 2장 | 최장 미결·논문 표 직접 소요·40GB 필수(4090 OOM 실측) |
| ~~②~~ | ~~ProbeA2-7B~~ | — | ✅ 완료(08-12, hpca100 GPU2-3) — H12 폐쇄, analysis §6 |
| ③ | **P49 @1024 학습 대조** | 4장 | @768 본런과의 해상도 대조(24GB no-go 실측으로 밀림) |

## ✅ 완료·판정 (재실행 금지)

> 🔎 **2026-08-10 발견**: `jarvis_muses_rgbl_P39_1_rank_2modal`(MUSES RGB-L 2모달)이 **이미 2026-08-06 완주**돼 있었음(val 82.00@136, 서버 로컬 미기록 실행) — 재실행 금지, test 제출 여부만 판단 대기. registry 행 참조.

| 실험 | 결론 |
|---|---|
| **A/B 격리 (Arm A/B)** | **radar(또는 4모달 구조)가 범인. lidar 재투영·event dilation·eff batch 전부 무죄.** Arm A ep24 best 73.85@ep18 — 대조군(ep10 74.24)에 앞서지 않음 → **DGF 투영 = 중립** |
| **TTA 판정** | **경쟁자 3종 전부 미사용** → 헤드라인 사용 불가. CMNeXt 논문 명시(*"single-scale test strategy"*). 우리 MSF는 **dead config**라 과거 수치 무오염 |
| **투영 정합** | DGFusion 파라미터 재현 완료(공개 PIXEL_MEAN 오라클로 −0.1% 적중). **실제 차이는 lidar뿐**(radar·event 30ms는 이미 동일). **성능 이득 0, 공정성만 확보** |
| **module ablation** | **제안 모듈 전부 ≈0**(ATTN_BIAS=RBMA 간판 포함). gate+calib만 test +0.26. **성능 출처 = DINOv3 백본 + per-modal LoRA** |
| **det 붕괴 진단** | 원인 = **BS1의 gradient 노이즈**(n_pos 1~3), LR 아님. 처방 = 배치↑ + **LR 유지**. warmup 5ep 완주로 검증 |
| **seg-P37a/b (bengio분)** | **사망 확정** — bengio 노드 CUDA 전체 장애(GPU5 HW 고장, 재부팅 후 SSH 미복귀)로 ep1~2에서 종료. jarvis 재기동분(P37a→P37b 체인)이 계보 승계 — 남 세션 소관이라 수치 갱신하지 않음 |
| **P43-PanopticDual (MUSES, hpca100)** | **완주** — best val 82.51@ep156 (seed2 82.62 −0.11 / P38 82.22 +0.29). val로는 seed2 미돌파. PQ 축(설계 헤드라인)은 MUSES panoptic GT 부재로 val PQ 미측정 → PQ 판정 보류. ckpt `outputs/ReliaDINO/hpca100_muses_rgbel_P43_pdual/epoch156_82.51_top1_checkpoint.pth`. test 제출 후보(mIoU 82.5대). Total Training Time 01:37:24는 로깅 아티팩트 |

## ⚠️ 사고 기록 (반복 금지)

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

### [대기 트리거] jarvis P37b-DELIVER 완주 시 → P34 per-class 비교 분석 (등록 2026-07-18)

- **트리거**: jarvis `train_reliadino ... jarvis-deliver_rgbdel_P37b_classtoken` 프로세스 종료(Monitor blf49vkbc 감시 중). ETA 07-19 02:20경, ep200 완주.
- **할 일**: P37b-best(top-1 val ckpt)와 P34-best를 **동일 프로토콜로 DELIVER val per-class IoU 산출 → 클래스별 Δ(P37b−P34) 테이블**. 실행=sonnet(tools/ 표준 분석 스위트), 비교 판정=opus.
- **P34 baseline ckpt**: `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/P34_final_20260713/`. P34 DELIVER val 68.19.
- **맥락**: P37b는 P37a(CEFR) 붕괴를 피했으나 val best 62.99로 P34 −5.2pt 미달. per-class로 "어느 클래스에서 classtoken이 P34 대비 이득/손해인지" 규명 = DELIVER analysis 목적(user 지정). P37a는 ep24 고착(실패), P37b는 중립(무해무익) 잠정 판정.
- 참고 그래프: P37a `jarvis_p37a_valtest.png`, P37b `jarvis_p37b_valtest.png`.

### [대기 트리거] yeon P37b-det 완주 → P38-det 자동 기동 (등록 2026-07-19)
- **체인**: yeon tmux jemo/p38_chain wrapper가 P37b-det(torchrun PID 3733229) 종료 감시 → 종료 시 P38-det 자동 기동. 세션 독립(서버측 실행).
- **P38-det**: 워크트리 `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38` (브랜치 worktree-p38-det, f775687, `ReliaDINOM2FDetector`=M2F query head를 detector로). config `configs/det/det_P38_m2f_yeon.yaml`(M2F on, CEFR/CLASS_TOKEN/ROUTER off, grad-ckpt false, GRAD_CLIP 0.1).
- **기동 설정**: env openmmlab, DET_GRAD_CLIP=0.1, port 29713, **4-GPU 고정**(eff-batch를 P37a/b-det와 일치). ETA 07-20 14:30경.
- **검증 필요**: P38 기동 후 rank0 util>0·iteration 전진·NaN 없음 확인(opus). yeon P37b-det 완주 run은 DRONE-NAS ckpts/로 회수.
