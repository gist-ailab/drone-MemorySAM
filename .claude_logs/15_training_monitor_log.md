# 학습 모니터 로그 (Training Monitor Log)

> 생성: 2026-06-24
> **이 파일은 `/loop` 모니터 세션이 주기적으로 append하고, 모든 세션이 읽어 분석·판단·개선에 쓰는 공유 로그다.**
> loop 세션의 채팅은 다른 세션에 안 보이지만, 여기 기록된 내용은 `.claude_logs` init 규칙을 통해 전 세션이 공유한다.
> 규칙: ① 매 점검마다 한 줄 timestamped 엔트리 추가(append-only, 과거 줄 수정 금지). ② 이상징후(사망/정체/완료/신기록)는 엔트리 아래 `> ⚠️`로 강조. ③ 학습 종료/사망 시 [01_project_status.md](01_project_status.md) 스냅샷의 해당 트랙도 갱신.

---

## RUN-1 · B200 P28 RBMA (DELIVER)

- **서버/소유자**: B200 (unix user `gm_huis`), repo `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/b200-deliver_rgbdel_P28_physaug.yaml` (순수 RBMA, AMF_MODE=uniform, λ_bias init 1.0, 4모달 img/depth/event/lidar, 목표 200 ep)
- **출력**: `outputs/MMSamP28/b200_deliver_rgbdel_P28_physaug/DELIVER_CMNeXt-B2_idel/` (`train.log`, `epochN_<val>_topK…pth`, `test_epochN_<test>…pth`)
- **비교 기준**: 직접 경쟁군(Cluster B, test) DGFusion 56.7 / CAFuser 55.6 · 구조적 base(Cluster A) MemorySAM val 65.38 — 자세히는 [12_novelty_and_related_work.md](12_novelty_and_related_work.md).

| 점검 시각(KST) | epoch | Val mIoU | Test mIoU | best | GPU(util/mem) | 프로세스 | 상태 판정 |
|---|---|---|---|---|---|---|---|
| 2026-06-24 ~16:00 | 12 | 57.87 | 50.61 | ep12 | G3-7 활성 | alive (8+4 proc) | baseline. ep8→12 상승 중(val 49→58, test 49→50.6). ⚠️동일 config 중복 프로세스 의심 |
| 2026-06-24 16:43 (KST) | 12 | 57.87 | 50.61 | ep12 | G4-7 util 99~100% / ~47GB each (shared box, G0-3은 타인 작업) | alive — 단일 DDP 1개(pgid 2670507: torchrun + rank 2675377-80) | **정상 진행 중**. ep12 [Test] 직후 44분 경과, 다음 [Test]=ep14 ~18:00 KST 예상. val/test 동일값은 정체 아님(아직 ep14 미도달). |
| 2026-06-24 18:30 (KST) | 14 (Test ep14 평가대기) | 53.22 (best 57.87 @ep12) | 50.61 (ep12; ep14 평가 임박) | val ep12 / test ep12 | G4-7 util 96~100% (단일 DDP, ~47GB/rank) | alive — 단일 DDP(pgid 2670507) | **진행 중**. ep12→14 정상 진척. Day-Val ep14 일시 하락(57.87→53.22)이나 best 유지, [Test] ep14는 18:09 val 직후 평가중(미로깅). |
| 2026-06-24 20:30 (KST) | 14 (ep15 학습중) | 53.22 (best 57.87 @ep12) | **47.35** (best 50.61 @ep12) | val ep12 / test ep12 | G4-7 util 94~100% (단일 DDP, ~47GB/rank) | alive — 단일 DDP(pgid 2670507) | **⚠️정체·하락**. ep14 val·test 동반 하락(val 57.87→53.22, test 50.61→47.35). best는 ep12 유지. train loss는 계속 하락(1.039→0.989) → overfit 의심. |
| 2026-06-24 22:30 (KST) | 16 (val만; test 미실행) | 57.67 (best 57.87 @ep12) | 47.35 (ep14; ep16 미실행) | val ep12 / test ep12 | G4-7 P28 점유 해제(타 사용자 작업만 잔존) | **🔴 사망 — P28 proc 0** | **사망**. ep16 Day-Val 57.67로 회복(ep14 dip=fluctuation 확정) 직후 21:09~22:30 사이 프로세스 소멸. train.log에 traceback/OOM 흔적 없음 → 외부 kill·OOM-killer·tmux/세션 종료 추정. |
| 2026-06-25 00:30 (KST) | 16 (사망 유지) | 57.67 (best 57.87 @ep12) | 47.35 (ep14; best 50.61 @ep12) | val ep12 / test ep12 | G4-7 P28 미점유(타 사용자 작업만) | 🔴 사망 지속 — proc 0, 미재개 | **변화 없음**. train.log mtime·최신 ckpt(epoch16) 모두 06-24 21:09에서 정지. 재시작 안 됨 → 추가 진척 없음. |
| 2026-06-25 02:30 (KST) | 16 (사망 유지) | 57.67 (best 57.87 @ep12) | 47.35 (best 50.61 @ep12) | val ep12 / test ep12 | G4-7 P28 미점유 | 🔴 사망 지속(3회 연속) — proc 0, 미재개 | **변화 없음**. ep16/06-24 21:09에서 정지 그대로. 재개 시 즉시 반영 예정. |

> ⚠️ baseline 시점 관찰: 동일 config 프로세스가 8-proc + torchrun(4-proc) 두 그룹 → 중복 실행/유령 프로세스 확인 필요(같은 SAVE_DIR 덮어쓰기 위험).
>
> ✅ **16:43 정정**: ps상 53개 P28 프로세스가 보였으나 PGID로 묶으면 단일 그룹(`2670507` = torchrun launcher + 4 rank `2675377-80` + 각 rank의 persistent dataloader 워커들)뿐. **중복 실행 아님 — 단일 DDP의 워커 군집**으로 확정. nvidia-smi compute-apps에서도 P28 GPU 점유는 2675377-80(랭크 4개, ~47GB)만 잡힘. SAVE_DIR 덮어쓰기 위험 없음.

> ⚠️ **18:30 관찰**: `[Day-Val]` ep14 mIoU=53.22로 ep12(57.87) 대비 하락(직전 2회 54.36→57.87→**53.22**). 단 best val은 ep12 57.87로 유지되고 단일 epoch 변동 범위 내(ep8 54.18·ep10 54.36과 유사대)라 추세하락 아닌 일시 fluctuation으로 판단. **핵심 지표인 야간 [Test] ep14는 아직 미로깅(평가중)** — 다음 점검(~20:30 KST)에서 test ep14로 정체/하락 여부 확정 필요. 프로세스·GPU 정상, 신기록 없음.

> ⚠️ **20:30 핵심**: ep14에서 **Day-Val(57.87→53.22)·야간 Test(50.61→47.35) 동반 하락**. 직전 점검(18:30)에선 test 미로깅이라 val만 의심했으나, 이번에 test ep14=47.35로 확정 — best(val 57.87 / test 50.61)는 둘 다 ep12 고정. 동시에 **train loss는 ep12 1.039→ep14 0.989로 하락** → 학습은 계속 적합 중인데 일반화가 꺾인 형태로 **overfitting onset 가능성**. 단 ep14 단일 점이라 ep12가 lucky peak였을 수도. **판정 보류·다음 ep16([Test] ~21:20 KST 예상)이 분기점**: ep16도 ep12 미만이면 하락 추세 확정 → early-stop/best(ep12) 채택 검토, ep16 회복이면 fluctuation. 프로세스·GPU·중복 모두 정상.

> 🔴 **22:30 사망 확정**: B200에 `train_sam2_lora_paper.py`/torchrun 프로세스 0개(잔존 python은 manipforce·robocasa 등 타 사용자). train.log는 **ep16 Day-Val(57.67) 기록(21:09:23)에서 깨끗이 끊김 — [Test] ep16 미실행, 에러/traceback 없음** → 인프로세스 크래시가 아니라 외부 종료(OOM-killer/수동 kill/세션 드롭) 가능성.
> | **회복 신호(사망 직전)**: ep14 dip(val 53.22/test 47.35)은 일시 변동이었고 ep16 val 57.67로 peak(57.87) 근접 회복 → 20:30의 overfit 우려는 기각.
> | **자산 보존**: best 체크포인트 디스크 보존 — val `epoch12_57.87_top1`, test `test_epoch12_50.61_top1`, 추가 `epoch16_57.67_top2`, 재개용 `last_checkpoint.pth`(RESUME_ENABLE=True). EPOCHS=200 중 ep16에서 중단 → **미완**.
> | **권고**: 재개하려면 빈 GPU 확인 후 `last_checkpoint.pth`로 resume. 현재 best test 50.61@ep12는 직접경쟁군 DGFusion 56.7·CAFuser 55.6에 아직 미달 → 추가 학습 필요.

<!-- 새 엔트리는 이 줄 위 표에 한 행씩 추가 -->

---

## RUN-2 · B200 P29 RBMA seg (DELIVER) — RUN-1(P28) 사망 후 재가동

- **서버/소유자**: B200 (`gm_huis`), repo `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/b200-deliver_rgbdel_P29_physaug.yaml` (P29 seg, 4모달 img/depth/event/lidar). torchrun nproc=3, GPU 4-7 사용(공유 박스, G0-3은 타 사용자).
- **출력**: `outputs/MMSamP29/b200_deliver_rgbdel_P29_physaug/DELIVER_CMNeXt-B2_idel/` (`train.log`, `test_epochN_*_topK_checkpoint.pth`, `last_checkpoint.pth`)
- **비교 기준**: 직접경쟁군 DGFusion test 56.7 / CAFuser 55.6 — [12_novelty_and_related_work.md](12_novelty_and_related_work.md).

| 점검 시각(KST) | epoch | Day-Val mIoU | Test mIoU | best | GPU(util/mem) | 프로세스 | 상태 판정 |
|---|---|---|---|---|---|---|---|
| 2026-06-30 01:56 | 126 (학습중) | 61.16 (best **63.20 @ep100**) | 53.89 (best **54.21 @ep122**) | val ep100 / test ep122 | G4-7 util 98~100% / ~131GB each (G0-3 타 사용자) | alive — 단일 DDP nproc=3 (pgid 1170601) | **정상 진행 중**. ep122에서 Test 신기록(54.21, 23:58). 직접경쟁군 DGFusion 56.7엔 아직 -2.5. train loss 0.59대 안정. EPOCHS 목표까지 학습 지속. |
| 2026-06-30 10:37 | 148 (학습중) | 62.01 (best **63.20 @ep100**) | 54.19 (best **54.34 @ep146**) | val ep100 / test ep146 | 공유박스 3 GPU util 100% (점유 인덱스 변동) | alive — DDP nproc=3, D-state 없음 | **정상**. Test best 지속 갱신 54.21→54.22(ep138)→**54.34(ep146, 09:34 신기록)**. Day-Val 61~62 plateau, best 63.20@ep100 유지. 목표 Day-Val 70까지 -8, DGFusion test 56.7까지 -2.4. |
| 2026-06-30 12:17 | 150 (⚠️정체) | 62.15 (best **63.20 @ep100**) | 53.93 (best **54.34 @ep146**) | 우리 rank 3개(844809-11) mem 131GB 점유 but **util ~8%**; 타 사용자 G2,3 100%@180GB | ⚠️ alive but **STALL** — train.log 11:03(ep150 Test)부터 **1h14m 정지** | **정체(stall) 의심**. procs/GPU mem 유지=크래시 아님. 단 util 98%→8% 붕괴+로그 1h14m 정지 → 공유박스 혼잡(타 사용자 G2,3 풀점유)으로 CPU/IO/PCIe 굶는 soft-stall 추정. ckpt(last_checkpoint.pth) 보존되어 재시작 시 ep150 부근 resume 가능. |
| 2026-06-30 12:37 | 150 (🔴정체 악화) | 62.15 (best **63.20 @ep100**) | 53.93 (best **54.34 @ep146**) | G5,6,7 util **100%**@131GB(우리 rank), G4 8%@39GB; 타 G2,3 100%@180GB | 🔴 procs 53 alive(D-state 없음)이나 **진척 0** | **정체 악화**. train.log 11:03부터 **1h34m 정지**(직전 1h14m→ 더 길어짐). GPU 100%인데 epoch 진척 0 → **NCCL collective busy-wait(hang은 100% util로 보임)** 또는 심각 starvation. best 무변동. **→ 사실상 진행 정지. last_checkpoint.pth(ep150 부근)로 빈 GPU 확보 후 재시작 권고.** |

> ❗ **12:48 정정 (오진 철회)**: 위 12:17·12:37 "정체/STALL" 판정은 **오진**이다. B200에서 실제 도는 프로세스의 `--cfg`를 확인한 결과 **P30**(`b200-deliver_rgbdel_P30_physaug.yaml`, nproc=4, ~11:20 시작)이었다. **P29는 정체가 아니라 ep150에서 종료**(P30 띄우려 11:03경 수동 중단)됐고, train.log가 11:03에 멈춘 것은 P29가 끝났기 때문. 내가 본 GPU util 8%↔100% 변동·메모리 점유는 전부 **P30의 train/eval**이었다. pid 844809-11도 P30. → P29 추적 종료(최종 best Val 63.20@ep100 / Test 54.34@ep146 보존), B200 현행 run은 아래 **RUN-4(P30)**.

## RUN-3 · Jarvis P29-Det 객체검출 (poongsan indoor RGB+LiDAR+Thermal)

- **서버/소유자**: jarvis (`jemo_maeng`, 172.27.183.201, 8×RTX4090 24GB), repo `/home/jemo_maeng/src/drone-MemorySAM`, branch `worktree-p29-det`
- **config**: `configs/det/det_P29_indoor_jarvis.yaml` (EPOCHS=50, WARMUP=5, SAVE_INTERVAL=5, 10 det classes). RBMA(P28) backbone + FPN+FCOS. torchrun nproc=6, GPU 0,1,3,5,6,7.
- **출력**: `outputs/det/det_P29_indoor_jarvis/` (ckpt + wandb). 데이터 로컬 `/SSDd/jemo_maeng/dset/poongsan/`. wandb project **`p29-det`** (마지막 실행 online, run f6swzimr).
- **⚠️ 진척 파싱 주의**: stdout이 tqdm 비-tty라 `logs/det_P29_indoor_jarvis_ddp.log`에는 dataset init + wandb + grad-stride 경고만 남고 **epoch/loss 숫자는 안 남는다**. 살아있는지는 (a) GPU util 변동 (b) .log mtime 신선도(=backward 경고 계속 출력) (c) `outputs/det/.../*.pth` ckpt로 판정. 정확한 epoch/loss는 **online wandb p29-det**에서 확인.

| 점검 시각(KST) | 진척 | GPU(util/mem) | 프로세스 | 상태 판정 |
|---|---|---|---|---|
| 2026-06-30 01:56 | epoch 미상(.log 미파싱) · ckpt 0개 | G0,1,3,5,6,7 util 9~97% 변동 / ~17.7GB each (G2/G4는 타 작업) | alive — DDP nproc=6 (4171202), D-state 없음, .log 1분전 갱신 | **정상 iterate 중**. tmux상 여러 차례 ^C 후 재실행 흔적, 현 live run=master_port 29555. ckpt 0개는 restart 직후 가능성 → 다음 점검 때 wandb로 epoch/mAP 확인 + ckpt 생성 여부 확인 필요. |
| 2026-06-30 10:37 | 🔴 **사망** (03:09:30) · ckpt 0개 | 점유 GPU(0,1,3,5,6,7) 전부 해제, idle ~24MiB | 🔴 dead — train_det proc 0개 | **사망**. 01:50 시작 후 ~1h19m 만에 **NCCL collective timeout**(rank4=GPU6, last NCCL work 53404)으로 전 rank SIGABRT(-6) @03:09:30. **체크포인트 0개 → 진척 전부 소실**. tmux 자동 재시작 안 됨. |
| 2026-06-30 12:17 | 🟢 **재시작됨**(11:17, nproc=**5**, online wandb run wjvo696y) · ckpt 0개(재시작 ~1h) | G3-7 util 9~79% 변동 ~17.7GB each | alive — train_det 26 proc, D-state 없음, .log 12:16 신선 | **재가동 정상**. (사용자/외부가 재시작; 직전 11:02 run euheovtm 1회 더 있었음.) nproc 6→5로 축소 — 03:09 NCCL timeout 유발 GPU 회피 목적 추정. 무크래시 ~1h 경과(직전 사망은 ~1h19m 지점) → 다음 점검에서 1h19m 넘겨 생존+ckpt(ep5) 생성 확인 필요. |
| 2026-06-30 12:37 | 🟢 생존(etime **1h19m33s**) · ckpt 0개 | G3-7 util 40~93% 변동 ~17.7GB each (G1 타 사용자) | alive — train_det 26 proc, D-state 없음, .log 12:37 신선, **신규 크래시 없음** | **정상(고비 통과)**. 직전 사망 지점(~1h19m30s)을 막 통과했는데 크래시 無 → 11:17 run은 직전 NCCL timeout 재발 안 함(nproc 6→5 효과 가능). ⚠️ 단 ckpt 여전히 0개(ep5 저장 시점 추정인데 미생성) → 저장 경로/주기 또는 epoch 진행 느림 점검 필요. |

> 🔴 **10:37 Jarvis P29-Det 사망 분석**: torchrun `ChildFailedError` — root cause rank1, 전 rank exitcode -6(SIGABRT) @2026-06-30 03:09:30. NCCL `ProcessGroupNCCL` watchdog가 **collective timeout**(rank4 발신, last enqueued==last completed NCCL work 53404 → 한 rank가 다음 collective에 도달 못함) 감지 후 flight-recorder dump→abort. 전형적 DDP rank desync: 메모리 노트대로 `find_unused_parameters=True`(per_modal_decoders/SQG가 det loss grad 없음)인데 **배치마다 unused 파라미터 집합이 달라지면 all-reduce bucket 순서가 rank간 어긋나 collective hang** 가능 → 1순위 의심. (또는 특정 rank 데이터 hang/느림.) **체크포인트 0개**라 재시작=처음부터. 
> | **권고**(사용자 결정 필요, 자동 재시작 안 함): ① 단순 재시작은 같은 NCCL timeout 재발 위험 → 먼저 `find_unused_parameters` desync 점검(또는 static_graph=True/모든 출력에 loss 연결). ② 우회로 `NCCL_TIMEOUT`/`TORCH_NCCL_BLOCKING_WAIT` 상향 또는 단일 GPU 디버그. ③ 재시작 시 SAVE_INTERVAL 도달 전 사망 방지 위해 초반 ckpt 저장 확인. **재시작/수정 여부는 사용자 지시 대기.**
>
> ⚠️ **01:56 TODO(다음 점검)**: ① Jarvis P29-Det 실제 epoch/loss/mAP를 online wandb `p29-det`(run f6swzimr)에서 확인 — .log로는 불가. ② SAVE_INTERVAL=5인데 ckpt 0개 → ep5 도달 후 `outputs/det/det_P29_indoor_jarvis/*.pth` 생성 확인. 안 생기면 저장 경로/권한 점검. ③ B200 P29는 best 갱신 추이(val 63.2 / test 54.21 넘는지)만 추적.

---

## RUN-4 · B200 P30 RBMA seg (DELIVER) — P29 종료 후 신규 run

- **서버/소유자**: B200 (`gm_huis`), repo `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/b200-deliver_rgbdel_P30_physaug.yaml` (06-28 22:42 생성, EPOCHS=200, WARMUP=10, warmuppolylr). torchrun **nproc=4**, ~11:20 시작.
- **출력**: `outputs/MMSamP30/b200_deliver_rgbdel_P30_physaug/DELIVER_CMNeXt-B2_idel/` (`train.log`, `epochN_*`/`test_epochN_*` ckpt, `last_checkpoint.pth`)
- **P29와의 차이**: 아키텍처/하이퍼 diff 미확인(다음에 config diff 확인 필요). P29 최종 best = Val 63.20@ep100 / Test 54.34@ep146.

| 점검 시각(KST) | epoch | Day-Val mIoU | Test mIoU | best | GPU(util/mem) | 프로세스 | 상태 판정 |
|---|---|---|---|---|---|---|---|
| 2026-06-30 12:48 | 4 (학습중) | 13.76 (best 13.76@ep4) | 13.77 (best 13.77@ep4) | val ep4 / test ep4 | nproc=4, G(공유) train/eval 전환으로 util 변동, ~131GB/rank | alive — DDP nproc=4, D-state 없음, .log 12:46 신선 | **정상(초기 ramp-up)**. fresh run, ep2 8.40→ep4 13.76 정상 상승. ckpt 정상 저장(epoch2/4/last). ~18.5분/epoch → 200ep까지 ~60h. 다음 점검부터 P30 추적. |
