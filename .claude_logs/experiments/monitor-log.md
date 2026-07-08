---
legacy_id: 15
legacy_file: 15_training_monitor_log.md
moved: 2026-07-08
---

# 학습 모니터 로그 (Training Monitor Log)

> 생성: 2026-06-24
> **이 파일은 `/loop` 모니터 세션이 주기적으로 append하고, 모든 세션이 읽어 분석·판단·개선에 쓰는 공유 로그다.**
> loop 세션의 채팅은 다른 세션에 안 보이지만, 여기 기록된 내용은 `.claude_logs` init 규칙을 통해 전 세션이 공유한다.
> 규칙: ① 매 점검마다 한 줄 timestamped 엔트리 추가(append-only, 과거 줄 수정 금지). ② 이상징후(사망/정체/완료/신기록)는 엔트리 아래 `> ⚠️`로 강조. ③ 학습 종료/사망 시 [status/current.md](../status/current.md) 스냅샷의 해당 트랙도 갱신.

---

## RUN-1 · B200 P28 RBMA (DELIVER)

- **서버/소유자**: B200 (unix user `gm_huis`), repo `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/b200-deliver_rgbdel_P28_physaug.yaml` (순수 RBMA, AMF_MODE=uniform, λ_bias init 1.0, 4모달 img/depth/event/lidar, 목표 200 ep)
- **출력**: `outputs/MMSamP28/b200_deliver_rgbdel_P28_physaug/DELIVER_CMNeXt-B2_idel/` (`train.log`, `epochN_<val>_topK…pth`, `test_epochN_<test>…pth`)
- **비교 기준**: 직접 경쟁군(Cluster B, test) DGFusion 56.7 / CAFuser 55.6 · 구조적 base(Cluster A) MemorySAM val 65.38 — 자세히는 [research/novelty-and-related-work.md](../research/novelty-and-related-work.md).

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
- **비교 기준**: 직접경쟁군 DGFusion test 56.7 / CAFuser 55.6 — [research/novelty-and-related-work.md](../research/novelty-and-related-work.md).

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
| 2026-06-30 14:37 | 🟢 생존(etime **3h19m**, ~ep9-10) · **ckpt 정상** | G3,4,5,6,7 util 8~91% 변동 ~17.7GB each (G1 타 사용자) | alive — train_det 26 proc(nproc=5), D-state 없음, .log 14:37 신선, 크래시 없음 | **정상 안정화**. 11:17 run 3h19m 연속 생존(직전 사망지점 1h19m 훌쩍 넘김). **체크포인트 저장 시작**: epoch9(14:32)·epoch4·best_checkpoint(SAVE_INTERVAL=5 정상 작동). 직전 ckpt 0개 우려 해소. epoch/mAP 수치는 online wandb p29-det 참조. |
| 2026-06-30 16:37 | 🟢 **재구성**(16:12 재시작, nproc 5→4, resume) · ep11 | G0,1,2 90~100%@~18.8GB(새 run), G3-7 잔여/타작업 | alive — train_det 21 proc(nproc=4), D-state 없음, 신규 .log 16:37 신선 | **정상(의도적 재시작)**. 11:17 run을 **16:09 사용자 수동 중단(SIGINT)** 후 **16:12 `--resume epoch9_checkpoint.pth`로 재시작(진척 유지), nproc=5→4**. tmux: Epoch[11] 25% 1.33it/s loss 정상. 크래시 아님. ⚠️ **로그파일 변경**: 이제 `logs/det_P29_indoor_jarvis_postboot_20260630_161155.log`(기존 `_ddp.log`는 정지). 모니터는 `logs/*det*postboot*.log` 최신 또는 tmux pane으로 epoch 확인. |
| 2026-06-30 18:37 | 🟢 정상(etime **2h25m**, **Epoch[16]** 57%) · ckpt epoch14 | G0,2,4 64~88%@~18.9GB(우리 rank, nproc=4); 타 작업 잔여 | alive — train_det 21 proc, D-state 없음, .log 18:37 신선, 크래시 없음 | **정상 진행**. 16:12 resume run(--resume epoch9) 연속 가동. tmux Epoch[16] 1.3it/s loss 정상. **ckpt 정상 저장**: epoch14·best(18:05), epoch9/4. EPOCHS=50 → 현재 ep16, ~07-01 06시경 종료 예상. |
| 2026-06-30 20:37 | 🟢 정상(etime **4h25m**, **Epoch[21]** 96%) · ckpt epoch19 | G0-4 91~97%@~18.9GB(우리 rank, nproc=4) | alive — train_det 21 proc, D-state 없음, .log 20:37 신선, 크래시 없음 | **정상 진행**. 16:12 resume run 연속. tmux Epoch[21] 1.38it/s. ckpt epoch19(19:58)·14·best 저장. ⚠️ best_checkpoint는 18:05(epoch14) 이후 미갱신 → mAP 정체 가능(wandb 확인 권장). EPOCHS=50 → ~07-01 06시경 종료. |
| 2026-06-30 22:06 | 🔴 epoch은 22(Resumed from ep10)지만 **AP≈0** | G0-4 91~97%@~18.9GB(우리 rank, nproc=4) | alive — 21 proc, D-state 없음, .log 22:06 신선, 크래시 없음 | **🔴 검출 학습 사실상 실패**. epoch 카운터는 정상(resume ep10→현재 ep22)이나 **Val AP=0.0058 / AP50=0.0152(1.5%) / AP75=0.0035**, best AP 0.0032→0.0058로만 미동 → mAP≈0(random 수준). best_ckpt가 ep14에서 정지한 원인. tmux loss에 `n_pos=0` 배치 빈발 → anchor/타깃 할당 또는 FCOS box stride 정렬 문제 의심. **이 postboot run은 wandb OFF**(No API key)라 p29-det에 곡선 없음. |
| 2026-06-30 21:05 | 🔄 **v2 신규 run**(20:49 시작, fresh **Epoch[0] 92%** 1346/1466, ckpt 0개) | G0/2/4 등 4 rank util 72~96%@18.5GB(우리 rank 1010296-99); 타 사용자 991218·56297 | alive — train_det(pgid **1010230**, nproc=**4**, port 29531) etime 16m, D-state 없음, .log(`..._v2_20260630_204907.log`) 21:05 신선 | **의도적 재시작 — 데이터 교체(크래시 아님)**. 16:12 resume run(ep21)을 20:48 중단 후 **config `det_P29_indoor_jarvis_v2.yaml`로 fresh 학습**. v2 = clean re-label: 기존 `_det_splits`의 **52% empty-frame(미레이블 정탐=false neg)이 batch=1 절반 스텝의 정탐을 억제 → AP≈0** 진단 → 새 라벨셋(`poongsan_v2`, 8캡처 empty 0%)·캡처단위 holdout(test=capture_115206+114808)으로 교체. 하이퍼파라미터 동일(EPOCHS50/WARMUP5/SAVE_INTERVAL5/batch1). loss ep0서 2.8→0.8 정상. wandb 이제 **offline**(kuydyd6a). |

> 🔴 **10:37 Jarvis P29-Det 사망 분석**: torchrun `ChildFailedError` — root cause rank1, 전 rank exitcode -6(SIGABRT) @2026-06-30 03:09:30. NCCL `ProcessGroupNCCL` watchdog가 **collective timeout**(rank4 발신, last enqueued==last completed NCCL work 53404 → 한 rank가 다음 collective에 도달 못함) 감지 후 flight-recorder dump→abort. 전형적 DDP rank desync: 메모리 노트대로 `find_unused_parameters=True`(per_modal_decoders/SQG가 det loss grad 없음)인데 **배치마다 unused 파라미터 집합이 달라지면 all-reduce bucket 순서가 rank간 어긋나 collective hang** 가능 → 1순위 의심. (또는 특정 rank 데이터 hang/느림.) **체크포인트 0개**라 재시작=처음부터. 
> | **권고**(사용자 결정 필요, 자동 재시작 안 함): ① 단순 재시작은 같은 NCCL timeout 재발 위험 → 먼저 `find_unused_parameters` desync 점검(또는 static_graph=True/모든 출력에 loss 연결). ② 우회로 `NCCL_TIMEOUT`/`TORCH_NCCL_BLOCKING_WAIT` 상향 또는 단일 GPU 디버그. ③ 재시작 시 SAVE_INTERVAL 도달 전 사망 방지 위해 초반 ckpt 저장 확인. **재시작/수정 여부는 사용자 지시 대기.**
>
> 🔴 **2026-06-30 22:06 — Jarvis P29-Det 학습 실패 정황(AP≈0)**: epoch은 진행되나(resume ep10→ep22) 검출 성능이 random 수준에 고착(Val AP 0.0058, AP50 1.5%). best AP가 12 epoch 동안 0.0032→0.0058만 움직임. 의심 원인: ① FCOS box stride/anchor-free 타깃 할당 오류(메모리 노트: 과거 strides [16,32,64]→[4,8,16] 4× 정렬버그 수정 이력 — 회귀 여부 확인) ② `n_pos=0` 배치 빈발 = positive 매칭 실패 ③ det loss는 내려가나 mAP 안 오르는 전형적 head/디코드 정렬 문제. **권고: 사용자에게 보고 후, 원하면 objdet/models det head·box decode·assign 로직 점검.** wandb OFF라 곡선 없음 → 진단은 로그 Val AP 라인 의존.
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
| 2026-06-30 14:37 | 10 (학습중) | **31.11** (best 31.11@ep10) | 23.60 (best 23.60@ep8) | val ep10 / test ep8 | G5,6,7 100%@131GB(우리 rank)+G4 7%@39GB; 타 G2,3; nproc=4 | alive — 53 proc, D-state 없음, .log 14:31 신선(6분전) | **정상 상승 중**. Day-Val ep4→10: 13.76→20.48→26.10→**31.11**, Test ep4→8: 13.77→17.70→**23.60** 매 평가 신기록. ~19분/epoch. 아직 초반(warmup 10ep), P29 best(val 63.2)까지는 한참 남음. |
| 2026-06-30 16:37 | 16 (학습중) | **37.09** (best 37.09@ep16) | **32.44** (best 32.44@ep16) | val ep16 / test ep16 | G4-7 19~92%@131GB(우리 rank), 타 G2,3 해제 | alive — 53 proc, D-state 없음, .log 16:34 신선(3분전) | **정상 상승 중**. Day-Val ep10→16: 31.11→33.89→**37.09**, Test ep8→16: 23.60→30.57→**32.44** 매 평가 신기록. ~19분/epoch. warmup(10ep) 지나 본격 상승 구간. |
| 2026-06-30 18:37 | 22 (학습중) | **40.11** (best 40.11@ep22) | 33.49 (best **33.98@ep20**) | val ep22 / test ep20 | G4-7 8%@131-146GB(우리 rank, epoch 경계); G3 타 사용자 | alive — 53 proc, D-state 없음, .log 18:29 신선(8분전) | **정상**. Day-Val ep16→22: 37.09→39.78→**40.11** 상승 지속. Test ep18→22: 33.26→**33.98(ep20)**→33.49 — ~34 부근 횡보 시작. ~19분/epoch. |
| 2026-06-30 20:37 | 28 (학습중) | 38.77 (best 40.15@ep24) | **35.51** (best **35.51@ep28** 신기록) | val ep24 / test ep28 | G4-7 99~100%@131GB(우리 rank); G3 타 사용자 | alive — 53 proc, D-state 없음, .log 20:23 신선(14분전) | **정상**. Test ep24→28: 35.16→34.71→**35.51(신기록)** 상승. Day-Val 40.15(ep24)→39.76→38.77 횡보/소폭 하락(초기 변동). ~19분/epoch. |
| 2026-06-30 22:06 | 32 (학습중) | **43.51** (best 43.51@ep32) | 36.41 (best **36.70@ep30**) | val ep32 / test ep30 | G5,6,7 100%@131GB(우리 rank), G4 8% | alive — 53 proc, D-state 없음, .log 21:40 신선 | **정상**. Day-Val ep28→32: 38.77→…→**43.51** 회복·상승, Test ep28→32: 35.51→**36.70(ep30)**→36.41 상승. ~19분/epoch. |
| 2026-06-30 22:13 | 34 (학습중) | **43.99** (best 43.99@ep34 신기록) | 36.41 (best 36.70@ep30) | val ep34 / test ep30 | G5,6,7 100%@131GB; nproc=4 | alive — 53 proc, .log 22:10 신선 | **정상**. Day-Val ep32→34: 43.51→**43.99** 상승 지속(신기록). Test ~36.5 횡보. ~19분/epoch. |
| 2026-06-30 22:37 | 34 (ep36 학습중) | 43.99 (best 43.99@ep34) | **37.80** (best **37.80@ep34** 신기록) | val ep34 / test ep34 | G5,6,7 100%@131GB; G4 8%; G3 타 사용자 | alive — 53 proc, D-state 없음, .log 22:18 갱신(ep34 eval) | **정상**. Test ep30→34: 36.70→**37.80(신기록)** 상승. Day-Val 43.99@ep34. ~19분/epoch. |
| 2026-07-01 00:37 | 40 (ep42 학습중) | 44.37 (best **46.06@ep38**) | **38.33** (best **38.33@ep40** 신기록) | val ep38 / test ep40 | G5,6,7 100%@131GB; G4 8%; G3 타 사용자 | alive — 53 proc, D-state 없음, .log 00:13 갱신(ep40 eval) | **정상 상승**. Day-Val ep34→38: 43.99→**46.06(신고점)**. Test ep34→40: 37.80→38.27→**38.33** 연속 신기록. ~19분/epoch. |
| 2026-07-01 02:37 | 48 (ep50 학습중) | **46.43** (best 46.43@ep48 신기록) | 39.94 (best **39.94@ep46**) | val ep48 / test ep46 | G5,6,7 100%@131GB; proc 1개 순간 D(ckpt I/O) | alive — 53 proc, .log 02:37 신선 | **정상 상승**. Day-Val ep38→48: 46.06→**46.43(신고점)**. Test ep40→46: 38.33→38.43→**39.94** 연속 신기록. ~19분/epoch. |
| 2026-07-01 03:43 | 50 (ep52 학습중) | 45.34 (best 46.43@ep48) | **40.72** (best **40.72@ep50** 신기록, 40돌파) | val ep48 / test ep50 | G5,6,7 100%@131GB | alive — 53 proc, .log 03:23 신선 | **정상**. Test ep46→50: 39.94→**40.72**(첫 40대). Day-Val 46.43@ep48 유지(ep50 45.34). ~19분/epoch, ep50/200. |
| 2026-07-01 04:31 | 52 (ep54 학습중) | 46.29 (best 46.43@ep48) | 40.08 (best **40.72@ep50**) | val ep48 / test ep50 | G4-7 99~100%@131GB | alive — 53 proc, .log 04:01 신선 | **정상(횡보 시작)**. Day-Val ~46(best 46.43@ep48), Test ~40(best 40.72@ep50). ep48~52 신기록 없이 횡보. ~19분/epoch, ep52/200. |
| 2026-07-01 06:31 | 60 (ep62 학습중) | 46.68 (best **47.31@ep56**) | 41.22 (best **41.22@ep58**) | val ep56 / test ep58 | G4-7 99~100%@131GB | alive — 53 proc, .log 06:24 신선 | **정상 상승**. Day-Val ep48→56: 46.43→**47.31(신고점)**. Test ep50→58: 40.72→40.88→**41.22** 연속 신기록. ~19분/epoch, ep60/200. |
| 2026-07-01 08:31 | 66 (ep68 학습중) | **48.48** (best 48.48@ep66 신기록) | **41.63** (best **41.63@ep66** 신기록) | val ep66 / test ep66 | G4-7 1~8%@131-146GB(epoch경계); proc 53 | alive — .log 08:25 신선 | **정상 강상승**. Day-Val ep56→66: 47.31→**48.48**. Test ep58→66: 41.22→**41.63** 동반 신기록(ep66 val·test 둘 다 best). ~19분/epoch, ep66/200. |
| 2026-07-01 10:31 | 72 (ep74 학습중) | 47.40 (best **48.48@ep66**) | 40.55 (best **41.63@ep66**) | val ep66 / test ep66 | G5,6,7 100%@131GB; G4 37% | alive — 53 proc, .log 10:19 신선 | **정상(ep66 후 소폭 pullback)**. ep68~72 신기록 없음(Day-Val 46.9~47.4, Test 40.5~41.3). best 48.48/41.63@ep66 유지. ~19분/epoch, ep72/200. |
| 2026-07-01 12:31 | 78 (ep80 학습중) | 47.94 (best **48.48@ep66**) | **41.95** (best **41.95@ep78** 신기록) | val ep66 / test ep78 | G5,6,7 100%@131GB; G4 99% | alive — 53 proc, .log 12:12 신선 | **정상**. Test ep66→78: 41.63→41.60→**41.95** 완만 신기록. Day-Val ~48 횡보(best 48.48@ep66). ~19분/epoch, ep78/200. |
| 2026-07-01 14:31 | 84 (ep86 학습중) | **48.77** (best 48.77@ep84 신기록) | **42.78** (best **42.78@ep84** 신기록) | val ep84 / test ep84 | G5,6,7 100%@131GB; G4 99% | alive — 53 proc, .log 14:06 신선 | **정상 강상승**. Day-Val ep66→84: 48.48→**48.77**. Test ep78→84: 41.95→42.16→**42.78** 연속 신기록(ep84 val·test 동반 best). ~19분/epoch, ep84/200. |
| 2026-07-01 16:31 | 92 (ep94 학습중) | **49.08** (best 49.08@ep92 신기록) | 42.38 (best **42.90@ep88**) | val ep92 / test ep88 | G4-7 99~100%@131GB | alive — 53 proc, .log 16:29 신선 | **정상 상승**. Day-Val ep84→92: 48.77→**49.08(신고점)**. Test ep84→88: 42.78→**42.90**(ep90 42.38 소폭↓). ~19분/epoch, ep92/200. |
| 2026-07-01 18:27 | 96 (ep98 학습중) | **49.25** (best 49.25@ep96 신기록) | **43.48** (best **43.48@ep96** 신기록) | val ep96 / test ep96 | G5,6,7 100%@131GB; G4 99% | alive — 53 proc, .log 18:23 신선 | **정상 상승**. Day-Val ep92→96: 49.08→**49.25**. Test ep88→96: 42.90→**43.48**(ep96 val·test 동반 best). ~19분/epoch, ep96/200. |
| 2026-07-01 21:14 | 106 (ep108 학습중) | 48.50 (best **49.25@ep96**) | 42.79 (best **43.48@ep96**) | val ep96 / test ep96 | G4-7 3~15%@131-146GB(epoch경계); 53 proc | alive — .log 21:03 신선 | **정상(ep96 후 횡보)**. ep98~106 신기록 없음(Day-Val 48.0~48.5, Test 42.2~42.8). best 49.25/43.48@ep96 유지. ~19분/epoch, ep106/200. |
| 2026-07-01 22:20 | 110 (ep112 학습중) | 47.75 (best **49.25@ep96**) | 42.18 (best **43.48@ep96**) | val ep96 / test ep96 | G4-7 80~96%@131-146GB | alive — 53 proc, .log 22:19 신선 | **정상(ep96 후 횡보~소폭↓)**. ep98~110 신기록 없음(Day-Val 47.7~48.4, Test 42.2~42.8). best 49.25/43.48@ep96 유지. ~19분/epoch, ep110/200. |
| 2026-07-02 00:13 | 116 (ep118 학습중) | 48.35 (best **49.25@ep96**) | 42.79 (best **43.48@ep96**) | val ep96 / test ep96 | G5,6,7 100%@131GB | alive — 53 proc, .log 00:05 신선 | **정상(ep96 후 횡보)**. ep98~116 신기록 없음(~48/~42.8). best 49.25/43.48@ep96 유지. ~19분/epoch, ep116/200. |
| 2026-07-02 08:53 | 142 (ep144 학습중) | 49.10 (best **49.76@ep136**) | 43.34 (best **43.49@ep116**) | val ep136 / test ep116 | G3-7 99~100%@131GB | alive — 53 proc, .log 08:28 신선 | **정상 상승**. Day-Val ep116→136: → **49.76(신고점)**. Test ~43(best 43.49@ep116). ~19분/epoch, ep142/200. |
| 2026-07-02 10:18 | 148 (ep150 학습중) | 48.76 (best **49.76@ep136**) | 44.10 (best **44.10@ep146** 신기록) | val ep136 / test ep146 | G3-7 8~100%@131GB | alive — 53 proc, .log 10:14 신선 | **정상**. Test ep144→146: 43.54→**44.10**(44 돌파, 신기록). Day-Val ~48.7(best 49.76@ep136). ~19분/epoch, ep148/200. |
| 2026-07-02 12:13 | 154 (ep156 학습중) | 48.23 (best **49.76@ep136**) | 44.02 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 7~100%@131GB | alive — 53 proc, .log 12:08 신선 | **정상(횡보)**. ep148~154 신기록 없음(Day-Val ~48.2, Test ~44). best 49.76/44.10 유지. ~19분/epoch, ep154/200. |
| 2026-07-02 14:13 | 160 (ep162 학습중) | 47.79 (best **49.76@ep136**) | 43.73 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 96~99%@131GB | alive — 53 proc, .log 14:10 신선 | **정상(횡보)**. ep148~160 신기록 없음(Day-Val ~48, Test ~43.7). best 49.76/44.10 유지. ~19분/epoch, ep160/200. |
| 2026-07-02 16:12 | 166 (ep168 학습중) | 48.47 (best **49.76@ep136**) | 43.61 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 95~98%@114-146GB | alive — 53 proc, .log 16:03 신선 | **정상(횡보)**. ep148~166 신기록 없음(Day-Val ~48, Test ~43.6). best 49.76/44.10 유지. ~19분/epoch, ep166/200. |
| 2026-07-02 18:12 | 172 (ep174 학습중) | 48.49 (best **49.76@ep136**) | 43.60 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 8~100%@131GB | alive — 53 proc, .log 17:58 신선 | **정상(횡보)**. ep148~172 신기록 없음(Day-Val ~48, Test ~43.6). best 49.76/44.10 유지. ~19분/epoch, ep172/200. |
| 2026-07-02 20:13 | 178 (ep180 학습중) | 49.01 (best **49.76@ep136**) | 43.86 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 8~100%@131GB | alive — 53 proc, .log 19:52 신선 | **정상(횡보)**. ep148~178 신기록 없음(Day-Val ~49, Test ~43.9). best 49.76/44.10 유지. ~19분/epoch, ep178/200. |
| 2026-07-02 22:13 | 184 (ep186 학습중) | 48.96 (best **49.76@ep136**) | 43.62 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 99~100%@131GB | alive — 53 proc, .log 21:46 신선 | **정상(횡보)**. ep148~184 신기록 없음(Day-Val ~49, Test ~43.7). best 49.76/44.10 유지. ~19분/epoch, ep184/200(→~03시 종료). |
| 2026-07-03 00:13 | 192 (ep194 학습중) | 48.55 (best **49.76@ep136**) | 44.03 (best **44.10@ep146**) | val ep136 / test ep146 | G3-7 8~100%@131GB | alive — 53 proc, .log 00:09 신선 | **정상(횡보)**. best 49.76/44.10 유지. ep192/200 → ~03시 종료. **⚠️P31 auto-launch watcher 대기 중**(이전 Claude세션 `~/launch_p31_after_p30.sh`: P30 종료+GPU4-7 free 시 P31(EPOCHS=200) 자동 시작). |
| 2026-07-03 02:13 | 198 (종료임박) | 48.36 (best **49.76@ep136**) | 43.95 (best **44.10@ep146**) | val ep136 / test ep146 | G5,6,7 62~88%@131GB | alive — 53 proc, .log 02:11 신선 | **정상**. ep198/200 → ~02:40 종료. best 49.76/44.10 최종 유력. 종료 직후 **P31 watcher가 GPU4-7에 P31 자동시작**(아직 watcher 대기). |
| 2026-07-03 04:13 | 🏁 **종료**(ep200, ~02:30) | 최종 best **Day-Val 49.76@ep136 / Test 44.10@ep146** | best_ckpt 보존 | proc 0(P31로 대체) | **🏁 완주**. P30 seg 200ep 완료. P29 seg(63.20/54.34) 대비 크게 낮음(-13.4/-10.2). GPU 4-7은 P31이 승계. |
| 2026-06-30 21:05 | 30 (학습중) | **42.48** (best **42.48@ep30** 신기록) | **36.70** (best **36.70@ep30** 신기록) | val ep30 / test ep30 | G4-7 97~98%@131GB(우리 rank 844809-12 R-state); 타 G2,3 100% | alive — 단일 DDP pgid 844654, nproc=4, 54 proc, .log 21:02 신선(3분전) | **정상·동반 신기록**. ep30에서 **Day-Val·Test 동시 신기록**: Day-Val 38.77(ep28)→**42.48**(ep24 plateau 40.15 돌파), Test 35.51(ep28)→**36.70**. 직전 ep24~28 Day-Val 횡보(40.15→39.76→38.77)를 ep30이 깨고 상승. train loss 1.289→1.271 하락 지속. ckpt 정상(test_epoch30_36.7_top1·epoch30_42.48_top1·last·periodic, 20:53~21:02 저장). ~19분/epoch, EPOCHS=200 중 ep30. |

> ❗ **22:13 정정 (22:06 보고 오류 + 사용자 분석 검증)**: 위 22:06 RUN-3 "ep22·AP≈0" 판정은 **이미 멈춘 v1 postboot run의 잔여 로그를 현재로 오독**한 것이다. 실제로는 **v1 postboot가 20:48 종료, v2가 20:49 시작**되어 있었다. 사용자 제공 분석(v1 vs v2 구분, 저장로직 (epoch+1)%5==0→첫ckpt epoch4, v2 n_pos 건강)은 **전부 사실로 검증됨**. v1의 AP≈0(Val AP 0.0058)은 dirty data(52% 빈-이미지) 탓으로, v2(`poongsan_v2`)에서 **n_pos=0 배치 0/13518개**로 해소됨 → stride/anchor 버그가 아니라 데이터 문제였을 가능성 큼(단 확정은 v2 첫 eval로). 아래 **RUN-5(v2)**로 추적 이관, RUN-3(v1)은 종료.

---

## RUN-5 · Jarvis P29-Det **v2** (poongsan_v2 클린 데이터) — v1 dirty-data 교체

- **서버/owner**: jarvis (`jemo_maeng`), repo `/home/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/det/det_P29_indoor_jarvis_v2.yaml` (06-30 19:15 생성, EPOCHS=50, WARMUP=5, SAVE_INTERVAL=5). **ROOT=`/SSDd/jemo_maeng/dset/poongsan_v2`**(클린업, REQUIRE_ALL_MODALITIES=true). torchrun nproc=4, master_port 29531, **20:49 시작**.
- **출력**: `outputs/det/det_P29_indoor_jarvis_v2/`. 로그 `logs/det_P29_indoor_jarvis_v2_<ts>.log`(현 `_20260630_204907.log`). 이 run은 **wandb 상태 확인 필요**.
- **v1 대비**: 데이터 클린업으로 iters/epoch 1602→**1466**, **n_pos=0 배치 사라짐**(13518샘플 중 0개). v1 best AP 0.0058(≈random)에서 회복하는지가 관건.
- **저장 타이밍**: `(epoch+1)%5==0` → 첫 eval+`epoch4_checkpoint.pth`가 **epoch4(=5번째 epoch) 끝**에 처음 생성, 이후 9/14/…(0-index).

| 점검 시각(KST) | epoch | Val AP/AP50 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-06-30 22:13 | **Epoch[4] 61%** (893/1466) | (아직 없음 — epoch4 끝나야 첫 eval) | 0개(예정 ~22:20 epoch4) | G0-4 91~97%@~18.9GB, 21 proc, D-state 없음, .log 신선 | **정상 학습 중**. n_pos 건강(0개=없음), loss 0.85 정상. **첫 Val AP가 ~22:20 epoch4에서 나옴 = v2 회복/붕괴 첫 판정점.** |
| 2026-06-30 22:33 | epoch4 (eval 완료) | **AP=0.2265 / AP50=0.4384 / AP75=0.2108** | epoch4_checkpoint(첫 저장) | 정상 | **✅ 회복 확정**. v1 best AP 0.0058→**v2 0.2265(~39×)**, AP50 0.0152→**0.4384(~29×)**. 첫 평가만에 정상 수준 → **AP≈0는 dirty data(52% 빈-이미지) 문제였음 확정, stride/anchor 코드버그 아님.** 이후 45ep 더 학습 시 추가 상승 기대. |
| 2026-06-30 22:37 | Epoch[5] 24% (347/1466) | (epoch4=AP 0.2265 유지; 다음 eval epoch9) | epoch4·best(22:32) | G0-4 39~100%@~18.6GB, 21 proc, D-state 없음, .log 신선 | **정상 학습 중**. etime 1h48m 연속, 1.37it/s, 크래시 없음. epoch4 회복 확인 후 정상 진행. 다음 판정점=epoch9 Val AP(~23:10). |
| 2026-07-01 00:37 | Epoch[11] 16% | **epoch9 AP=0.2330**/AP50=0.4072/AP75=0.2410 (epoch4 0.2265→상승) | epoch9·best(00:15), epoch4 | G0,3,4 98~100%@~18.6GB, 21 proc, D-state 없음, .log 신선 | **정상 상승 중**. etime 3h48m 연속, 1.25it/s, 크래시 없음. AP epoch4 0.2265→epoch9 **0.2330**(AP75 0.21→0.24 개선). 다음 eval=epoch14. EPOCHS=50 → ~07-01 새벽 종료. |
| 2026-07-01 02:37 | Epoch[17] (ep14 eval done) | mAP 0.2377 / **mAP50 0.4058** / mAP75 0.2548 | epoch14·best(01:59) | G0-4 58~100%@~18.6GB, 21 proc, D 없음, .log 신선 | **alive·학습중이나 ⚠️목표지표 하락**. etime 5h48m 연속, 크래시 없음. |

> ✅ **22:33 해결**: v2 epoch4 첫 Val **AP=0.2265 / AP50=0.4384 / AP75=0.2108** — v1(best AP 0.0058, AP50 0.0152) 대비 ~29-39× 회복. **AP≈0 원인 = dirty data(빈-이미지 52%) 확정**, FCOS stride/anchor 코드 의심은 기각. v2 정상 학습 중(EPOCHS=50, ~07-01 새벽 종료 예상). 다음 eval=epoch9.
>
> ⚠️ **2026-07-01 02:37 — v2 목표지표(mAP50) 역행 주의**: COCO mAP는 ep4→14 0.227→0.233→0.238로 오르고 mAP75도 0.211→0.255로 오르는데, **목표인 mAP50(@IoU0.5)은 0.438→0.407→0.406으로 하락**. 즉 모델이 tight-localization 쪽으로 이동 중. ① `best_checkpoint`는 COCO mAP 기준이라 현재 ep14인데 **mAP50 기준 최선은 ep4(0.438)** → mAP50-best ckpt 별도 보관 또는 best 기준을 mAP50로 변경 검토. ② 목표 **mAP50 0.85**([[det-target-map50-085]])와 갭 ~0.41 + 추세 하락 → 단순 더 학습으로 도달 난망, 개선 레버 필요. 다음 eval=ep19/24.
>
> ⚠️ **22:13 다음 점검 필수**: v2 **epoch4 첫 Val AP**(~22:20, `epoch4_checkpoint.pth`)를 확인하라. v1 best AP 0.0058 대비 **유의미하게 오르면 데이터 문제 확정·회복**, 여전히 ≈0이면 head/stride/anchor 코드 문제로 범위 좁힘. (원하면 ckpt 저장 즉시 hinton에서 per-class AP·score분포 진단 가능.)

---

## RUN-6 · Jarvis P29-Det **v2_bundle** (수정판: letterbox+aug+ATSS) — v2 mAP50 정체 대응

- **서버/owner**: jarvis (`jemo_maeng`), repo `/home/jemo_maeng/src/drone-MemorySAM`. tmux `jemo:p29bundle`. **03:17 시작**, nproc=4, master_port 29523.
- **config**: `configs/det/det_P29_v2_bundle.yaml` (git `37811db`). ROOT=`/SSDd/.../poongsan_v2`(클린), EPOCHS=50, SAVE_INTERVAL=5, IMG_SIZE 1024². **3대 수정**: ① RESIZE_MODE=**letterbox**(기존 stretch 종횡비왜곡 → 종횡비보존+center-pad, eval invert) ② **augmentation ON**(기존 미배선=OFF) ③ ASSIGNER=**atss**(topk=9, scale=8.0; 기존 fcos center-sampling 대체). "dense-head ceiling experiment".
- **출력**: `outputs/det/det_P29_v2_bundle/`. 로그 `logs/det_P29_v2_bundle_<ts>.log`(현 `_20260701_031738.log`).
- **동기**: v2(RUN-5)가 COCO mAP는 오르나 **목표 mAP50가 0.438(ep4)→0.406(ep14) 하락** → letterbox/aug/ATSS로 mAP50 천장 올리기. **목표 mAP50 0.85**([12_..]·메모리), v2 best mAP50=0.438@ep4가 현재 기준선.
- **리포트 규칙**: det는 **mAP / mAP50(목표) / mAP75** 3종 모두 기록, mAP50 헤드라인.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-01 03:43 | Epoch[1] 41% | (아직 없음 — 첫 eval=epoch4) | 0개(예정) | G0-4,6 활성 ~18.6GB, 21 proc, D 없음, .log 신선 | **정상 초기 학습**. 03:17 시작, 1.23it/s, n_pos=9(ATSS topk), 크래시 없음. **첫 mAP50=epoch4(~04:40)** → letterbox+ATSS가 v2 baseline(mAP50 0.438@ep4) 넘는지 첫 판정점. |
| 2026-07-01 04:31 | Epoch[4] 1% | (아직 없음 — epoch4 끝 ~04:50 첫 eval) | 0개 | G0-4 활성 ~18.6GB, 21 proc, D 없음, .log 신선 | **정상 학습 중**. etime 1h13m, 1.36it/s, 크래시 없음. epoch4 진입 = **첫 mAP50 임박(~04:50)**, v2 baseline 0.438@ep4와 비교 예정. |
| 2026-07-01 06:31 | Epoch[9] 100%(ep9 eval중) | **ep4**: mAP 0.2156 / **mAP50 0.4320** / mAP75 0.1830 | epoch4·best(05:00) | G2-4 100%@~18.6GB, 9 proc(ep경계), D 없음, .log 신선 | **정상·첫판정 보류**. etime 3h13m, 크래시 없음. bundle ep4 mAP50 0.432 vs **v2 baseline ep4 0.438** → 거의 동률·미세 하회. aug ON이라 초반 lag 정상 → 추세가 관건(v2는 ep4→14 0.438→0.406 하락했음). ep9 eval 임박. |
| 2026-07-01 08:31 | Epoch[15] 43% (ep14 eval done) | **ep9 mAP 0.269 / mAP50 0.4455 / mAP75 0.283**(=현 best mAP50); ep14 0.257/0.416/0.282 | epoch14·epoch9·best(=ep9, 06:41) | G1-4 51~95%@~18.6GB, 21 proc, D 없음, .log 신선 | **✅ v2 대비 개선**. etime 5h13m, 크래시 없음. mAP50 추세 ep4 0.432→**ep9 0.4455(v2 최고 0.438 돌파)**→ep14 0.416. mAP·mAP75는 v2 확실히 상회(ep9 mAP 0.269 vs v2 0.233, mAP75 0.283 vs 0.241; ATSS+letterbox localization 개선). ⚠️ ep14 mAP50 다시 출렁(변동성 잔존). best_ckpt=ep9(이번엔 mAP50-best와 일치). 목표 0.85까지 갭 ~0.40. |
| 2026-07-01 10:31 | Epoch[21] 45% (ep19 eval done) | **ep19 mAP 0.227 / mAP50 0.3672 / mAP75 0.256** (best=ep9 0.269/**0.4455**/0.283) | epoch19·14·9·best(=ep9) | G1-4 69~94%@~18.6GB, 21 proc, D 없음, .log 신선 | **🔻 ep9 후 하락**. etime 7h13m, 크래시 없음. mAP50 ep9 0.4455→ep14 0.416→**ep19 0.367**, mAP도 0.269→0.227 동반 하락. best_ckpt=ep9 고정. |
| 2026-07-01 12:31 | Epoch[27] 37% (ep24 eval done) | **ep24 mAP 0.234 / mAP50 0.3661 / mAP75 0.262** (best=ep9 0.269/**0.4455**/0.283) | epoch24…9·best(=ep9) | G0-4 25~92%@~18.6GB, 21 proc, D 없음, .log 신선 | **🔻 하락→평탄(peak 미회복)**. etime 9h13m, 크래시 없음. mAP50 ep9 0.4455→ep19 0.367→**ep24 0.366**(평탄화, peak 대비 -0.08 고착). best_ckpt=ep9 고정. ep27/50, 남은 23ep로 ep9 재돌파 난망. |
| 2026-07-01 14:31 | Epoch[33] 20% (ep29 eval done) | **ep29 mAP 0.235 / mAP50 0.3689 / mAP75 0.258** (best=ep9 0.269/**0.4455**/0.283) | epoch29…9·best(=ep9) | G0-4 9~99%@~18.6GB, 21 proc, D 없음, .log 신선 | **🔻 평탄 지속(peak 미회복)**. etime 11h13m, 크래시 없음. mAP50 ep24 0.366→ep29 0.369, ~0.367에서 6ep째 정체. best_ckpt=ep9 고정. ep33/50, ep9(0.4455) 재돌파 사실상 무망. |
| 2026-07-01 16:31 | ⏹ **종료**(~ep37, 16:07 중단) | 최종 ep34 mAP50 0.358; **best=ep9 mAP50 0.4455**(peak 미회복 확정) | best=ep9 | — | **종료**. P30-Det(det_P30_v2)로 교체. bundle 결론: letterbox+aug+ATSS로 peak는 v2(0.438)→0.4455로 올렸으나 ep9 후 하락→평탄(~0.36), 후반 하락 미해결. → RUN-7로 이관. |

> 🔻 **2026-07-01 10:31 — bundle도 ep9 후 mAP50 하락(v2와 동일 패턴, peak만 높음)**: mAP50 ep9 **0.4455**(peak, v2 최고 0.438 상회)→ep14 0.416→ep19 **0.367**; mAP 0.269→0.227 동반. letterbox+aug+ATSS가 **peak는 올렸지만 후반 하락은 못 막음**. aug ON에도 하락 → 순수 overfit보다 LR 스케줄/학습동역학 의심(mAP50가 mAP보다 민감). best_ckpt=ep9(=현 mAP50-best). 목표 mAP50 0.85와 갭 ~0.40 + 추세 하락 → **단순 학습지속으로 도달 난망**. 검토 레버: LR/스케줄 조정, EMA, mAP50 기준 best 저장, ep9 부근 early-stop 후 fine-tune. 다음 eval=ep24.
>
> ⚠️ **03:43 다음 점검**: bundle **epoch4 첫 eval의 mAP50**을 v2 baseline **0.438@ep4**와 직접 비교하라. 넘으면 letterbox+aug+ATSS 효과 확인. 아래 RUN-5(기존 v2, 03:14 중단)는 종료.

---

## RUN-7 · Jarvis **det_P30_v2** (P30-Det: reliability-router + object-query decoder) — bundle 후속

- **서버/owner**: jarvis, repo `/home/jemo_maeng/src/drone-MemorySAM`. tmux `jemo:p30det`. **16:07 시작**, nproc=4.
- **config**: `configs/det/det_P30_v2.yaml`. ROOT=poongsan_v2(클린), **EPOCHS=40**, SAVE_INTERVAL=5. `SEG_MODEL=LoRA_Sam_P30_Det`(P30 backbone: RBMA+SDC), `DET_MODEL=MemorySAMDetectorP30`(**reliability-router fusion + object-query decoder=primary head + FCOS aux**). letterbox+aug+ATSS(bundle wins) 계승, FREEZE_BACKBONE=false.
- **출력**: `outputs/det/det_P30_v2/`. 로그 `logs/det_P30_v2_<ts>.log`(현 `_20260701_160741.log`).
- **동기**: P29-bundle이 mAP50 peak 0.4455(ep9) 후 하락·평탄(~0.36). P30 아키텍처(object-query decoder primary)로 천장 재도전. **목표 mAP50 0.85**. 비교기준: bundle best mAP50 0.4455@ep9.
- **리포트**: mAP/mAP50(목표)/mAP75 3종.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-01 16:31 | Epoch[1] 24% | (아직 없음 — 첫 eval=epoch4) | 0개 | G0-4 활성 ~19.5GB, 21 proc, D 없음, .log 신선 | **정상 초기 학습**. 16:07 시작, 1.29it/s, loss 3.66(새 object-query head 초기), n_pos=8(ATSS), 크래시 없음. 첫 mAP50=epoch4(~17:20)에서 bundle best 0.4455와 비교 예정. |
| 2026-07-01 18:23 | Epoch[6] 36% (ep4 eval done) | **ep4 mAP 0.011 / mAP50 0.0360 / mAP75 0.004** (bundle ep4=0.432) | epoch4·best(17:56) | G0-4 활성, D 없음, .log 신선 | **⚠️ 첫 eval 매우 낮음**. mAP50 0.036 = bundle ep4(0.432) 대비 ~12× 뒤짐. object-query decoder(DETR계열)라 초기 수렴 느린 건 예상되나 EPOCHS=40으론 부족 위험. loss 2.05 하강 중. 판정=추세(ep9,14). |
| 2026-07-01 18:27 | Epoch[6] 58% | ep4 mAP 0.011 / **mAP50 0.036** / mAP75 0.004 (다음 eval=ep9) | epoch4·best(17:56) | G0-4 21~99%@~19.5GB, 21 proc, D 없음, .log 신선 | **⚠️ 진행중, 판정 대기**. etime 2h20m, 크래시 없음. loss 3.29→**1.97** 하강(경미 긍정). 아직 ep4 eval(mAP50 0.036)만 → **ep9(~19:30) 기울기가 진짜 판정점**. |
| 2026-07-01 21:16 | Epoch[14] 58% (ep9 eval done) | **ep9 mAP 0.015 / mAP50 0.0482 / mAP75 0.004** (ep4 0.036 → ep9 0.048) | epoch9·best(=ep9,19:45) | G0-4 9~99%@~19.5GB, 21 proc, D 없음, .log 신선 | **🔻 상승 과도하게 느림**. etime 5h08m, 크래시 없음. mAP50 ep4→9: 0.036→0.048(+0.012/5ep). bundle ep9(0.4455) 대비 ~9× 뒤짐. 이 기울기면 ep40 끝까지 ~0.1 예상 → bundle·목표 못 따라잡을 궤도. |
| 2026-07-01 22:20 | Epoch[17] 37% (ep14 eval done) | **ep14 mAP 0.037 / mAP50 0.1031 / mAP75 0.016** (ep9 0.048→ep14 0.103) | epoch14·best(=ep14,21:35) | G0-4 9~100%@~19.5GB, 21 proc, D 없음, .log 신선 | **↗ 가속 시작(긍정)**. etime 6h14m, 크래시 없음. mAP50 ep4 0.036→ep9 0.048→**ep14 0.103**(기울기 5× 급증). DETR류 query decoder 워밍업 후 가속 전형. 아직 bundle 0.4455엔 미달이나 궤도 회복 → ep19/24 지속 가속 여부 관건. |
| 2026-07-02 00:13 | 🔴 **NaN 발산**(ep20~22) | ep19 mAP 0.047 / **mAP50 0.1384** / mAP75 0.021 (마지막 정상); 이후 loss=nan | best/epoch19(23:24) | G0-4 활성이나 무의미(가중치 NaN) | **🔴 학습 발산**. tmux `loss=nan cls=nan`. ep14→19 가속(mAP50 0.103→0.138) 직후 ep20~22 발산. best 사용가능 ckpt=**epoch19(mAP50 0.1384)**. |
| 2026-07-02 08:38 | 🟢 **복구 재기동**(ep20, resume ep19) | ep19 best **mAP50 0.1384** 유지(재개학습, 다음 eval=ep24) | epoch19 resume | G1-4 100%@~21.7GB(batch4), 21 proc, D 없음, .log 신선 | **✅ NaN 근절·정상**. AMP off(fp32)+grad-ckpt+batch4, loss finite(2.64), nan-skip 0, OOM 없음. ~28min/epoch(fp32+ckpt), ep20/40. |
| 2026-07-02 08:53 | Epoch[20] 61% (AMP-off b4 run) | 아직 없음(현 run 첫 eval=ep24); best 유지 mAP50 0.1384@ep19 | ⚠️stale epoch24~39(직전 NaN-skip run 잔재, frozen ep19) | G1-4 100%@21.7GB(b4), 21 proc, D 없음, .log 신선 | **✅ 정상(NaN 근절 유지)**. AMP-off+ckpt+b4, finite loss 446, nan-skip 0. etime 17:21, ~28min/ep. **첫 실질 eval=epoch24**(stale ckpt 덮어씀). |
| 2026-07-02 10:18 | Epoch[23] 60% (AMP-off b4) | 아직 없음(첫 eval=ep24, ~1ep 뒤); best 유지 mAP50 0.1384@ep19 | ⚠️stale epoch24~39(직전 NaN-skip 잔재) | G1-4 100%@21.7GB, 21 proc, D 없음, .log 신선 | **✅ 정상**. finite loss 2638, nan-skip 0, loss 4.08→2.0 하강(건강). etime 1h42m, ep20→23 진행. epoch24 eval에서 첫 실질 mAP50. |
| 2026-07-02 12:13 | Epoch[27] 27% (ep24 eval done) | **ep24 mAP 0.108 / mAP50 0.2285 / mAP75 0.088** (재개점 ep19 0.1384 → 상승) | epoch24·best(=ep24, 11:08, 실제값으로 덮음) | G1-4 100%@21.7GB, 21 proc, D 없음 | **✅ 정상 상승**. AMP-off b4, finite loss 5332, nan-skip 0. mAP50 ep19 0.1384→**ep24 0.2285**(+0.09/5ep) → NaN 없이 pre-NaN best 돌파·상승. 목표 0.85 갭 ~0.62. |
| 2026-07-02 14:13 | Epoch[31] 13% (ep29 eval done) | **ep29 mAP 0.108 / mAP50 0.2341 / mAP75 0.086** (ep24 0.2285→소폭↑) | epoch29·best(=ep29,13:40) | G1-4 97%@21.7GB, 21 proc, D 없음 | **✅ 정상(상승 완만)**. AMP-off b4, finite loss 8172, nan-skip 0. mAP50 ep24 0.2285→**ep29 0.2341**(+0.006, 초반 급등 후 ~0.23대 완만). small-object collapse가 천장(메모리 P29vs P30 분석). ep31/40, 목표 0.85 갭 큼. |
| 2026-07-02 16:12 | Epoch[34] 100%(eval 진입) | 최신 ep29 mAP50 0.2341(ep34 eval ~10분뒤); best=ep29 | epoch29·best, epoch24 | G1-4 8~100%@21.7GB, 9 proc(ep경계), D 없음 | **✅ 정상**. AMP-off b4, finite loss 11010, nan-skip 0. epoch34 학습완료→eval중. ep34/40 종료 임박(남은 6ep). mAP50 ~0.23대 수렴 추정. |
| 2026-07-02 18:12 | **Epoch[39] 24% = 마지막(39/40)** | **ep34 mAP 0.114 / mAP50 0.2490 / mAP75 0.091**(best); ep29 0.2341→ep34 0.2490 | epoch34·best(=ep34,16:13) | G1-4 100%@21.7GB, 21 proc, D 없음 | **✅ 정상·종료임박**. AMP-off b4, finite loss 14122, nan-skip 0. mAP50 ep24 0.2285→ep29 0.2341→**ep34 0.2490** 완만 상승. 마지막 epoch39 eval(~30분뒤)이 최종. 목표 0.85 미달(small-obj 한계), 검출 best-overall은 P29-Det ep9 0.446. |
| 2026-07-02 20:13 | 🏁 **학습완료**(ep39/40, 18:45) | **최종 ep39 mAP 0.1291 / mAP50 0.2562 / mAP75 0.1163** (best) | epoch39·best(=ep39) | proc 0, GPU 1-4 해제 | **🏁 완주(NaN·크래시 없음)**. 로그 'Training complete. Best AP: 0.1291'. mAP50 ep24 0.229→29 0.234→34 0.249→**39 0.256** 완만 상승 마감. 목표 0.85 미달·검출 best-overall=P29-Det ep9 0.446(small-obj 한계). AMP-off 복구 성공적 완주. jarvis GPU 1-4 이제 free. |

> ⚠️ **2026-07-01 18:23 — P30-Det ep4 mAP50=0.036(매우 낮음)**: bundle ep4 0.432 대비 ~12× 뒤짐. object-query decoder(DETR계열) primary head는 초기 수렴이 원래 느려 예상 범위일 수 있으나, **EPOCHS=40은 query decoder엔 짧을 수 있음**. **ep9/ep14 mAP50 기울기가 관건** — 가파른 상승=정상 수렴, 정체=헤드/loss 배선(object-query loss·matcher) 점검 필요. 참고: FCOS aux head도 있어 eval이 어느 head 기준인지 확인 가치 있음.
>
> 🔻 **2026-07-01 21:16 — P30-Det mAP50 상승 과도하게 느림(ep4 0.036→ep9 0.048)**: query decoder 느린 수렴 감안해도 slope가 너무 완만(+0.012/5ep). EPOCHS=40 끝까지 외삽 시 ~0.1로, bundle 0.4455·목표 0.85 도달 불가 궤도. **핵심 점검거리**: det_P30_v2엔 object-query decoder(primary)+**FCOS aux head** 둘 다 있는데, 보고되는 Val AP가 어느 head인지 불명 — **primary(query) head 기준이면 FCOS aux는 bundle급(0.44)일 수도** 있으니 aux head 별도 eval 권장. 그 외 object-query matcher/loss/query수/LR 점검. 계속 두기보다 원인규명 우선 권고.
>
> 🏁 **2026-07-02 20:13 — P30-Det(det_P30_v2) 학습 완료**: AMP-off 복구 run이 40 epoch 완주(NaN 0, 크래시 0). 최종 **mAP 0.1291 / mAP50 0.2562 / mAP75 0.1163**(best=ep39, ckpt epoch39/best_checkpoint). mAP50 궤적 ep19(0.138, resume)→24(0.229)→29(0.234)→34(0.249)→39(0.256). 결론: NaN 근원(fp16 overflow)은 AMP-off로 완전 해결됐고 모델은 정상 수렴했으나 **mAP50 0.256로 목표 0.85·bundle 0.446 미달** — small-object collapse(메모리 P29vsP30 분석)가 아키텍처 한계. jarvis GPU 1-4 해제. 후속 레버: query 수↑/deformable high-res/FCOS-aux가 small-obj 담당.
>
> ✅ **2026-07-02 08:38 — P30-Det NaN 복구 완료**: 안정화 패치(clamp+fp32loss+GIoU guard+grad-clip+nan-guard, commit 4a80058) 후 epoch19 resume했으나 **AMP 유지 시 여전히 100% 배치 NaN**(fp16 backbone forward overflow는 loss-fp32로 못 막음; nan-guard가 전량 skip=실효학습 0으로 확인). → **AMP off(full fp32)로 전환하니 NaN 완전 근절**(loss finite, skip 0). 부수 조정: grad-ckpt ON(fp32 메모리 대응), 근데 b1이 ~7GB뿐 → **batch 1→4**(effective 16)로 처리량 확보(~21.7GB fit). ~28min/epoch(fp32+ckpt 비용). commit d00d4d6(AMP off)·087aac8(batch4). ep19(mAP50 0.1384)에서 재개, ep20/40 진행. **교훈**: 이 P30-Det 구성은 AMP 시 fp16 overflow로 NaN → AMP off 필수.
>
> 🔴 **2026-07-02 00:13 — P30-Det loss=NaN 발산 + 원인규명 + 복구계획 승인**: ep14→19 가속(mAP50 0.103→0.1384) 직후 **ep20~22에서 loss=nan**. 근본원인(Explore 2건): ① FCOS reg 무제한출력(`fcos_head.py:141`)→AMP fp16 overflow→inf ② GIoU(`losses.py:83,92`) inf 미방어→loss_reg nan→공유 backbone 경유 cls 오염 ③ `train_det.py` grad-clip·NaN가드 부재로 증폭(LR은 ep22 ~1.1e-4로 원인 아님). **복구계획 승인**: 안정화 패치(FCOS reg clamp + loss fp32 + GIoU inf방어 + grad clip + NaN가드) 후 **epoch19_checkpoint(mAP50 0.1384)에서 resume**, AMP 유지. 상세 plan: cheerful-swinging-lollipop.md.
>
> ⚠️ **16:31 다음 점검**: det_P30_v2 **epoch4 첫 mAP50**을 bundle best **0.4455@ep9**와 비교. object-query decoder가 dense FCOS head 대비 mAP50 천장 올리는지 판정. EPOCHS=40(짧음).

---

## RUN-8 · Jarvis **det_P29_v2_bundle** resume (ep34→50 연장) — P30-Det 완료 후 재개

- **서버/owner**: jarvis, tmux `jemo:p29hold`. **2026-07-02 20:15 시작**, nproc=4, GPU 1-4, `--resume outputs/det/det_P29_v2_bundle/epoch34_checkpoint.pth`(start_epoch=35). **AMP-on**(P29-Det 계열은 NaN 없음, ~19GB).
- **config**: `configs/det/det_P29_v2_bundle.yaml`(RUN-6와 동일, letterbox+aug+ATSS FCOS dense). EPOCHS=50 → ep35~50 연장. 로그 `logs/det_P29_v2_bundle_resume_*.log`(현 `_20260702_201555.log`), 출력 `outputs/det/det_P29_v2_bundle/`.
- **맥락**: RUN-6 bundle은 07-01 ep~39까지 갔다가 P30-Det로 교체됐음. 이제 ep50까지 마저 돌리는 연장 run. bundle 최고=mAP50 **0.4455@ep9**(peak). 검출 best-overall=P29-Det ep9 0.446.
- **리포트**: mAP/mAP50(목표)/mAP75.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-02 22:13 | Epoch[40] 64% (ep39 eval done) | **ep39 mAP 0.223 / mAP50 0.3565 / mAP75 0.249** | epoch39(22:01) | G1-4 25~91%@~19GB, 21 proc, nan-skip 0 | **정상(AMP-on)**. resume ep35→현 ep40. mAP50 0.3565 = ep9 peak(0.4455) 미회복(post-ep9 하락 그대로). ep50까지 10ep 남음. 연장으로 0.4455 재돌파 여부 관찰. |
| 2026-07-03 00:13 | Epoch[46] 38% (ep44 eval done) | **ep44 mAP 0.229 / mAP50 0.3663 / mAP75 0.250** (ep39 0.3565→소폭↑) | epoch44(23:46)·39 | G1-4 39~94%@~19GB, 21 proc, nan-skip 0 | **정상**. resume run ep46/50, mAP50 0.3663 여전히 peak(0.4455@ep9) 아래. 남은 4ep. |
| 2026-07-03 02:13 | Epoch[52] 18% (ep49 eval done) | **ep49 mAP 0.223 / mAP50 0.3544 / mAP75 0.246** (ep44 0.3663→소폭↓) | epoch49(01:32)·44 | G1-4 52~64%@~19GB, 21 proc, nan-skip 0 | **정상(정체)**. resume run ep52(EPOCHS 50 넘겨 진행 — config total 재확인 요). mAP50 ~0.35대 진동, peak 0.4455 미회복. best(COCO mAP)=ep44. |
| 2026-07-03 04:13 | Epoch[58] (ep54 eval done) | **ep54 mAP 0.224 / mAP50 0.3523 / mAP75 0.248** (ep49 0.3544→평탄) | epoch54(03:17)·49 | G1-4 8~61%@~19GB, 21 proc, nan-skip 0 | **정상(정체)**. resume run ep58, mAP50 ~0.35 진동 지속(peak 0.4455 미회복). EPOCHS 50 초과 진행(60 추정). |
| 2026-07-03 06:13 | Epoch[63] 83% (ep59 eval done) | **ep59 mAP 0.248 / mAP50 0.3844 / mAP75 0.274** (ep54 0.3523→**+0.032 반등**) | epoch59(05:01)·54 | G1-4 66~91%@~19GB, 21 proc, nan-skip 0 | **정상·반등**. resume run ep63, mAP50 오랜 ~0.35 정체 후 **0.3844로 상승**(peak 0.4455 근접 시도). best(COCO mAP)=ep59. |
| 2026-07-03 08:13 | Epoch[69] 66% (ep64 eval done) | **ep64 mAP 0.221 / mAP50 0.3423 / mAP75 0.248** (ep59 0.3844→되돌림) | epoch64(06:46)·59 | G1-4 43~98%@~19GB, 21 proc, nan-skip 0 | **정상(진동)**. resume run ep69, mAP50 ~0.34-0.38 진동(ep59 0.3844 스파이크 후 ep64 0.3423). peak 0.4455 미회복, 지속개선 아님. best(COCO mAP)=ep59. |
| 2026-07-03 10:13 | Epoch[74] 100% (ep69 eval done) | **ep69 mAP 0.206 / mAP50 0.3116 / mAP75 0.230** (ep64 0.3423→하락) | epoch69(08:30)·64 | G2-4 100%@~19GB, 9 proc(ep경계), nan-skip 0 | **🔻 하락세**. resume run ep74, mAP50 ep59 0.384→64 0.342→**69 0.312** 하락 — 연장이 오히려 악화(overfit). best 가용 ckpt는 초기(ep9 peak 0.4455) 또는 ep59. 계속 연장 무의미. |
| 2026-07-03 12:13 | ⏹ **종료**(~ep74, ~10:52) | 최종 오실레이션 mAP50 ~0.31~0.38, best_ckpt(COCO mAP)=ep59(mAP50 0.384) | epoch74·69·64 | proc 0→det_P31_v3clip로 교체 | **종료**. bundle 연장(ep35~74)은 mAP50 peak(0.4455@ep9) 미회복·후반 하락. → RUN-10(det_P31_v3clip)로 이관. |

> ⚠️ **22:13**: bundle 연장(ep35~50)은 mAP50 0.3565로 여전히 peak(0.4455@ep9) 아래. 남은 10ep로 재돌파 가능성 낮음(post-ep9 하락 패턴). 목적 확인 필요할 수 있음(단순 50ep 완주 vs 개선 기대).

---

## RUN-9 · B200 **P31** RBMA seg (DELIVER) — P30 종료 후 watcher 자동 승계

- **서버/owner**: B200 (`gm_huis`). **2026-07-03 ~03:11 자동 시작**(이전 Claude세션 watcher `~/launch_p31_after_p30.sh`가 P30 종료+GPU4-7 free 감지 후 실행), torchrun nproc=4, **GPU 4,5,6,7**.
- **config**: `configs/b200-deliver_rgbdel_P31_physaug.yaml`(07-02 22:34 생성, develop @f7e3050). EPOCHS=200. P30 대비 아키/하이퍼 diff 미확인(추후 config diff).
- **출력**: `outputs/MMSamP31/b200_deliver_rgbdel_P31_physaug/DELIVER_CMNeXt-B2_idel/train.log`(Day-Val/Test), 런치로그 `logs/p31/p31_<ts>.log`.
- **비교선**: P29 seg best Val 63.20/Test 54.34 · P30 seg best Val 49.76/Test 44.10. P31이 P30 회귀를 만회하는지 관건.

| 점검 시각(KST) | epoch | Day-Val | Test | best | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|---|
| 2026-07-03 04:13 | 5/200 (초기) | (아직 미로깅) | (미로깅) | — | G5,6,7 100%@~150GB, 42 proc, .log 04:12 신선 | **정상 초기 학습**. 03:11 시작, ep5 eval 진입, loss 2.95(ep4). 첫 Day-Val/Test는 다음 점검에서. P30 대비 개선 여부 관찰. |
| 2026-07-03 06:13 | 8/200 | **49.03** (best 49.03@ep8) | 44.18 (best **45.18@ep6**) | val ep8 / test ep6 | G5,6,7 100%@~150GB, 42 proc, .log 05:47 신선 | **✅ 정상·유망**. Test ep4→6: 39.51→**45.18**(신기록), **P30 최종 best 44.10 이미 돌파(@ep6)**. Day-Val 49.03@ep8도 P30 best(49.76) 근접. ep8/200이라 상방 큼 → P31이 P30 회귀 만회 조짐. |
| 2026-07-03 08:13 | 14/200 | **57.70** (best 57.70@ep12) | **50.82** (best **50.82@ep14** 신기록) | val ep12 / test ep14 | G5,6,7 100%@~150GB, 42 proc, .log 07:44 신선 | **✅ 강상승**. Day-Val ep8→12: 49.03→**57.70**(+8.7). Test ep6→14: 45.18→50.32→**50.82** 연속신기록. **P30(49.76/44.10) 대폭 상회**, P29(63.20/54.34) 근접 중. ep14/200 상방 큼. |
| 2026-07-03 10:13 | 22/200 | **59.76** (best 59.76@ep22 신기록) | 50.24 (best **51.69@ep18**) | val ep22 / test ep18 | G5,6,7 100%@~150GB, 42 proc, .log 10:12 신선 | **✅ 계속 강상승**. Day-Val ep12→22: 57.70→**59.76**. Test ep14→18: 50.82→**51.69**. **P29(63.20/54.34)에 바짝 근접**(Day-Val -3.4/Test -2.7), P30 완전 상회. ep22/200 → P29 추월 유력. |
| 2026-07-03 12:13 | 28/200 | 59.53 (best **59.76@ep22**) | 51.24 (best **51.69@ep18**) | val ep22 / test ep18 | G5,6,7 100%@~150GB, 42 proc, .log 12:10 신선 | **✅ 정상(소폭 정체)**. ep24~28 신기록 없이 Day-Val ~57~59.5, Test ~48~51. best 59.76/51.69 유지. P29(63.20/54.34)에 -3.4/-2.7. ep28/200, 재상승 여지. |
| 2026-07-03 14:13 | 34/200 | **60.71** (best 60.71@ep32 신기록) | 52.29 (best **52.29@ep32** 신기록) | val ep32 / test ep32 | G5,6,7 100%@~150GB, 42 proc, .log 14:08 신선 | **✅ 계속 상승**. Day-Val ep22→32: 59.76→**60.71**. Test ep28→32: 51.95→**52.29** 신기록. P29(63.20/54.34)에 -2.5/-2.0 근접. ep34/200 상승 지속. |
| 2026-07-03 16:13 | 40/200 | 59.30 (best **60.71@ep32**) | 52.23 (best **52.29@ep32**) | val ep32 / test ep32 | G5,6,7 100%@~150GB, 42 proc, .log 16:08 신선 | **✅ 정상(소폭 정체)**. ep34~40 신기록 없이 Day-Val ~57~59.3, Test ~50~52.2. best 60.71/52.29@ep32 유지. P29(63.20/54.34)에 -2.5/-2.0. ep40/200. |
| 2026-07-03 18:13 | 46/200 | 58.06 (best **60.71@ep32**) | 51.87 (best **52.29@ep32**) | val ep32 / test ep32 | G5,6,7 100%@~150GB, 42 proc, .log 18:06 신선 | **⚠️정체(ep32 peak)**. ep34~46 Day-Val ~57~59(peak 60.71 미회복), Test ~51~52. best 60.71/52.29@ep32 고착. P29(63.20/54.34) 추월 불투명(peak-2.5/-2.0). ep46/200. |
| 2026-07-03 20:13 | 52/200 | 57.73 (best **60.71@ep32**) | 51.24 (best **52.57@ep46** 신기록) | val ep32 / test ep46 | G5,6,7 100%@~150GB, 42 proc, .log 20:05 신선 | **정상**. Test ep46 **52.57 신기록**(52.29→52.57). Day-Val ep32 peak 60.71 유지(56~58 진동, 미회복). ep52/200. P29(63.20/54.34)와 gap 유지. |
| 2026-07-03 22:13 | 58/200 | 57.94 (best **60.71@ep32**) | 53.79 (best **53.88@ep56** 신기록) | val ep32 / test ep56 | G5,6,7 0~8%@~150GB(epoch경계), 42 proc, .log 22:11 신선 | **✅ Test 상승**. Test ep54→56: 53.55→**53.88** 연속신기록, **P29 Test 54.34에 -0.46 근접!** Day-Val 60.71@ep32 peak 유지(~58 진동). ep58/200. |
| 2026-07-04 00:13 | 64/200 | 59.75 (best **60.71@ep32**) | 52.49 (best **53.88@ep56**) | val ep32 / test ep56 | G4-7 93~98%@~150GB, 38 proc, .log 00:09 신선 | **정체**. ep58~64 Day-Val ~59-60(peak 60.71 미회복), Test ~52-53.8. best 60.71/53.88 유지. 공식목표 val66.51/test56.71 미달, P29(63.20/54.34)도 미달. ep64/200. |
| 2026-07-04 06:37 | 84/200 | **60.87** (best 60.87@ep80 신기록) | 53.98 (best **54.06@ep78**) | val ep80 / test ep78 | G5,6,7 100%@~150GB, 42 proc, .log 06:34 신선 | **✅ 재상승**. Day-Val ep32 peak 넘어 **60.87@ep80** 신고점. Test 54.06@ep78(P29 54.34에 -0.28). P29 추격 재개. ep84/200. |
| 2026-07-04 08:37 | 90/200 | **61.48** (best 61.48@ep86 신기록) | 51.58 (best **54.06@ep78**) | val ep86 / test ep78 | G5,6,7 100%@~150GB, 42 proc, .log 08:31 신선 | **✅ 상승 지속**. Day-Val ep80→86: 60.87→**61.48**(신고점, P29 63.20에 -1.7). Test 54.06@ep78(P29 -0.28). ep90/200. |
| 2026-07-04 12:37 | 102/200 | **62.87** (best 62.87@ep96 신기록) | **54.67** (best **54.67@ep96** 신기록) | val ep96 / test ep96 | G4-7 89~96%@~150GB, 38 proc, .log 12:36 신선 | **✅🏆 P29 Test 돌파**. Day-Val 61.48→**62.87@ep96**(P29 63.20에 -0.33). **Test 54.67@ep96 > P29 54.34(+0.33)!** P31 seg=현 best/tied-best seg. ep102/200 더 여지. |
| 2026-07-04 14:37 | 108/200 | **63.20** (best 63.20@ep106 신기록) | 54.29 (best **54.67@ep96**) | val ep106 / test ep96 | G4-7 76~95%@~150GB, 38 proc, .log 14:33 신선 | **✅🏆 P29 동률 달성**. Day-Val **63.20@ep106 = P29 63.20**! Test 54.67@ep96 > P29 54.34. **P31=최선 DELIVER seg**(Day-Val 동률·Test 우세). ep108/200. 공식목표 66.51/56.71엔 아직 갭. |
| 2026-07-04 16:37 | 114/200 | 58.79 (best **63.20@ep106**) | 52.84 (best **54.67@ep96**) | val ep106 / test ep96 | G4-7 8%@~150GB(epoch경계), 38 proc, .log 16:30 신선 | **정상**. best 63.20@ep106(=P29)/54.67@ep96(>P29) 유지, ep114 일시 dip(58.79). 최선 seg 지위 유지. ep114/200. |
| 2026-07-04 18:37 | 120/200 | 62.22 (best **63.20@ep106**) | 54.08 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 18:26 신선 | **정상(정체)**. best 63.20/54.67 유지(61-62/54 진동). 최선 seg 지위 유지. ep120/200. |
| 2026-07-04 20:37 | 126/200 | 62.69 (best **63.20@ep106**) | 53.38 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 20:24 신선 | **정상(정체)**. best 63.20/54.67 유지(62~62.7/53~54 진동). ep126/200. 최선 seg 유지. |
| 2026-07-04 22:37 | 132/200 | 61.00 (best **63.20@ep106**) | 54.47 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 22:22 신선 | **정상(정체)**. best 63.20/54.67 유지(61~62/53~54.5 진동). ep132/200. 최선 seg 유지. |
| 2026-07-05 00:37 | 138/200 | 61.29 (best **63.20@ep106**) | 54.14 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 00:19 신선 | **정상(정체)**. best 63.20/54.67 유지(~61/~54 진동). ep138/200. 최선 seg 유지. |
| 2026-07-05 02:37 | 144/200 | 61.85 (best **63.20@ep106**) | 54.09 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 02:18 신선 | **정상(정체)**. best 63.20/54.67 유지(~61.5/~54 진동). ep144/200. 최선 seg 유지. |
| 2026-07-05 04:37 | 150/200 | 61.18 (best **63.20@ep106**) | 53.94 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 04:17 신선 | **정상(정체)**. best 63.20/54.67 유지(~61/~54 진동). ep150/200. 최선 seg 유지. |
| 2026-07-05 06:37 | 156/200 | 61.55 (best **63.20@ep106**) | 53.82 (best **54.67@ep96**) | val ep106 / test ep96 | 42 proc, .log 06:17 신선 | **정상(정체)**. best 63.20/54.67 유지(~61/~54). ep156/200. 최선 seg 유지. |
| 2026-07-05 08:37 | 162/200 | 61.21 (best **63.20@ep106**) | 54.20 (best **54.75@ep158** 신기록) | val ep106 / test ep158 | 42 proc, .log 08:17 신선 | **✅ Test 소폭↑**. Test **54.75@ep158**(54.67→54.75, P29 54.34에 +0.41). Day-Val 63.20@ep106 유지. 최선 seg. ep162/200. |
| 2026-07-05 10:37 | 168/200 | 61.76 (best **63.20@ep106**) | 54.00 (best **54.75@ep158**) | val ep106 / test ep158 | 42 proc, .log 10:18 신선 | **정상(정체)**. best 63.20/54.75 유지(~61.8/~54). ep168/200. 최선 seg 유지. |
| 2026-07-05 12:37 | 174/200 | 62.08 (best **63.20@ep106**) | 54.22 (best **54.75@ep158**) | val ep106 / test ep158 | 42 proc, .log 12:18 신선 | **정상(정체)**. best 63.20/54.75 유지(~62/~54). ep174/200. 최선 seg 유지. |
| 2026-07-05 16:37 | 186/200 | 62.07 (best **63.20@ep106**) | 54.49 (best **54.85@ep182** 신기록) | val ep106 / test ep182 | .log 16:18 신선 | **✅ Test 소폭↑**. Test **54.85@ep182**(54.75→54.85, P29 54.34에 +0.51). Day-Val 63.20@ep106. 최선 seg. ep186/200. |
| 2026-07-05 18:37 | 192/200 | 62.50 (best **63.20@ep106**) | 54.44 (best **54.85@ep182**) | val ep106 / test ep182 | .log 18:18 신선 | **정상(종료근접)**. best 63.20/54.85 유지(~62.5/~54.5). ep192/200(~8ep 남음). 최선 seg. |
| 2026-07-05 20:37 | 198/200 (곧 완주) | 62.44 (best **63.20@ep106**) | 54.37 (best **54.85@ep182**) | val ep106 / test ep182 | .log 20:18 | **정상·종료임박**. best 63.20/54.85 유지. ep198/200(~20분 뒤 완주). 최선 seg 확정 예정. |
| 2026-07-05 22:37 | 🏁 **완주**(200ep) | 최종 best **Day-Val 63.20@ep106 / Test 54.85@ep182** | best_ckpt 보존 | proc 0→P32로 대체 | **🏁 완주**. P31 seg 200ep 완료. **P29(63.20/54.34) 동률(Day-Val)+Test 우세(54.85)** → 최선 DELIVER seg. 공식목표 66.51/56.71엔 미달. → P32로 승계. |

---

## RUN-10 · Jarvis **det_P31_v3clip** (P31.1 det: calibrated reliability + decisive router + FCOS primary) — P30-Det 교훈 반영

- **서버/owner**: jarvis. **2026-07-03 ~11:03 시작**, nproc=4, GPU 1-4(~24GB, batch4 fp32), `--cfg configs/det/det_P31_v3clip_jarvis.yaml`.
- **개선점(P30-Det 대비)**: SEG_MODEL=**LoRA_Sam_P31_Det**(RBMA + **calibrated reliability**(포화 해소) + **decisive router**(비적응 fusion 수정) — 메모리 feature-probe 발견 대응). DET_MODEL=MemorySAMDetectorP30, **FCOS primary**(object-query decoder small-obj 약점 회피). **AMP off**(P30-Det NaN 교훈), grad-ckpt ON, batch4, **v3clip split**(det_{train,test}_v3clip.json), letterbox+aug+ATSS 계승. EPOCHS=40.
- **출력**: `outputs/det/det_P31_v3clip_jarvis/`, 로그 `logs/det_P31_v3clip_*.log`. 비교선: P29-Det ep9 mAP50 0.446(검출 best-overall), P30-Det 0.256.
- **리포트**: mAP/mAP50(목표)/mAP75.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-03 12:13 | Epoch[2] 48% | (아직 없음 — 첫 eval=ep4) | 0개 | G1-4 9~92%@~24GB, 21 proc, nan-skip 0, finite loss 1818 | **정상 초기 학습**. AMP off로 NaN 없음(P30-Det 교훈 적용). loss 1.07. 첫 mAP50=ep4(~40분뒤) → P30-Det(0.256)·P29-Det(0.446) 대비 개선 여부 첫 판정. |
| 2026-07-03 14:12 | Epoch[6] 32% (ep4 eval done) | **ep4 mAP 0.2531 / mAP50 0.4619 / mAP75 0.2560** 🏆 | epoch4·best(13:35) | G1-4 100%@24GB, 21 proc, nan-skip 0 | **✅🏆 검출 best-overall 경신**. ep4만에 mAP50 **0.4619** — 기존 best P29-Det ep9(0.446)·bundle peak(0.4455) 돌파, P30-Det(0.256) 압도. AMP off로 NaN 없음. ep4/40 상방 큼 → 목표 0.85 가장 유망. |
| 2026-07-03 14:13 | Epoch[6] 36% | 최신 **ep4 mAP50 0.4619**(ep9 eval ~1h뒤); best=ep4 | epoch4·best(13:35) | G1-4 100%@24GB, 21 proc, nan-skip 0 | **✅ 정상**. fp32라 ~28분/ep(느림). ep4 mAP50 0.4619(검출 best-overall) 유지, ep9 상승 지속 여부 관건. |
| 2026-07-03 16:13 | Epoch[10] 20% (ep9 eval done) | **ep9 mAP 0.2814 / mAP50 0.4701 / mAP75 0.3101** 🏆(ep4 0.4619→상승) | epoch9·best(16:07)·4 | G1-4 8~100%@24GB, 21 proc, nan-skip 0 | **✅🏆 계속 경신**. mAP50 ep4 0.4619→**ep9 0.4701**, mAP75 0.256→0.310 큰 개선(localization↑). 검출 best-overall 유지·상승. ep10/40 상방 큼. |
| 2026-07-03 18:13 | Epoch[14] 47% | 최신 **ep9 mAP50 0.4701**(ep14 eval ~15분뒤); best=ep9 | epoch9·best·4 | G1-4 79~100%@24GB, 21 proc, nan-skip 0 | **✅ 정상**. fp32 ~28분/ep. ep9 mAP50 0.4701(검출 best-overall) 유지, ep14 상승 지속 여부 대기. |
| 2026-07-03 20:13 | Epoch[18] 34% (ep14 eval done) | **ep14 mAP 0.287 / mAP50 0.4724 / mAP75 0.310** 🏆(ep9 0.4701→상승) | epoch14·best(18:38)·9·4 | G1-4 95~100%@24GB, 21 proc, nan-skip 0 | **✅🏆 계속 경신**. mAP50 ep4 0.4619→9 0.4701→**14 0.4724** 완만 상승. 검출 best-overall 유지. ep18/40 상방 남음. |
| 2026-07-03 22:13 | Epoch[22] 21% (ep19 eval done) | **ep19 mAP 0.294 / mAP50 0.4674 / mAP75 0.3217** (ep14 0.4724→소폭↓) | epoch19·best(=ep19 by COCO mAP)·14 | G1-4 98%@24GB, 21 proc, nan-skip 0 | **✅ 정상(mAP50 평탄)**. mAP50 ep4→19: 0.4619→9 0.4701→14 **0.4724(peak)**→19 0.4674. mAP75·mAP은 상승(localization↑). **mAP50-best=ep14 0.4724**, 여전히 검출 best-overall. ep22/40. |
| 2026-07-04 00:13 | Epoch[26] 9% (ep24 eval done) | **ep24 mAP 0.295 / mAP50 0.4639 / mAP75 0.321** (ep14 peak 0.4724→하락) | epoch24·best(=ep24 COCO mAP)·19·14 | G1-4 100%@24GB, 21 proc, nan-skip 0 | **✅ 정상(mAP50 하락)**. mAP50 ep14 0.4724(peak)→19 0.4674→24 0.4639. COCO mAP 평탄(0.295). **mAP50-best=ep14 0.4724**(검출 best-overall). ep26/40, ep14 재돌파 불확실. |
| 2026-07-04 06:37 | Epoch[38] 99% (ep34 eval done) | ep34 mAP 0.2912 / **mAP50 0.4636** / mAP75 0.312 (mAP50-best=ep14 0.4724) | epoch34·best(=ep29 COCO mAP) | G1-4 @24GB, 21 proc, nan-skip 0 | **정상·종료임박**. ep38/40. mAP50 ~0.464 수렴(v3clip split). best mAP50=ep14 0.4724. 곧 완료. |
| 2026-07-04 08:37 | 🏁 **완료**(40ep, ~07:16) | 최종 ep39 mAP 0.2936 / **mAP50 0.4637** / mAP75 0.315; **best mAP50=ep14 0.4724** | epoch39·best(=ep29 COCO mAP 0.2961) | proc 0, GPU free | **🏁 완주**. 'Training complete. Best AP 0.2961'. mAP50 ~0.464 수렴(v3clip split=비공식). NaN 0. 검출 공식 best는 여전히 bengio egofill 0.85. |

---

## RUN-11 · bengio **det_P29_egofill** (P29-Det + egofill lidar 데이터) — 🎯 mAP50 0.85 도달

- **서버/owner**: bengio (`jemo_maeng`, 8-GPU). **별도 체크아웃** `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM-p29det-egofill`. GPU **1,3,5,6,7**(nproc=5), AMP on. 로그 `train_p29_egofill.log`(tee), 출력 `outputs/det_egofill/`.
- **config**: `configs/det/det_P29_egofill_bengio.yaml`. SEG_MODEL=LoRA_Sam_P29_Det, **EPOCHS=50**, BATCH_SIZE=1. ROOT=poongsan_v2. **핵심=데이터**: lidar **egofill**(ego-motion 보정 최근접 스캔 재투영)으로 조밀화 + train 5,862→**11,799장(2.01×)**. VAL=`det_test_v2_orig1772.json`(=P29/P30-Det과 동일 1772 test, 직접비교 가능).
- **의미**: P30 feature-probe의 "lidar degenerate/sparse" 병목을 데이터로 해결 → 검출 대약진.
- **리포트**: mAP/mAP50(목표)/mAP75.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | 상태 판정 |
|---|---|---|---|---|
| 2026-07-04 01:08 | Epoch[11] 61% | **~ep9 mAP 0.5128 / mAP50 0.8501 / mAP75 0.5514** 🎯(~ep4 0.8279) | best(=ep9) | **🎯 목표 mAP50 0.85 도달!** 기존 검출 best(det_P31_v3clip 0.4724·P29-Det 0.446) 압도. egofill lidar+2× 데이터 효과. EPOCHS=50 아직 상방. AMP on nan-skip 0. **검출 best-overall = det_P29_egofill으로 갱신.** |
| 2026-07-04 06:37 | Epoch[17] 62% (ep14 eval done) | **ep14 mAP 0.5081 / mAP50 0.8486 / mAP75 0.5395** 🎯 (ep9 0.8501→유지) | epoch14·best(=ep9)·9·4 | G1,3,5,6,7 활성, 27 proc, nan-skip 0 | **🎯 목표 유지**. mAP50 ep4 0.8279→9 **0.8501**→14 0.8486, ~0.85 안정. best(COCO mAP)=ep9. EPOCHS=50, ep17/50 상방 남음. 검출 best-overall. |
| 2026-07-04 08:37 | Epoch[19] 95% | ep14 mAP 0.5081 / **mAP50 0.8486** / mAP75 0.540 🎯 (ep19 eval 임박); best=ep9 | epoch14·best(=ep9) | G1,3,5,6,7 활성, 27 proc, nan-skip 0 | **🎯 목표 유지**. mAP50 ~0.85 안정(ep9 0.8501/ep14 0.8486). ep19/50 상방 남음. 검출 best-overall. |
| 2026-07-04 12:37 | Epoch[24] 21% (ep19 eval done) | ep19 mAP 0.5080 / **mAP50 0.8360** / mAP75 0.545 (ep14 0.8486→소폭↓) | epoch19·best(=ep9 mAP50 0.8501) | G1,3,5,6,7 활성, 27 proc, nan-skip 0 | **🎯 목표권 유지**. mAP50 ep9 0.8501→14 0.8486→19 0.8360(~0.84 진동). best mAP50=ep9 0.8501. ep24/50. 검출 best-overall. |
| 2026-07-04 14:37 | Epoch[26] 15% (ep24 eval done) | ep24 mAP **0.5142** / mAP50 0.8364 / mAP75 0.545 (COCO mAP 신기록) | epoch24·best(=ep24 COCO mAP) | G1,3,5,6,7 활성, 27 proc, nan-skip 0 | **🎯 목표권 유지**. mAP50 ~0.836(ep19 0.8360/ep24 0.8364), peak mAP50=ep9 0.8501. COCO mAP는 ep24 0.5142로 상승(best_ckpt=ep24). ep26/50. 검출 best-overall. |
| 2026-07-04 16:37 | Epoch[28] 39% (P29 egofill) | 최신 ep24 mAP 0.5142 / mAP50 0.8364; **best mAP50=ep9 0.8501** | epoch24·best | G1,3,5,6,7(nproc5), 27 proc, nan-skip 0 | **🎯 정상 진행**. mAP50 ~0.84 유지. ep28/50. 검출 best-overall. ⚠️ 같은 서버에 P31 egofill(RUN-12) 병렬 시작됨. |
| 2026-07-04 18:37 | Epoch[30] 24% (ep29 eval) | ep29 mAP 0.5102 / mAP50 0.8315 / mAP75 0.544 (best mAP50=ep9 0.8501) | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지**. mAP50 ~0.83(ep24 0.8364→ep29 0.8315). best COCO mAP=ep24 0.5142. ep30/50. |
| 2026-07-04 20:37 | Epoch[32] 44% | 최신 ep29 mAP50 0.8315(ep34 eval 대기); best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지**. mAP50 ~0.83권. best COCO mAP=ep24 0.5142. ep32/50. |
| 2026-07-04 22:37 | Epoch[34] 66% | 최신 ep29 mAP50 0.8315(ep34 eval 대기); best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지**. mAP50 ~0.83권. ep34/50. |
| 2026-07-05 00:37 | Epoch[36] 44% (ep34 eval) | ep34 mAP 0.5015 / mAP50 0.8202 / mAP75 0.535 (best mAP50=ep9 0.8501) | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지(완만↓)**. mAP50 ep9 0.8501→29 0.8315→34 0.8202. peak는 ep9. ep36/50. |
| 2026-07-05 02:37 | Epoch[38] 70% | 최신 ep34 mAP50 0.8202(ep39 eval 대기); best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지**. ep38/50, 곧 종료. best mAP50 ep9 0.8501 유력. |
| 2026-07-05 04:37 | Epoch[40] 55% (ep39 eval) | ep39 mAP 0.4982 / mAP50 0.8170 / mAP75 0.531 (best mAP50=ep9 0.8501) | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지(완만↓)**. mAP50 0.85(ep9)→0.82(ep39). ep40/50, 곧 종료. best=ep9. |
| 2026-07-05 06:37 | Epoch[42] 79% | 최신 ep39 mAP50 0.8170(ep44 eval 대기); best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지**. mAP50 ~0.82권. ep42/50, 곧 종료. best=ep9 0.8501. |
| 2026-07-05 08:37 | Epoch[44] 100%(ep45 진입) | 최신 ep39 mAP50 0.8170; best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, proc 11(ep경계), nan-skip 0 | **🎯 유지**. mAP50 ~0.82권. ep44/50, ~5ep 남음. best=ep9 0.8501. |
| 2026-07-05 10:37 | Epoch[46] 97% (ep44 eval) | ep44 mAP 0.4941 / mAP50 0.8135 / mAP75 0.524 (best mAP50=ep9 0.8501) | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 유지(완만↓)**. mAP50 0.85(ep9)→0.81(ep44). ep46/50, ~4ep 남음. best=ep9. |
| 2026-07-05 12:37 | **Epoch[49]=마지막** 28% | 최신 ep44 mAP50 0.8135; best mAP50=ep9 0.8501 | epoch24·best | G1,3,5,6,7, 27 proc, nan-skip 0 | **🎯 완주 임박**. ep49/50 마지막 epoch(~35분 뒤 완료). 최종 mAP50 ~0.81 예상이나 best=ep9 0.8501(목표달성분). |
| 2026-07-05 16:37 | 🏁 **완주**(50ep, ~13:35) | 최종 best **mAP50 0.8501@ep9** / best COCO mAP 0.5142@ep24; 최종ep49 mAP50 0.8119 | best·epoch49 | proc 0, GPU 반환 | **🏁🎯 완주·목표달성**. 'Training complete. Best AP 0.5142'. mAP50 ep9 0.8501(목표 0.85 달성)이 최선, 후반 완만 하락(→0.81). egofill 데이터가 검출 결정타. |

> 🎯 **2026-07-04 01:08 — det 목표(mAP50 0.85) 사실상 달성**: bengio P29-Det **egofill** 데이터 변형이 **mAP50 0.8501@~ep9**(동일 v2 test 1772). 아키텍처(P29/P30/P31) 개선보다 **lidar egofill 데이터 조밀화 + 2× 증량**이 결정적. 이전 최고 det_P31_v3clip 0.4724 대비 +0.38. → 검출 방향은 이 egofill 데이터가 정답. EPOCHS=50, ~ep11 진행 중(최종 더 오를 수 있음).

---

## RUN-12 · bengio **det_P31_egofill** (P31.1 backbone × egofill 데이터) — 최선 backbone × 최선 데이터 조합

- **서버/owner**: bengio, 체크아웃 `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM`(P29 egofill과 다른 체크아웃). **2026-07-04 14:59 시작**(nproc=1 단독, 느림) → **2026-07-05 20:19 nproc=6 재기동**(GPU 0,1,3,5,6,7, master_port 29532, fresh ep0; ~69분/epoch). 로그 `logs/det_P31_egofill_6gpu_*.log`. 로그 `logs/det_P31_egofill_20260704_145911.log`, 출력 `outputs/det/det_P31_egofill_bengio/`.
- **config**: `configs/det/det_P31_egofill_bengio.yaml`. SEG_MODEL=**LoRA_Sam_P31_Det**(P31.1: calibrated reliability+decisive router), DET_MODEL=MemorySAMDetectorP30(FCOS primary), **egofill 데이터**(det_train_v2_egofill.json, 2×), AMP off, grad-ckpt, batch4, EPOCHS=40.
- **동기**: egofill 데이터(P29에서 mAP50 0.85)와 P31.1 개선 backbone을 결합 → 검출 추가 향상 기대. 비교선: P29 egofill 0.85(RUN-11), det_P31_v3clip 0.4724(v3clip).

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | GPU/proc | 상태 판정 |
|---|---|---|---|---|
| 2026-07-04 16:37 | **Epoch[0] 29%** | (아직 없음 — 첫 eval=ep4) | G4 단독(nproc=1), 6 proc | **⚠️ 극도로 느림**. **nproc=1(단일 GPU)**로 **7.96s/it → ~6.5h/epoch, 40ep=~11일**. AMP off·fp32·grad-ckpt·2×egofill data(2949 iter/ep). 첫 mAP50(ep4)이 ~26h 뒤 → 실용성 낮음. |
| 2026-07-04 18:37 | **Epoch[0] 54%** (eval 아직) | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1), 6 proc | **⚠️ 여전히 단일GPU 느림**. 8.18s/it, ep0 54%(~6.7h/ep). 사용자 GPU 재배정 결정 대기(jarvis 아직 huisu_kim 점유, bengio 만석). 첫 mAP50까지 하루+. |
| 2026-07-04 20:37 | **Epoch[0] 84%**(5.6h째) | (없음) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.4s/it. ep0조차 미완(~1h 더). GPU 재배정 미결정 → 계속 느림. 첫 eval(ep4)까지 하루+. |
| 2026-07-04 22:37 | **Epoch[1] 14%** | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 7.9s/it, ep0완료→ep1. eval(ep4) 아직 멀음. GPU 재배정 미결정. |
| 2026-07-05 00:37 | Epoch[1] 44% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.1s/it. ep1 진행, eval(ep4) 멀음. GPU 재배정 미결정. |
| 2026-07-05 02:37 | Epoch[1] 75% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.0s/it. ep1 마무리 중, eval(ep4) 멀음. GPU 재배정 미결정. |
| 2026-07-05 04:37 | Epoch[2] 5% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.0s/it. ep2 진입, eval(ep4)까지 ~13h. GPU 재배정 미결정. |
| 2026-07-05 06:37 | Epoch[2] 35% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.0s/it. ep2 진행. eval(ep4)까지 ~10h. GPU 재배정 미결정. |
| 2026-07-05 08:37 | Epoch[2] 66% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 7.9s/it. ep2 진행. eval(ep4)까지 ~8h. GPU 재배정 미결정. |
| 2026-07-05 10:37 | Epoch[2] 96% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.0s/it. ep3 진입 임박, eval(ep4)까지 ~6h. GPU 재배정 미결정. |
| 2026-07-05 12:37 | Epoch[3] 27% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 8.1s/it. ep3 진행. eval(ep4)까지 ~6.5h. GPU 재배정 미결정. |
| 2026-07-05 16:37 | Epoch[3] 88% | (없음 — 첫 eval ep4) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 7.8s/it. ep4 진입 임박 → 첫 eval ~7h 뒤. GPU 재배정 미결정. (P29 egofill 완주로 bengio GPU 5장 방금 반환됨 → nproc↑ 재기동 가능해짐) |
| 2026-07-05 18:37 | Epoch[4] 19% | (없음 — ep4 끝에서 첫 eval) | 0개 | G4 단독(nproc=1) | **⚠️ 단일GPU 지속**. 7.8s/it. ep4 진입, 첫 eval ~5h 뒤. **bengio 6장(0,1,3,5,6,7) idle인데 재기동 미승인** → nproc=6로 재기동 시 5배↑ 가능. |
| 2026-07-05 20:19 | 🚀 **nproc=6 재기동**(fresh ep0) | (없음 — 첫 eval ep4 ~4.6h뒤) | 0개 | G0,1,3,5,6,7 98~100%@~22GB, 31 proc, nan-skip 0 | **✅ 다중GPU 재기동 성공**. 단일GPU run 중단 후 nproc=6(GPU 0,1,3,5,6,7)로 재시작(사용자 승인). 491 iter/ep, 8.4s/it → **~69분/epoch**(단일 6.5h 대비 5.6×). 40ep=~46h, 첫 eval ep4 ~01시. finite loss, OOM 없음. 로그 det_P31_egofill_6gpu_20260705_201747.log. |
| 2026-07-05 20:37 | Epoch[0] 27% (6-GPU) | (없음 — 첫 eval ep4 ~4h뒤) | 0개 | G0,1,3,5,6,7 98~100%@~22GB, 31 proc, nan-skip 0 | **✅ 6-GPU 정상**. loss 2.38→1.25 하강. 8.6s/it, 491iter/ep(~71분/ep). 재기동 안정. 첫 mAP50 ~01시. |
| 2026-07-05 22:37 | 🔴 **크래시**(SIGKILL -9, ep0 56%, ~20:57) | (없음) | 0개 | 외부 kill, GPU 타작업 재점유 | **🔴 6-GPU run 사망(원인규명)**. exitcode -9(SIGKILL). **사용자가 20:55 새 `det_P29_event`(nproc=5, GPU 0,1,3,4,6) 실험을 띄우며 GPU가 겹쳐(0,1,3,6) 메모리 초과→내 P31 egofill이 OOM-kill**됨(내 잘못/config문제 아님). 사용자가 event ablation 우선. 빈 GPU 5,7뿐 → 6-GPU 재기동 불가. best 검출=P29 egofill 0.8501. → RUN-14로 GPU 이관됨. |

> ⚠️ **2026-07-04 16:37 — P31 egofill 단일 GPU 문제**: RUN-12(det_P31_egofill)가 **nproc=1**로 돌아 ~6.5h/epoch(40ep≈11일)로 극도로 비효율. jarvis(4 GPU idle, 단 egofill 데이터 없음) 또는 bengio 다중 GPU 배정으로 **nproc↑ 재기동 권장**. 아니면 첫 결과(ep4)까지 하루 이상 소요.

---

## RUN-13 · B200 **P32** RBMA seg (DELIVER) — P31 완주 후 승계

- **서버**: B200. **2026-07-05 ~22:02 시작**(P31 완주 후 자동 승계), torchrun nproc=4, master_port 29538.
- **config**: `configs/b200-deliver_rgbdel_P32_physaug.yaml`. EPOCHS=200. P31 대비 diff 미확인.
- **출력**: `outputs/MMSamP32/.../train.log`. 비교선: P31 seg best Day-Val 63.20@ep106/Test 54.85@ep182(현 최선), P29 63.20/54.34. 공식목표 val 66.51/test 56.71.

| 점검 시각(KST) | epoch | Day-Val | Test | best | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-05 22:37 | 2/200 (초기) | 36.37 | (미로깅) | val ep2 | **정상 초기 학습**. fresh ramp-up(Day-Val 36.37@ep2). P31(63.20/54.85) 넘는지 관찰. 첫 Test는 다음 점검. |
| 2026-07-06 00:37 | 6/200 | **51.11** (best 51.11@ep6) | **43.47** (best 43.47@ep6) | val ep6 / test ep6 | 41 proc, .log 00:18 신선 | **정상 초기상승**. Day-Val ep2→6: 36.37→**51.11**, Test **43.47@ep6**(신기록). P31 ep6(48.96/45.18) 대비 Day-Val↑/Test↓ 혼조(초기). ep6/200. |
| 2026-07-06 02:37 | 12/200 | **56.60** (best 56.60@ep12) | **48.89** (best 48.89@ep12 신기록) | val ep12 / test ep12 | 37 proc, .log 02:34 신선 | **정상 상승이나 P31 열세**. Day-Val ep10→12: 52.14→56.60, Test 46.93→**48.89**. **P31 ep12(57.70/50.32) 대비 -1.1/-1.4 뒤짐** 지속. baseline 56.71/66.51엔 한참. ep12/200. |
| 2026-07-06 04:37 | 16/200 | 53.17 (best **56.60@ep12**) | 46.54 (best **49.56@ep14** 신기록) | val ep12 / test ep14 | 41 proc, .log 04:04 신선 | **⚠️ ep12-16 정체·하락**. Day-Val ep12→14→16: 56.60→55.49→**53.17**(하락), Test ep14 **49.56**(peak, 신기록)→ep16 46.54. P32가 P31 궤도 못 따라가고 조기 정체 조짐. baseline 66.51/56.71엔 여전히 큰 격차. ep16/200. |
| 2026-07-06 06:37 | 22/200 | 58.04 (best **59.45@ep20**) | 51.16 (best **51.16@ep22** 신기록) | val ep20 / test ep22 | 41 proc, .log 06:19 신선 | **🎯 반등(정체 아님)**. Day-Val ep16→20→22: 53.17→**59.45**(신기록)→58.04, Test ep16 46.54→ep20 50.61→**ep22 51.16**(연속 신기록). 직전 '조기정체' 우려는 epoch 노이즈였음. Test 51.16이 P31 ep12(50.32) 상회, 궤도 추격 중. baseline 66.51/56.71엔 아직 격차. ep22/200. |
| 2026-07-06 08:37 | 28/200 | 57.52 (best **60.23@ep26**) | 51.65 (best **51.86@ep26** 신기록) | val ep26 / test ep26 | 37 proc, .log 08:35 신선 | **✅ 상승 지속**. Day-Val ep22→26: 58.04→**60.23**(신기록), Test 51.16→**51.86@ep26**(신기록)→ep28 51.65. 짝수ep 고점/홀수 저점 진동 유지되며 best 꾸준↑. P31 best(63.20/54.85)엔 val -3/test -3, 아직 추격 중. ep28/200. |
| 2026-07-06 10:37 | 32/200 | 61.19 (best **61.65@ep30**) | 52.24 (best **52.24@ep32** 신기록) | val ep30 / test ep32 | 41 proc, .log 10:05 신선 | **✅ 신기록 지속**. Day-Val ep26→30: 60.23→**61.65**(신기록), Test ep26 51.86→**ep32 52.24**(신기록). val loss 0.844로 하강 지속. P31 best(63.20/54.85) 대비 val −1.6/test −2.6로 격차 축소 중. baseline 66.51/56.71엔 아직. ep32/200. |
| 2026-07-06 12:37 | 38/200 | 58.40 (best **61.65@ep30**) | 51.28 (best **52.36@ep34** 신기록) | val ep30 / test ep34 | 41 proc, .log 12:22 신선 | **✅ Test 신기록·Val 정체조짐**. Test ep32→34 52.24→**52.36**(신기록), ep36 52.34/ep38 51.28. Day-Val ep30 61.65 이후 미갱신(ep36 61.23, ep38 58.40 진동). val loss 0.829로 하강 지속. P31(63.20/54.85) 대비 val −1.55/test −2.49. ep38/200. |
| 2026-07-06 14:37 | 44/200 | 55.44 (best **61.65@ep30**; vs 66.51 **−4.86**) | 53.45@ep40 신기록 (best **53.45**; vs 56.71 **−3.26**) | val ep30 / test ep40 | 41 proc, .log 14:27 신선 | **✅ Test 신기록 −3.26**. Test ep34 52.36→**ep40 53.45**(신기록, SOTA 56.71에 −3.26 근접). Day-Val ep30 61.65 이후 미갱신·진동폭↑(ep42 54.93/ep44 55.44). val loss 0.80~0.86. SOTA대비 test −3.26 / val −4.86. (진행 상세로그는 완주 후 정리) ep44/200. |
| 2026-07-06 20:37 | 60/200 | **62.52@ep56 신기록** (best 62.52; vs 66.51 **−3.99**) | 52.69 (best **53.45@ep40**; vs 56.71 **−3.26**) | val ep56 / test ep40 | 41 proc, .log 20:30 신선 | **✅ Day-Val 신기록 −3.99**. Day-Val ep30 61.65→**ep56 62.52**(신기록, SOTA 66.51에 −3.99로 개선). Test는 ep40 53.45 유지(ep56 53.33 근접). val loss 0.78 하강 지속. SOTA대비 val −3.99 / test −3.26. ep60/200. |
| 2026-07-07 00:37 | 70/200 | **62.56@ep66 신기록** (best 62.56; vs 66.51 **−3.95**) | **53.52@ep68 신기록** (best 53.52; vs 56.71 **−3.19**) | val ep66 / test ep68 | 37 proc, .log 00:26 신선 | **✅ val·test 동시 신기록(소폭)**. Day-Val 62.52→**62.56@ep66**, Test 53.45→**53.52@ep68**(둘 다 미세 갱신). val loss 0.72로 하강 지속. SOTA대비 val −3.95 / test −3.19(소폭 개선). 여전히 수렴권 미세 상방. ep70/200. |
| 2026-07-07 02:37 | 76/200 | **62.70@ep74 신기록** (best 62.70; vs 66.51 **−3.81**) | **53.62@ep74 신기록** (best 53.62; vs 56.71 **−3.09**) | val ep74 / test ep74 | 41 proc, .log 02:31 신선 | **✅ val·test 동시 신기록(ep74)**. Day-Val 62.56→**62.70**, Test 53.52→**53.62**(둘 다 ep74에서 갱신). 느리지만 꾸준한 상방 드리프트 지속(수렴권이나 미세 개선 누적). SOTA대비 val −3.81 / test −3.09. ep76/200. |
| 2026-07-07 04:37 | 80/200 | 62.08 (best **62.70@ep74**; vs 66.51 **−3.81**) | **53.63@ep80 신기록** (best 53.63; vs 56.71 **−3.08**) | val ep74 / test ep80 | 41 proc, .log 04:20 신선 | **✅ Test 미세 신기록**. Test 53.62→**53.63@ep80**(+0.01, 사실상 수렴 상한). Day-Val 62.70@ep74 유지(ep78/80 62.17/62.08). val loss 0.707까지 하강. SOTA대비 val −3.81 / test −3.08. Test는 ~53.6에서 상한 도달한 모습. ep80/200. |
| 2026-07-07 06:37 | 84/200 | **63.25@ep84 신기록** (best 63.25; vs 66.51 **−3.26**) | 53.55 (best **53.63@ep80**; vs 56.71 **−3.08**) | val ep84 / test ep80 | 41 proc, .log 06:12 신선 | **🎯 Day-Val P31 추월**. Day-Val 62.70→**63.25@ep84**(신기록) → **P31 best 63.20을 val 기준 근소 상회**(+0.05). Test는 53.63@ep80 유지(P31 54.85엔 아직 −1.2). val loss 0.70. SOTA대비 val −3.26 / test −3.08. **P32가 절반(84/200) 만에 P31 val 도달** → 남은 116ep 상방 기대. |
| 2026-07-07 08:37 | 90/200 | 60.52 (best **63.25@ep84**; vs 66.51 **−3.26**) | **54.05@ep88 신기록** (best 54.05; vs 56.71 **−2.66**) | val ep84 / test ep88 | 37 proc, .log 08:27 신선 | **🎯 Test 신기록·P31 근접**. Test 53.63→86 53.87→**88 54.05**(신기록). **P31 best 54.85에 −0.80까지 추격**(val은 이미 63.25>63.20 추월). val loss 0.68. SOTA대비 val −3.26 / test −2.66(개선 지속). P32가 양지표 모두 P31 경신 시야권 → 남은 110ep 관건. ep90/200. |
| 2026-07-07 14:44 | 106/200 | **64.12@ep98 신기록** (best 64.12; vs 66.51 **−2.39**) | **54.74@ep106 신기록** (best 54.74; vs 56.71 **−1.97**) | val ep98 / test ep106 | 41 proc, .log 14:29 신선 | **🎯 P32가 P31 추월(양지표)**. Day-Val **64.12**(>P31 63.20, +0.92), Test **54.74**(P31 54.85에 −0.11, 사실상 동률·계속 상승중). SOTA대비 val −2.39 / test −1.97(둘 다 개선 지속). **P32 = 사실상 최선 DELIVER seg** 등극(val 우세·test 근동률). val loss 0.66. ep106/200, test 아직 상승 → P31 test도 넘길 시야권. |
| 2026-07-07 16:37 | 110/200 | 61.49 (best **64.12@ep98**; vs 66.51 **−2.39**) | **54.79@ep108 신기록** (best 54.79; vs 56.71 **−1.92**) | val ep98 / test ep108 | 41 proc, .log 16:11 신선 | **🎯 Test P31 근접(−0.06)**. Test 54.74→**54.79@ep108**(신기록). **P31 test best 54.85에 −0.06까지**(다음 갱신 시 추월). val은 이미 64.12>63.20 우세. SOTA대비 val −2.39 / test −1.92. val loss 0.65. P32가 양지표 P31 완전추월 임박. ep110/200(아직 90ep). |
| 2026-07-08 02:07 | 134/200 | 62.69 (best **64.12@ep98**; vs 66.51 **−2.39**) | 54.71 (best **54.79@ep108**; vs 56.71 **−1.92**) | val ep98 / test ep108 | 41 proc, .log 01:35 신선 | **수렴(신기록 없음)**. ep110~134 val 61~63/test 53~54.7 진동, best 미갱신(val 64.12@ep98, test 54.79@ep108). **P32 최종상: val 64.12(>P31 63.20) / test 54.79(P31 54.85에 −0.06)** — val 우세·test 근동률로 최선 seg. val loss 0.63. ep134/200(아직 66ep이나 수렴). |

---

## RUN-14 · bengio **det_P29_event** (모달리티 ablation: lidar egofill → event) — 사용자 신규 실험

- **서버/owner**: bengio, 체크아웃 `.../drone-MemorySAM-p29det-egofill`. **2026-07-05 ~20:55 시작**(사용자 jemo_maeng), nproc=5, **GPU 0,1,3,4,6**. 로그 `train_p29_event.log`, 출력 `outputs/det_event/det_P29_event_bengio/`. (이 실행이 내 P31 egofill 6-GPU를 GPU 겹침으로 밀어냄 → RUN-12 사망.)
- **config**: `configs/det/det_P29_event_bengio.yaml`. **모달리티 ablation**: MODALS ['img,lidar,thermal']→['img,**event**,thermal'](event_aligned, 100% 커버리지). 학습=egofill_common11799 split, val=동일 1772. **유일 변인=lidar(egofill) vs event**. P29-Det backbone.
- **의미**: egofill lidar(mAP50 0.85)의 대안으로 event 카메라 모달리티가 검출에 얼마나 기여하는지 clean 비교.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | GPU/proc | 상태 판정 |
|---|---|---|---|---|
| 2026-07-05 22:37 | Epoch[1] 95% | (없음 — 첫 eval ep4) | G0,1,3,4,6(nproc5) 17.5GB, nan-skip 0 | **정상 초기 학습**. ep1, 1.3s/it. 첫 mAP50=ep4 → egofill(0.85) 대비 event 성능 첫 비교점. |
| 2026-07-06 00:37 | Epoch[4] 18% | (없음 — ep4 끝 첫 eval, ~40분뒤) | 0개 | G0,1,3,4,6(nproc5) 17.5GB, 27 proc, nan-skip 0 | **정상 초기학습**. ep4 진입, 1.45s/it. 첫 mAP50 ~01:20 → egofill(0.85) 대비 event 모달리티 첫 비교. 빈GPU 2,5,7. |
| 2026-07-06 02:37 | Epoch[5] 92% (ep4 eval done) | **ep4 mAP 0.4820 / mAP50 0.8250 / mAP75 0.5186** | epoch4·best | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 event≈egofill**. ep4 mAP50 0.8250 ≈ **egofill ep4 0.8279**(거의 동률!). event 카메라 모달리티가 egofill-lidar만큼 검출 기여 → ablation 유의미. ep5/50, ~0.85 궤도 예상. |
| 2026-07-06 04:37 | Epoch[7] 96% | 최신 **ep4 mAP50 0.8250**(ep7 eval 대기); best mAP50=ep4 0.8250 | epoch4·best | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 정상**. ep5-6 eval 미로깅(SAVE_INTERVAL), ep7 진행 중 1.49s/it. event ep4 0.8250 = egofill 수준 유지. ep7/50. |
| 2026-07-06 06:37 | Epoch[9] 93% | 최신 **ep4 mAP50 0.8250**(ep9 eval 임박); best mAP50=ep4 0.8250 | epoch4·best | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 정상**. ep5-8 eval 미로깅(eval~5ep 간격), ep9 93%→ep9 eval 곧. 1.42s/it. event ep4 0.8250 유지. ep9/50. |
| 2026-07-06 08:37 | Epoch[11] 67% | ep9 mAP 0.4781 / **mAP50 0.8113** / mAP75 0.5047 (peak mAP50=ep4 0.8250) | epoch9·best(=ep4) | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 유지(완만↓)**. mAP50 ep4 0.8250→ep9 0.8113(egofill과 동일 패턴: ep초반 peak 후 완만↓). COCO mAP best 여전히 ep4 0.4820. event≈egofill(0.85) 수준 확인. ep11/50. |
| 2026-07-06 10:37 | Epoch[13] 92% | 최신 ep9 mAP50 0.8113(ep14 eval 임박); peak mAP50=ep4 0.8250 | epoch9·best(=ep4) | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 정상**. ep13 92%→ep14 eval 곧. 1.35s/it. event peak mAP50 ep4 0.8250 유지(egofill 0.85 동급). ep13/50. |
| 2026-07-06 12:37 | Epoch[15] 68% | **ep14 mAP 0.5142 / mAP50 0.8427 / mAP75 0.5600** 🎯(신기록) | epoch14·best | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **🎯 event 신기록·egofill 근접**. mAP50 ep4 0.8250→9 0.8113→**14 0.8427**(반등·신기록). COCO mAP 0.5142 = egofill best mAP와 동일. **event≈egofill(0.8501) 거의 동급 확정** — 모달리티 ablation 강한 증거. ep15/50. |
| 2026-07-06 18:37 | Epoch[21] 75% (ep19 eval) | ep19 mAP **0.5174** / mAP50 0.8324 / mAP75 0.5581 (best_ckpt=ep19 by COCO mAP; **목표 mAP50 peak=ep14 0.8427**) | epoch19·best | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **✅ COCO mAP 신기록·mAP50는 미갱신**. COCO mAP 0.5142(ep14)→**0.5174(ep19)** best_ckpt 이동. 그러나 **헤드라인 목표 mAP50은 ep14 0.8427이 여전히 peak**(ep19 0.8324로 소폭↓). event best mAP50 0.8427 = egofill 0.8501에 −0.008 근접 유지. ep21/50. |
| 2026-07-07 14:44 | Epoch[42] 9% (ep39 eval) | ep39 mAP 0.5033 / mAP50 0.8261 / mAP75 0.5379 (peak mAP50=ep14 0.8427) | epoch39·best(=ep19 COCO mAP) | G0,1,3,4,6(nproc5), 27 proc, nan-skip 0 | **수렴(완만↓)**. mAP50 ep34 0.8274→39 0.8261. best mAP50 여전히 ep14 0.8427(egofill 0.8501에 −0.008). ep42/50, ~8ep 남음(곧 완주). |
| 2026-07-08 02:07 | 🏁 **완주**(50ep, ~22:38) | 최종 best **mAP 0.5174@ep19 / mAP50 0.8427@ep14 / mAP75 0.5600**; 최종ep49 mAP50 0.8233 | best·epoch49 | proc 0→det_P29_final_full로 승계 | **🏁 완주**. 'Training complete. Best AP 0.5174'. **event 모달리티: 목표 mAP50 peak 0.8427@ep14 = egofill 0.8501에 −0.008**(거의 동급). 후반 완만↓(→0.823). nan 0. **결론: event≈egofill-lidar**(검출 ablation 유의미). → GPU를 det_P29_final_full로 승계. |

## RUN-15 · bengio **det_P29_final_full** (P29-Det + egofill lidar, **최종 annotation셋**) — 사용자 신규 실험

- **서버/owner**: bengio(egofill 체크아웃 `/SSDb/.../drone-MemorySAM-p29det-egofill`). **2026-07-08 ~01:15 시작**, nproc=5(GPU 0,1,3,4,6), master_port 21834, `--cfg configs/det/det_P29_final_full.yaml`. det_P29_event 완주(RUN-14) 후 GPU 승계.
- **구성**: SEG_MODEL=LoRA_Sam_P29_Det, MODALS=['img','lidar','thermal'](egofill lidar 복귀), **최종 annotation `_final_ann/instances_train_egofill.json`**(기존 _det_splits 아닌 최종본), EPOCHS=50, batch1. 데이터 외 하이퍼파라미터는 det_P29_egofill_bengio와 동일(변수격리=annotation셋만).
- **의도**: 검출 best(egofill mAP50 0.8501@ep9)를 **최종 확정 annotation**으로 재학습해 제출용 final 수치 확보. 로그 `train_m3_full.log`, 출력 `outputs/det_final_full/`.
- **리포트**: mAP/mAP50(목표 0.85)/mAP75.

| 점검 시각(KST) | epoch | mAP / **mAP50** / mAP75 | ckpt | GPU/proc | 상태 판정 |
|---|---|---|---|---|---|
| 2026-07-08 02:07 | Epoch[0] 91% | (없음 — 첫 eval ep4) | 0개 | G0,1,3,4,6(nproc5), 26 proc, nan-skip 0 | **정상 초기 학습**. 2399 iter/ep, 1.4s/it(~56분/ep). loss 0.9~1.3 정상, n_pos 정상. 첫 mAP50=ep4(~4h 뒤) → 최종 annotation 기준 egofill(0.8501) 재현 여부 첫 판정. |
| 2026-07-08 04:20 | Epoch[5] 62% (ep4 eval) | **ep4 mAP 0.4689 / mAP50 0.7515 / mAP75 0.5311** | epoch4·best | G0,1,3,4,6(nproc5), 26 proc, nan-skip 0 | **첫 eval — egofill보다 낮게 출발**. ep4 mAP50 **0.7515** < egofill ep4 0.8279·event ep4 0.8250(동일 epoch 대비 −0.07). **단 최종 annotation셋(_final_ann)이라 val 분포 다름 → 직접 비교 주의**. egofill은 ep9에 peak(0.8501)였으므로 아직 초반. ep5/50, ep9 eval에서 궤도 재판정. |