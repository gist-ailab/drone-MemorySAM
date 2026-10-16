# 서버별 저장 위치 지도 (내 디렉터리·용량·무엇이 어디 있나)

updated: 2026-10-15 (최초 작성. 비-best 웨이트 NAS 이관과 같이 만듦)

> **용도**: "이 서버의 내 디렉터리에 무엇이 있고, 옮긴 것은 어디로 갔나"에 답하는 조회처다.
> 체크포인트 하나하나의 위치(헤드라인 ckpt·이전 이관 묶음)는 [artifact-locations.md](artifact-locations.md)가 정본이고,
> 이 문서는 **서버·볼륨 단위의 배치**만 다룬다. 실험의 의미·수치는 `experiments/`에 있다.

---

## 0. 이번 정리(2026-10-15)의 규칙

사용자 지시: "서버별로 사용 중인 디렉터리의 용량을 확인해서 부족하면 drone_nas로 옮기고 기록하라.
`/ailab_mat2/personal/jemo_maeng` 안의 내 산출물도 포함한다. best가 아닌 쓸데없는 웨이트가 대상이다."

| 항목 | 내용 |
|---|---|
| 대상 볼륨 | 사용률 **90% 이상**인 볼륨 위의 내 디렉터리 + `/ailab_mat2/personal/jemo_maeng` 전체(그 밖은 조사·이동하지 않음) |
| 옮긴 것 | 끝난 런의 **비-best 웨이트**: `*_top2~5*`(학습기 val 순위 2~5위), `test_*`(test-best: 판정·헤드라인 인용 금지 규약 대상), `last_checkpoint.pth`/`last.pth`, 중간 에폭(`epoch_N.pth`·`epochN_checkpoint.pth`·`periodic_epochN_checkpoint.pth`·`iter_N.pth`·detectron2 `model_00NNNNN.pth`) |
| 남긴 것 | `*_top1*`(학습기 val-best), `best*`, `model_final.pth`, 사전학습 가중치·HF/torch 캐시, 데이터셋, 제출물(`submission/`), 공유 분석 사본(`ailab_mat2/.../analysis/`) |
| 건드리지 않은 런 | 실행 중이거나 최근 쓰인 런: 파일 수정일이 2026-10-10 이후인 디렉터리, `A1hires`·`A2hires`(학습 중), `e1confirm`·`ckpt/E1_s[123]_epoch*`(lecun 재채점 중), bengio `dgfusion_train/`(프로세스 살아 있음) |
| 크기 하한 | 200MB 이상 파일만 집계·이관(웨이트가 아닌 작은 파일은 그대로) |
| NAS 목적지 | `/drone_nas/drone/personal/jemo_maeng/nonbest_weights_archive_20261015/<서버>/<원래 절대경로>` (NAS 내부 경로 `/volume1/drone/...`) |
| 검증 | 묶음(약 40GB)마다 ① 원본 md5(서버에서 계산) ② tar 스트림 전송(서버 → 허브 → NAS) ③ NAS에서 `md5sum -c` ④ 원본 크기·mtime 재확인 ⑤ 통과한 묶음만 원본 삭제 |
| 대장 | NAS 같은 폴더의 `MANIFEST_<서버>.tsv`(열: 서버 · 원래 경로 · 바이트 · mtime · md5 · NAS 경로 · 상태 · 처리 시각) + `README.md` |
| 도구 | `scripts/nas_verified_move.sh <태그> <ssh호스트> <목록> <작업폴더> [묶음GB]` (허브에서 tmux로 실행) |

**되돌리기**: 원래 경로가 NAS 하위 경로에 그대로 들어 있으므로 한 파일씩 되살릴 수 있다.

```bash
# 허브에서. <서버>, <절대경로> 는 MANIFEST 의 1·2열
ssh jemo.maeng@172.27.190.182 "cat /volume1/drone/personal/jemo_maeng/nonbest_weights_archive_20261015/<서버><절대경로>" \
  | ssh <서버> "cat > <절대경로>"
ssh <서버> "md5sum <절대경로>"    # MANIFEST 5열과 같아야 한다
# ailab_mat2 는 <서버> 자리에 gyuri(NFS 마운트)를 쓴다
```

---

## 1. 서버·볼륨 현황 (2026-10-14 조사, 이관 전)

사용률은 볼륨 전체(다른 사용자 포함)이고, "내 것"은 `jemo_maeng` 디렉터리 크기다.

| 서버 | 볼륨 | 사용률·여유 | 내 디렉터리 | 내 것 | 주 내용 |
|---|---|---|---|---|---|
| lecun | `/SSDb` | 100% · 26G | `/SSDb/jemo_maeng` | 806G | `src/Project/Drone24/detection/{drone-MemorySAM, drone-mmdetection-jm, UniRGB-IR}`, `dset/`(243G), `ckpt/`, `tmp_ckpt/`, `cafuser_train/` |
| lecun | `/SSDc` | 100% · 63G | `/SSDc/jemo_maeng` | 133G | `dset/`(126G), `hf_cache_p56`, `ckpt_bf`, `muses_seed825` |
| lecun | `/` | 85% · 542G | `/home/jemo_maeng` | 92G | miniconda3(67G), 캐시 |
| bengio | `/SSDb` | 98% · 85G | `/SSDb/jemo_maeng` | 324G | `src/Project`(224G: drone-MemorySAM, dMS-p29det-egofill), `src/p37_train`(57G), `dgfusion_train`, `radar_diag_repo` |
| bengio | `/SSDe` | 98% · 209G | `/SSDe/jemo_maeng` | 미측정(큼) | **일일 카드 체크아웃** `src/drone-MemorySAM-daily`(B0/E1~E4/E1shared/R1/R2/elora 런), `ckpt_e1conf` |
| bengio | `/SSDf` | 69% · 2.2T | `/SSDf/jemo_maeng` | — | 현재 학습(A1hires) 체크아웃, Amphibious rNs, aloha — 대상 아님 |
| levine | `/SSDe` | 92% · 616G | `/SSDe/jemo_maeng` | 322G | `src/Project/Drone/drone-MemorySAM/outputs`(MULTIAQUA·DELIVER P8~P26 MMSam 계보) |
| levine | `/SSDb`·`/SSDd` | 92%·88% | | 21G·26G | 구 체크아웃, DELIVER 데이터 |
| yeon | `/` | 99% · 38G | `/home/jemo_maeng` | 105G | anaconda3(54G), .cache(27G), `wd`(12G) — 웨이트 아님, 이관 대상 없음 |
| yeon | `/SSDc` | 95% · 189G | `/SSDc/jemo_maeng` | 256G | `dset/`만 있음 — 대상 없음 |
| yeon | `/SSDb` | 83% · 622G | `/SSDb/jemo_maeng` | 431G | `src/Project/Drone/detection/drone-MemorySAM-develop`, `dset/`, `dgfusion_train`, `rescore_elora`, `dump_ckpts` |
| yeon | `/SSDe` | 77% · 1.7T | `/SSDe/jemo_maeng` | 465G | `outputs/`(200G: P39·P42·P44·P49 런), `dset/`(218G), 현재 학습(A2hires) 체크아웃 |
| jarvis | `/` | 99% · 50G | `/home/jemo_maeng` | 1.3T | `src/drone-MemorySAM-develop` 등(1.2T) |
| jarvis | `/SSDb` | 90% · 378G | `/SSDb/jemo_maeng` | 765G | `src/`(517G), `dgfusion_train`, `cache`, `rescore_elora`, `qaf` |
| jarvis | `/SSDd` | 81% | `/SSDd/jemo_maeng` | 57G | isaac·touchnplug 데이터 — 대상 아님 |
| hinton | `/SSDb` | 96% · 166G | `/SSDb/jemo_maeng` | 976G | `src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs`(2025년 검출 실험), `dset/`(235G), `ckpt/hoi_smoke`(사전학습 모델) |
| hinton | `/SSDd` | 100% · 3G | `/SSDd/jemo_maeng` | 632G | `habitat-lab/data`(455G, 2023 데이터셋), `dset/` — 웨이트 아님, 대상 없음 |
| hinton | `/` | 51% | `/home/jemo_maeng` | 333G | `probe_cache`(182G), `wd`(124G) — 볼륨 여유 있음, 대상 아님 |
| hpca100 | overlay `/`·`~/SSDb` | 75% · 201G | `~/SSDb/jemo_maeng`, `/tmp/jemo_scratch` | 약 90G · 168G | 9월 이관 후 잔여. `/tmp/jemo_scratch`(휘발성)에 9월 런 162G — 대상 볼륨 아님(§3 참고) |
| gyuri | 전 볼륨 | ≤88% | `/SSDc`·`/SSDe` | 작음 | amphibious 출력 24G — 대상 아님 |
| ailab_mat2 | NAS | 92% · 8.7T | `/ailab_mat2/personal/jemo_maeng` | 웨이트만 1.6T | §2 |

---

## 2. `/ailab_mat2/personal/jemo_maeng` 구성

이 경로 밖은 다른 사용자 공간이라 조사·이동하지 않는다. 여러 서버(lecun·yeon·gyuri NFS, jarvis sshfs)가 함께 보는 공유 볼륨이다.

| 하위 경로 | 무엇 | 이번 처리 |
|---|---|---|
| `drone-MemorySAM_weights/yeon_archive_20260724/` | 7월 yeon 정리 때 옮긴 P37a·P38·P39 웨이트(19G) | top2·test-best 등 비-best만 이관, top1 남김 |
| `src/Project/Drone/drone-memorysam/archive/yeon-old-checkouts/` | yeon 구 체크아웃들(p37b·p38·p49·res 등)의 `outputs/` 통째 사본 — 비-best 웨이트가 대부분 | 비-best 이관(가장 큰 몫) |
| `src/Project/Drone/drone-memorysam/weights/` | B200 런(P34·P35·P36) 출력 사본, MULTIAQUA P9·P14·P16 | 비-best 이관, top1 남김 |
| `src/Project/Drone/drone-memorysam/submission/` | 제출 코드·zip·제출용 ckpt | **남김**(규약상 제출물 위치) |
| `src/Project/Drone/drone-memorysam/analysis/` | 서버 간 공유 분석 산출물 | **남김**(규약상 공유 분석 위치) |
| `src/Project/Amphibious/wd_bengio/runs/` | Amphibious 과제 YOLO 런(시드 71~73, A1~A3·V3·N 계열) | 중간 에폭(`epochN.pt`) 이관(184G), `best.pt`·`last.pt` 남김 |
| `src/Project/Amphibious/{rfdetr-Wt,rNs,amphibious-working}` | Amphibious RF-DETR·rNs 런 | 남김(파일 이름이 이관 규칙에 해당하지 않음) |
| `src/Project/ceiling/` | 2023년 ceiling-segmentation 체크포인트(약 90G) | 남김(파일 이름이 이관 규칙에 해당하지 않음) |
| `src/Project/{Tactile,KIRO,etc}`, `src/{feats,toolforce,compliance,ctrl_bench}` | 다른 과제 코드·결과 | 남김(이관 대상 파일 없음) |
| `dset/` | 데이터셋(argoverse2·nuscenes·Places365·poongsan 등) | **남김** |

---

## 3. 이번에 옮기지 않았지만 알아 둘 것

- **jarvis `/ailab_mat2`(sshfs)는 2026-09-21부터 응답 정지 상태**다. 이번 조사 중 `df -h`가 그 마운트를 조회하다 D 상태로 멈춰 jarvis에 프로세스 하나(PID 1155071, `df -h`)가 남았다. kill 이 듣지 않는다. jarvis에서는 `df`를 경로 없이 부르지 말고 `df -h /home /SSDb` 처럼 경로를 준다.
- **yeon `/`(99%)**는 내 홈의 anaconda3·캐시가 차지한다. 웨이트가 아니라 이번 대상이 아니다. 줄이려면 `~/.cache`(27G) 정리를 따로 결정해야 한다.
- **yeon `/SSDb`·`/SSDe`**(83%·77%)는 대상 기준(90%)에 못 미쳐 그대로 뒀다. 비-best 웨이트가 약 680GB 있다(test-best 241G·top2~5 261G·last 98G·중간 80G).
- **hinton `/SSDd`(100%)**는 2023년 habitat-lab 데이터셋(455G)이 대부분이다. 웨이트가 아니어서 지우지 않았다.
- **hpca100 `/tmp/jemo_scratch`**(162G)는 9월 런(E12·E2s2·E3s2·E13s2·B0s2·E1M)의 전체 top-k다. top1은 `ckpts/daily_cards_20260908/_tmp_volatile/`에 이미 회수돼 있다. overlay 볼륨이 75%라 이번에는 두었다.
- 주간 백업 cron(`~/.local/bin/drone_ckpt_backup.sh`)은 jarvis·yeon·hpca100의 top1·best·last를 `ckpts_backup/`에 계속 복사한다. 이번 이관과 별개다.

---

## 4. drone_nas 내 구획 (`/drone_nas/drone/personal/jemo_maeng/`)

| 경로 | 무엇 |
|---|---|
| `src/Project/drone/drone-MemorySAM/ckpts/` | 정본 웨이트(헤드라인·완주 대표 ckpt, 서버별 이관 묶음). 색인 = [artifact-locations.md](artifact-locations.md) §4 |
| `src/Project/drone/drone-MemorySAM/ckpts_backup/` | 주간 백업 cron 사본(서버별) |
| `src/Project/drone/drone-MemorySAM/analysis_logs/` | 평가·분석·시각화 |
| `src/Project/drone/drone-MemorySAM/videos/` | 발행 영상 |
| `nonbest_weights_archive_20261015/<서버>/...` | **이번 비-best 웨이트 이관**(원래 절대경로 유지). 대장 `MANIFEST_<서버>.tsv` |

<!-- nonbest-archive-20261015-result:begin -->
## 5. 이관 결과 (2026-10-16 11:40 기준)

모든 묶음이 NAS `md5sum -c` 를 통과한 뒤에만 원본을 지웠다. 실패·건너뜀·원본 변경으로 막힌 묶음은 0건이다.
대장은 NAS `nonbest_weights_archive_20261015/MANIFEST_<서버>.tsv`(허브 원본과 md5 일치 확인).

| 출처 | 옮긴 파일 | 용량 | 완료 | 이관 전 여유 → 2026-10-16 여유 |
|---|---|---|---|---|
| lecun `/SSDb` | 471 | 426GB | 10-15 09:50 | 26G → 371G (100% → 95%) |
| levine `/SSDe` | 449 | 245GB | 10-15 11:25 | 616G → 743G (92% → 90%) |
| bengio `/SSDb`·`/SSDe` | 415 | 523GB | 10-15 16:52 | `/SSDb` 85G → 290G · `/SSDe` 209G → 491G |
| ailab_mat2 `personal/jemo_maeng` | 956 | 1067GB | 10-15 17:43 | (공유 볼륨, 측정 생략) |
| hinton `/SSDb` | 894 | 511GB | 10-16 08:09 | 166G → 642G (96% → 81%) |
| jarvis `/`·`/SSDb` | 450/756 진행 중 | 703/1273GB | 진행 중 | `/` 50G → 691G (99% → 80%) |

- levine 의 2개는 시험 이관분(목록 451개 중 2개를 먼저 옮김)이라 대장 `levine` 행은 449개 + 시험 2개다(시험분 대장은 허브 `~/nas_moves/20261015/test/`).
- 이관 후 여유가 일부 다시 줄어든 볼륨(lecun·levine)은 다른 사용자·학습이 그 사이 쓴 것이다.
- ailab_mat2 의 `videos/`(Drone 7편·touchNplug 375편·Amphibious 20편)는 목록에 없었고 그대로 있다(gyuri 에서 확인).
<!-- nonbest-archive-20261015-result:end -->

## 6. 홈 디렉터리(OS 볼륨) 전수조사 — 2026-10-16

사용자 규칙: 큰 파일은 우분투가 설치된 볼륨(홈 디렉터리)이 아니라 SSD(`/SSDx`)에 둔다. 아래는 조사 결과이며 **아직 옮기지 않았다**.
모든 서버에서 홈은 OS 루트와 같은 파일시스템이다(hpca100만 예외: 25G 컨테이너 볼륨).

| 서버 | OS 볼륨 사용률 | 내 홈 | 산출물·데이터(옮겨야 할 것) | 환경·캐시(설치물) |
|---|---|---|---|---|
| jarvis | 80% | 599G | `src/drone-MemorySAM-develop`(281G)·`-probe`(128G)·`-s1`(56G)·`-p47`(12G) 등 체크아웃의 `outputs/` 웨이트 | HF 캐시 30G(dinov3 7B 27G), miniconda 37G |
| hinton | 51% | 333G | `probe_cache/`(182G: sam2 103G·dinov3 79G 특징 캐시), `wd/runs`(112G) | miniconda 13G |
| bengio | 94% | 323G | 없음 | HF 캐시 127G, anaconda 104G, ollama 모델 28G, pip 24G·uv 17G, openpi 12G |
| yeon | 92% | 105G | `wd/`(12G: kitti·macvi·nuscenes YOLO 데이터), `Workspace/src`(8G) | anaconda 54G, 캐시 27G |
| lecun | 93% | 92G | `dset/poongsan_v2_yolo_rgb_modal`(3.4G) | miniconda 67G, pip 18G |
| levine | 44% | 80G | `jemo_maeng/`(45G: dset 27G, src 18G — Amphibious rNs 출력이 10-15에도 쓰임) | miniconda 22G, 캐시 9.5G |
| gyuri | 34% | 64G | 없음 | anaconda 40G, cursor-server 15G |
| drone-demo | 77% | 공용 계정 | `workspace/jemo_maeng`(90G) — 나머지 1T는 다른 사용자(sangmin_park 등) | — |
| hpca100 | 컨테이너 홈 45% | 12G | 없음(작업은 `~/SSDb` wekafs) | — |
