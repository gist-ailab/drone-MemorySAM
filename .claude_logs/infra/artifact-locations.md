# 산출물 위치 지도 (체크포인트·로그·분석) — 모든 세션 공용

updated: 2026-09-29 (hpca100 SSDb 2차 이관, 아래 2b 절)

> **이 문서의 용도**: "그 실험 체크포인트가 어디 갔지?"에 답하는 단일 조회처다.
> 서버에서 체크포인트가 사라져 있으면 **지워진 것이 아니라 NAS로 이관된 것**이므로, 지우기 전에 여기서 먼저 찾아라.
> 실험의 의미·판정·수치는 여기가 아니라 `experiments/registry.md`와 `experiments/log.md`에 있다. 이 문서는 **위치만** 다룬다.

---

## 1. 정규 루트 (규약)

| 종류 | 정규 위치 |
|---|---|
| 학습 웨이트(`.pth`) | `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/<묶음>_<YYYYMMDD>/` |
| 평가·분석·시각화 | 같은 루트의 `analysis_logs/<model>_eval_<YYYYMMDD>/` |
| 학습 런 로그 | 같은 루트의 `train_logs/` |
| 제출(submission) 코드·zip | `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/{code,muses}/` |
| **공유 분석 산출물(서버 간 연동)** | `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/<run_id>/` (user 지시 2026-09-18) |

원격 서버의 `outputs/`는 **작업 사본**이지 정본이 아니다. 서버 디스크가 차면 위 루트로 이관하고 서버에서는 지운다.
`/ailab_mat2`는 2026-09-18 기준 90%(여유 11T)이다. 제출물 외에 **여러 서버가 함께 읽고 쓰는 분석 산출물**도 여기 둔다
(user 지시 2026-09-18). 학습 웨이트 아카이브 정본은 그대로 `/drone_nas`이며, 분석에 필요한 체크포인트 사본만 `analysis/<run_id>/raw/ckpts/`에 둔다.

### 공유 분석 산출물 운용 (`scripts/nas_analysis_sync.sh`)

- 루트 = `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/<run_id>/`
  하위 규약: `raw/`(입력 원본 사본: 체크포인트·예측 JSON) · `preds/`(복원 예측 PNG) · `metrics/`(이미지별 지표)
  · `mining/`(실패 채굴) · `reports/`(판정 문서) · `code/`(실행 시점 도구 사본) · `MANIFEST.tsv`(대장).
- `/ailab_mat2` 마운트 현황(2026-09-18 확인): hub·yeon·lecun 있음, **hpca100 없음**(hub 경유).
- 부명령: `init` 루트 생성 · `push`/`pull` 전송(서버 인자로 그 서버에서 실행) · `verify` 체크섬 대조(전송 없음)
  · `manifest` 대장 갱신(크기·mtime·md5) · `ls` 용량·목록 · `mounts` 마운트 점검.
- NFS가 그룹 변경을 거부하므로 전송은 `rsync -rt --no-perms --no-owner --no-group`으로 한다
  (속성 보존 옵션을 켜면 `chgrp ... Operation not permitted`로 실패한다 — 2026-09-18 실측).
- 첫 사용 예: `baseline_failure_20260917`(DGFusion·CAFuser 실패 분석). DGFusion은 yeon, CAFuser는 lecun에서
  예측 JSON과 분석 대상 체크포인트를 이 루트로 복사했다(lecun은 작업 배치 금지 서버라 파일 복사만 했다).
`/drone_nas`는 여유 37T이며 웨이트 아카이브의 정본이다.

---

## 1b. 🔴 헤드라인 체크포인트 — 정본 위치 (2026-09-17 신설)

논문·벤치 표의 "우리 최고"를 만든 체크포인트다. **이 표에 없는 헤드라인 수치는 재평가할 수 없다** — ISSUE-035(56.99 런 경로 미기록) 이후 신설했다.

| 수치 | 런 | 체크포인트 정본(NAS) | md5 | 남은 사본 |
|---|---|---|---|---|
| **DELIVER test 56.99**(val-best ep70, 768 학습 트레이너 val 67.79 · 1024 재평가 val 69.44) | `jarvis_deliver_rgbdel_P46_ctr_c3only`(C3 λ0.1, TAPS 없음, 시드 고정 기능 없던 무작위 시드 런) | `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/P46_c3only_base_ep70_test5699_20260730/epoch70_67.79_top1_checkpoint.pth` | `d340e3fe9824bd922fc7fe6eff7a8b26` | yeon `/SSDe/jemo_maeng/temp_ckpts/` · yeon `drone-MemorySAM-p49/_eval_ckpts/p46_c3only_lam01_ep70/` (임시 경로 — 지워질 수 있음). jarvis 원본 디렉터리는 정리됨 |
| **MUSES 공식 test 79.788** | P39.1-rank 시드2 3모달 | (미기록 — 확인 필요) | — | 제출 zip `/ailab_mat2/.../submission/muses/muses_P39_1_seed2_3modal_ep208_submission.zip` |

평가 config: 56.99 런 = `configs/eval/jarvis-deliver_rgbdel_P46_ctr_c3only_eval1024.yaml`(학습 config 에서 EVAL·TEST IMAGE_SIZE 만 1024 로 파생).

## 2. hpca100(A100×4) 이관 — 2026-09-09

### 왜 옮겼나

hpca100의 작업 볼륨 `~/SSDb`(2.0T, **다른 사용자와 공유**)가 95%까지 차서 여유가 121G밖에 남지 않았다.
그 때문에 9월 8일에 기동한 학습 3건(E12·E1M·E2s2)이 SSDb가 아니라 `/tmp` 오버레이로 우회해 돌아가고 있었다.
내 산출물 약 148G를 NAS로 옮겨 SSDb 여유를 약 270G로 회복시키는 것이 이번 작업의 목적이다.

### 이관 목적지

```
/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/hpca100_archive_20260909/
```

run 디렉터리를 통째로 `<run이름>.tar` 로 묶어 저장했다(2026-09-02 아카이브와 같은 방식).
서버 디스크를 쓰지 않도록 스트리밍 tar(`ssh hpca100 "tar cf - ..." > NAS/<이름>.tar`)로 전송했다.

### 실험별 대조표

| 실험(run 이름) | 무엇을 본 실험인가 | 데이터셋 | 판정 | 이관 후 위치 |
|---|---|---|---|---|
| `hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E2` | 일일 카드 **E2 — LoRA를 attention QKV뿐 아니라 전 선형층으로 확장**(`LORA_TARGETS` qkv_full·proj·fc1·fc2) | DELIVER 4모달 | 완주(40ep, 트레이너 val 68.65@ep40) | `hpca100_archive_20260909/…_screen40_E2.tar` |
| `hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821` | **P50-EXT 사전학습 가중치를 채택할지 판정하는 게이트 파인튠** | DELIVER 4모달 | 🔴 기각(2026-09-07, 공식 legal test 53.10 < 기준 55.25) | `…_p50ext_seed821.tar` |
| `hpca100_muses_rgbelr_P52_seed20260901` | **P52 RxDINO 본런 시드1** | MUSES 4모달 | 🟡 보류(감사 중, val-best 79.63@ep54) | `…_P52_seed20260901.tar` |
| `hpca100_muses_rgbelr_P52_seed20260902` | **P52 RxDINO 본런 시드2** | MUSES 4모달 | 🟡 보류(GPU를 E2에 양보) | `…_P52_seed20260902.tar` |
| `hpca100_muses_rgbel_P46_c3only_lam02` | **P46-CTR의 C3(프로토타입) 단독 λ=0.2를 MUSES로 이식** | MUSES 3모달 | 완주 81.65@ep136, base −0.97 미달 | `…_P46_c3only_lam02.tar` |
| `hpca100_muses_rgbel_P39_1_seed2_physaugoff_screen40_E7` | 일일 카드 **E7 — PhysAug 증강을 끈 기준선** | MUSES 3모달 | 완주(공식 native val 80.0756) | `…_screen40_E7.tar` |
| `hpca100_muses_rgbel_P39_1_seed2_physaugon_screen40_E7c` | 일일 카드 **E7c — PhysAug를 켠 대조군**(E7과 그 항목만 다름) | MUSES 3모달 | 완주 79.9417 = E7 대비 −0.13 → **PhysAug 폐기 근거** | `…_screen40_E7c.tar` |
| `hpca100_deliver_rgbdel_P46_ctr_c2c3`(probe 레포) | **C2(masked consistency)의 순기여 측정** A/B | DELIVER 4모달 | 조기종료(2026-08-16) — C2 유해 확정 | `…_ctr_c2c3_probe.tar` |
| `tmp_ckpt/`(느슨한 파일 2개) | MUSES seed2 82.62 · DELIVER test 57.05(무효 수치, test-best라 규약상 사용 금지) | — | 참고용 보관 | `hpca100_tmp_ckpt_loose.tar` |
| `ckpt/p50ext_deliver/` | P50-EXT 게이트용 스테이징 웨이트 | DELIVER | 기각 실험의 부속물 | `hpca100_ckpt_p50ext_deliver.tar` |
| `hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_screen40_E1M`(SSDb 잔여분) | 일일 카드 **E1M — 카드 E1(중간층 4탭 읽기)을 MUSES로 이식** | MUSES 3모달 | `/tmp`에서 학습 진행 중(ep25까지) — 이건 그 이전 구간 | `…_screen40_E1M_ssdb.tar` |
| `ckpts_p38muses_import/` · `ckpts_p39muses_import/` · `ckpt_stage/` | 타 서버에서 가져온 P38·P39 MUSES 웨이트와 그 변형(router 제거·seg 전용) | MUSES | 대부분 NAS에 정본 존재(§2.2) | `hpca100_ckpts_p38muses_import.tar` 등 |

| `/tmp/jemo_scratch/…_screen40_E1M`(완주분) | 위 E1M이 **2026-09-08 22:44에 40 epoch 완주**한 결과 — 휘발성 `/tmp`에 있어 함께 회수 | MUSES 3모달 | 트레이너 val-best **80.64@ep35**(최종 ep40 80.49), 총 학습 6시간 15분 | `…_screen40_E1M_COMPLETED_tmp.tar` |

**결과**: 15개 tar, 총 158G를 옮겼다. hpca100 SSDb의 여유는 **121G → 287G**(사용률 95% → 87%)로 늘었고,
레포 디렉터리는 123G → 880M, probe 레포는 20G → 163M로 줄었다.

### 2.1 서버에 남긴 것 (지우지 않았다)

| 대상 | 왜 남겼나 |
|---|---|
| `/tmp/jemo_scratch/outputs/…E12`, `…E2s2` | **아직 돌고 있는 학습 2건의 저장 경로**다. 손대면 학습이 깨진다. 완주하면 회수해야 한다. |
| `/tmp/jemo_scratch/outputs/…E1M` | 완주했고 아카이브로 **회수 완료**. 후속 공식 채점에 쓰일 수 있어 `/tmp` 사본은 남겨 뒀다(오버레이 여유 319G). |
| `~/SSDb/jemo_maeng/dset/` | 데이터셋. 옮기면 학습이 못 읽는다. |
| `~/SSDb/jemo_maeng/cache/`(56G) | HuggingFace 백본 캐시. 지우면 백본을 못 찾아 **랜덤 초기화로 학습이 조용히 망가진다**(ISSUE: HF_HUB_OFFLINE 함정). |
| `~/SSDb/jemo_maeng/venv/`(5.7G) | 학습 실행 환경. |

`/tmp`는 컨테이너 오버레이라 **휘발성**이다. E1M의 ep25 산출은 이미
`ckpts/daily_cards_20260908/_tmp_volatile/` 에 회수해 두었다. `/tmp`에서 도는 학습이 끝나면 반드시 회수해야 한다.

### 2.2 이미 NAS에 정본이 있던 것 (md5로 동일함을 확인)

서버의 스테이징·import 폴더에는 다른 서버에서 가져온 사본이 섞여 있었다.
아래 5개 파일은 **NAS 정본과 md5가 완전히 같음(bit-identical)** 을 확인했으므로, 서버에서 지워도 잃는 것이 없다.

| 서버 파일 | NAS 정본 | md5 |
|---|---|---|
| `ckpts_p38muses_import/epoch156_82.22_top1_checkpoint.pth` | `ckpts/P38_MUSES_20260720/epoch156_82.22_top1_checkpoint.pth` | `6ba4873c…` 일치 |
| `ckpt_stage/P38_ep156.pth` | 위와 같은 파일 | `6ba4873c…` 일치 |
| `ckpts_p39muses_import/epoch146_81.52_top1_checkpoint.pth` | `ckpts/P39_MUSES_3modal_20260720/epoch146_81.52_top3_checkpoint.pth` | `8d382554…` 일치 |
| `ckpt_stage/P39_ep146.pth` | 위와 같은 파일 | `8d382554…` 일치 |
| `ckpts_p39muses_import/epoch122_82.01_top1_checkpoint_brokenradar.pth` | `ckpts/MUSES_P39_4modal_brokenradar_20260721/epoch122_82.01_top1_checkpoint.pth` | `d2a4c493…` 일치 |

`epoch156_..._nodetrouter.pth`(router를 뺀 변형)와 `..._segonly.pth`(seg 전용 변형)는 NAS에 없는 고유 산출물이다.
그래서 이 세 폴더는 **통째로 tar에 담아 회수**했다 — 중복 5개까지 함께 들어가지만, 그렇게 해야 아카이브가 그 자체로 완결된다.

`~/SSDb/…/outputs/ReliaDINO/hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_screen40_E1M`(7.4G)은
`/tmp` 쪽 진행분이 같은 파일을 포함한 상위 집합으로 보였지만(ep5·10·15·last 크기 동일, `/tmp`에는 ep20·25가 추가),
저장될 때 top 순위가 재배열돼 파일명이 서로 달라 자동 대조가 되지 않았다.
같은 파일임을 추정만 하고 지우는 대신 **그대로 회수했다**(`…_E1M_ssdb.tar`).

### 2.3 전송 무결성을 어떻게 확인했나 (원본 삭제 전 3단계)

서버 원본을 지우기 전에 세 단계를 모두 통과시켰다.

1. **크기 대조** — run 디렉터리의 실제 바이트 수(`du -sb`)와 만들어진 tar 크기를 비교했다. tar는 512바이트 블록 정렬과 헤더 때문에 원본보다 조금 크다. 스트림이 중간에 끊기면 크기가 크게 모자라므로 절단이 잡힌다.
2. **구조 대조** — 각 tar를 `tar tf`로 끝까지 읽어 헤더 정렬이 깨지지 않았는지 확인하고, 담긴 항목 수를 서버 원본의 파일 개수(`find | wc -l`)와 맞춰 봤다. 14건 전부 일치했다(E2 6,028개 등).
3. **데이터 대조** — 각 tar에서 체크포인트 파일을 하나씩 실제로 꺼내 md5를 서버 원본과 대조했다. 14건 전부 일치했다.

🔴 **3단계를 붙인 이유**(이번 작업에서 실제로 난 사고): 전송 스크립트를 `nohup`으로 분리해 띄웠는데,
세션이 그 작업을 완료로 처리한 뒤에도 프로세스가 살아 있었고 `ps`에는 보이지 않았다.
죽은 줄 알고 새 스크립트를 띄운 탓에 **여러 프로세스가 같은 출력 파일에 겹쳐 쓰는 상황**이 벌어졌다.
크기만 봤다면 정상으로 보였을 것이다. **장시간 전송은 `nohup`으로 분리하지 말고 세션이 추적하는 방식으로 띄우고,
겹쳐 쓴 정황이 있으면 크기가 아니라 내용으로 확인하라.**

`~/SSDb/…/outputs/ReliaDINO/hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_screen40_E1M`(7G)은
`/tmp` 쪽 진행분이 같은 파일을 모두 포함한 상위 집합이라(ep5·10·15·last 크기 동일, /tmp에는 ep20·25가 추가) 별도 회수 없이 정리 대상으로 뒀다.

---

<!-- hpca100-archive-20260929:begin -->
## 2b. hpca100(A100×4) 이관 — 2026-09-29 (user 승인 "체크포인트는 이동해두되 … 모두 기록")

### 왜 옮겼나

hpca100 작업 볼륨 `~/SSDb`(2.0T, 다른 사용자와 공유)가 98%(여유 46~49G)까지 찼다. P56-A(모달 충돌 학습) 두 시드가 같은 볼륨에 체크포인트를 쓰므로, 끝난 실험의 체크포인트를 NAS 로 옮겼다.

### 요약

| 항목 | 값 |
|---|---|
| 이관 목적지(NAS) | `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/hpca100_archive_20260929/` |
| 폴더 구조 | hpca100 `/home/jovyan/SSDb/jemo_maeng/` 아래 상대 경로를 그대로 유지 |
| 대장 | 같은 폴더의 `MANIFEST.tsv`(기계 판독용) · `README.md`(설명) |
| 검증 | 파일마다 크기 + md5 를 서버 계산값과 대조. 35개 중 35개 통과 |
| 서버에서 지운 것 | 32개, 59.7G (검증 통과분만. 삭제 직전에 서버 크기·실행 중 프로세스 사용 여부 재확인) |
| 서버에 남긴 것 | 사본만 만든 3개 + 제외 2개(아래 표) |
| 서버 쪽 기록 | `~/SSDb/jemo_maeng/ckpt_inventory_20260929_md5.tsv`(이관 전 전수 목록) · `ckpt_del_result_20260929.tsv`(마지막 삭제 회차) |

### 체크포인트별 위치

경로는 hpca100 `~/SSDb/jemo_maeng/` 기준 상대 경로이며, NAS 에서는 위 목적지 폴더 아래 같은 상대 경로에 있다.

| 상대 경로 | 무엇 | 처리 | md5 | 크기 | 서버 원본 |
|---|---|---|---|---|---|
| `rescore_armc/armc821.pth` | 미확인(프로젝트 로그에 기록 없음. 폴더 이름은 재채점용 사본을 뜻함) | 이동 | `c6e9d10cee971c552dc772a419878a31` | 1.87G | 삭제됨(NAS 가 유일본) |
| `d4_ours/ckpts/E1_s3_epoch70.pth` | E1(중간층 4탭 읽기) 확정 200에폭 시드3, 학습기 val-best epoch70 | 이동 | `d8e3e8babe75d4fdc227f99556cc2e2f` | 1.92G | 삭제됨(NAS 가 유일본) |
| `d4_ours/ckpts/E13_s1_epoch100.pth` | E13(4탭 + 센서별 클래스 prototype) 확정 200에폭 시드1, val-best epoch100 | 이동 | `1517c12a64ddf7ff23d06965e34c2654` | 1.92G | 삭제됨(NAS 가 유일본) |
| `d4_ours/ckpts/E1_s2_epoch134.pth` | E1(중간층 4탭 읽기) 확정 200에폭 시드2, 학습기 val-best epoch134 | 이동 | `79a3ff39600610a80aebf0ae2134801b` | 1.92G | 삭제됨(NAS 가 유일본) |
| `d4_ours/ckpts/E17_s1_epoch35.pth` | E17(E1 + 고해상도 세부 가지) 40에폭 스크린 시드821, val-best epoch35 | 이동 | `eedbd1983ffc6b55c5de64d8e7ac8496` | 1.92G | 삭제됨(NAS 가 유일본) |
| `ckpt/b0screen821_epoch40_65.4_top1_checkpoint.pth` | B0(탭 없는 기준 구조) 40에폭 스크린 시드821, val-best epoch40(학습기 val 65.4) | 이동 | `cbaa692f154c153eda99a171f548f7a8` | 1.87G | 삭제됨(NAS 가 유일본) |
| `ckpt/e1shared821_bengio_epoch40_67.03_top1.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821, bengio 학습 사본 val-best epoch40(67.03). 게이트 판정에 쓴 사본 | 이동 | `de6490fb4a22bd6675c1a864166ed876` | 1.90G | 삭제됨(NAS 가 유일본) |
| `ckpt/e1screen821_epoch35_67.25_top1_checkpoint.pth` | E1(중간층 4탭 읽기) 40에폭 스크린 시드821, val-best epoch35(67.25). 40에폭 카드의 짝 기준선 | 이동 | `e69d7deb20fe04d8d7b86f34aa265421` | 1.92G | 삭제됨(NAS 가 유일본) |
| `ckpt/epoch90_67.3_top1_checkpoint.pth` | C3-only(클래스 prototype 손실만, 탭 없음) 200에폭 시드821 val-best epoch90. E1·E13 확정 게이트의 분모. NAS 정본은 ckpts/p46_c3only_seed20260821_200ep_20260918/ 에도 있음 | 이동 | `854a2c9d9f772560261ec3725c8b1146` | 1.87G | 삭제됨(NAS 가 유일본) |
| `ckpt/e13m_full200_s3407_epoch130_82.21_top1_checkpoint.pth` | MUSES E13M(4탭 + 센서별 클래스 prototype, PhysAug off) 200에폭 시드3407, val-best epoch130(82.21). 제출 zip 20260920_E13M 의 원본 | 이동 | `fc71970601c394c335ffd5236407acf9` | 1.86G | 삭제됨(NAS 가 유일본) |
| `p55_0929/loo_deg0/smoke/p55_gate_full.pth` | P55(무감독 다중 블록 게이트) leave-one-out 목표 팔(모델 자신의 모달 제거 손실을 게이트 목표로), Q2 시드821 위, 기동 점검용 20배치 학습 산출 | 이동 | `1ce3879f07b0b4c26901ecfc88b8ed3e` | 1.45G | 삭제됨(NAS 가 유일본) |
| `p55_0929/none_deg0/smoke/p55_gate_full.pth` | P55(무감독 다중 블록 게이트) 대조군 팔(분할 손실만으로 게이트 학습), Q2 시드821 위, 기동 점검용 20배치 학습 산출 | 이동 | `58b4411d889859b508521fafed1ab73c` | 1.45G | 삭제됨(NAS 가 유일본) |
| `p55_0929/none_deg0/train/p55_gate_full.pth` | P55(무감독 다중 블록 게이트) 대조군 팔(분할 손실만으로 게이트 학습), Q2 시드821 위, 게이트 4에폭 학습 산출(전체 모델 + 게이트) | 이동 | `dbe77d29c434f4d98ef6f55ff6ea362b` | 1.45G | 삭제됨(NAS 가 유일본) |
| `src/dms-hub-develop/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_Q2/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth` | Q2(두 패스 + 동결 E1 교사 증류) 40에폭 스크린 시드902, 마지막 에폭 체크포인트 | 이동 | `cc5862bd1fcc8d1f9b327113c76fcf48` | 1.92G | 삭제됨(NAS 가 유일본) |
| `src/dms-hub-develop/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_Q2/DELIVER_ReliaDINO-ViTL16_idel/epoch40_66.06_top1_checkpoint.pth` | Q2(두 패스 + 동결 E1 교사 증류) 40에폭 스크린 시드902, 학습기 val-best epoch40(66.06) | 이동 | `3b892ef3cb6dda07ff52301184879854` | 1.92G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/epoch40_65.64_top2_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 학습기 val 상위 체크포인트(epoch40_65.64_top2) | 이동 | `c79a14f6f47f450a9d2738e72eb1d16c` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/test_epoch25_55.29_top3_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, test 상위 체크포인트(test_epoch25_55.29_top3). 🔴 test-best 는 판정·헤드라인에 인용 금지 | 이동 | `1396f501b7911c514379ebe53f4f8b50` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/test_epoch30_55.28_top4_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, test 상위 체크포인트(test_epoch30_55.28_top4). 🔴 test-best 는 판정·헤드라인에 인용 금지 | 이동 | `4bbda17045951bc9949ccde8d34fc384` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/test_epoch35_55.46_top2_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, test 상위 체크포인트(test_epoch35_55.46_top2). 🔴 test-best 는 판정·헤드라인에 인용 금지 | 이동 | `c48ca0b535d260544ade36f1b874c78a` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/epoch15_65.02_top3_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 학습기 val 상위 체크포인트(epoch15_65.02_top3) | 이동 | `ba6839ff54e01f8711d0358c846e6d9d` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/epoch35_65.72_top1_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 학습기 val 상위 체크포인트(epoch35_65.72_top1) | 이동 | `599e9f6e09afa8230c98ff4ca9b31c60` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/epoch30_65.02_top4_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 학습기 val 상위 체크포인트(epoch30_65.02_top4) | 이동 | `dd973bb3d9e290af3d679caea8f1f650` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/test_epoch20_55.56_top1_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, test 상위 체크포인트(test_epoch20_55.56_top1). 🔴 test-best 는 판정·헤드라인에 인용 금지 | 이동 | `70025cc7ba6e0a8e32be8cc926ae929e` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/test_epoch15_55.15_top5_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, test 상위 체크포인트(test_epoch15_55.15_top5). 🔴 test-best 는 판정·헤드라인에 인용 금지 | 이동 | `87e8711dfee44ad3f451dce30151d13a` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/epoch20_64.67_top5_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 학습기 val 상위 체크포인트(epoch20_64.67_top5) | 이동 | `724f7ce9a167aad1f4d347d138b945ad` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1shared/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth` | E1-shared(전 센서 공유 LoRA r16) 40에폭 스크린 시드821 hpca100 학습 사본, 마지막 에폭 체크포인트 | 이동 | `22c0b39c6aa4d383e82e2b49fcc6526e` | 1.90G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_R2/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth` | R2(연결 성분 soft-IoU 손실) 40에폭 스크린 시드821 hpca100 사본, 마지막 에폭 체크포인트 | 이동 | `a767cad243bd7d30b1df1d552d1f93c0` | 1.92G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_R2/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.6_top1_checkpoint.pth` | R2(연결 성분 soft-IoU 손실) 40에폭 스크린 시드821 hpca100 사본, 학습기 val 상위 체크포인트(epoch40_67.6_top1) | 이동 | `baa6acd9115ca5a3078cdafecfbfb656` | 1.92G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_R1/DELIVER_ReliaDINO-ViTL16_idel/epoch25_67.62_top1_checkpoint.pth` | R1(depth 경계 prior refinement) 40에폭 스크린 시드821 hpca100 사본. 기각된 카드, 학습기 val 상위 체크포인트(epoch25_67.62_top1) | 이동 | `e749135adcb0e58dfb357cb472fb2a77` | 1.93G | 삭제됨(NAS 가 유일본) |
| `src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_R1/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth` | R1(depth 경계 prior refinement) 40에폭 스크린 시드821 hpca100 사본. 기각된 카드, 마지막 에폭 체크포인트 | 이동 | `e3b315951c17e6331460cffd41c80f75` | 1.93G | 삭제됨(NAS 가 유일본) |
| `robust_0923/ckpts/Q3_s821_epoch30_66.74.pth` | Q3(품질 헤드 QAF 본 카드) 40에폭 스크린 시드821, val-best epoch30(66.74). 종료된 카드 | 이동 | `1bafa392ff3a4be268d6dae16b9e16c5` | 2.08G | 삭제됨(NAS 가 유일본) |
| `dump_e1screen/epoch35_67.25_top1_checkpoint.pth` | E1 40에폭 스크린 시드821 val-best epoch35 의 중복 사본(ckpt/e1screen821_… 과 md5 동일) | 이동 | `e69d7deb20fe04d8d7b86f34aa265421` | 1.92G | 삭제됨(NAS 가 유일본) |
| `d4_ours/ckpts/E1_s1_epoch140.pth` | E1(중간층 4탭 읽기) 확정 200에폭 시드1(20260821), 학습기 val-best epoch140. 헤드라인 3시드 중 하나이며 P56-A 의 동결 교사 | 사본 | `c648f925fb7c131daf1324af2b1d7e84` | 1.92G | 남아 있음 |
| `p55_0929/loo_deg0/train/p55_gate_full.pth` | P55(무감독 다중 블록 게이트) leave-one-out 목표 팔(모델 자신의 모달 제거 손실을 게이트 목표로), Q2 시드821 위, 게이트 4에폭 학습 산출(전체 모델 + 게이트) | 사본 | `0106ffd3b4ffa6a895b3bd529be10068` | 1.45G | 남아 있음 |
| `robust_0923/ckpts/Q2_s821_epoch40_67.0.pth` | Q2(두 패스 + 동결 E1 교사 증류) 40에폭 스크린 시드821, val-best epoch40(67.0). 강건 측정·P55·오라클 분석의 바탕 | 사본 | `c93de37b389a20a77e3bfd369123734c` | 1.92G | 남아 있음 |

### 옮기지 않은 것

| hpca100 경로 | 이유 |
|---|---|
| `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/weights/rf-detr-large-2026.pth` | 사전학습 가중치(학습 산출물 아님) |
| `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A/DELIVER_ReliaDINO-ViTL16_idel/last_checkpoint.pth` | 학습 중인 SAVE_DIR(파일이 교체됨) |

### 되돌리는 법

```bash
# 허브에서 실행. <상대경로> 는 위 표의 첫 열
rsync -t --no-perms --no-owner --no-group "/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/hpca100_archive_20260929/<상대경로>" "hpca100:/home/jovyan/SSDb/jemo_maeng/<상대경로>"
ssh hpca100 "md5sum /home/jovyan/SSDb/jemo_maeng/<상대경로>"   # 표의 md5 와 같아야 한다
```

서버의 실행 스크립트·eval config 가 가리키던 경로는 바꾸지 않았다. 옮긴 체크포인트를 다시 쓰려면 위 명령으로 같은 경로에 되돌려 놓는다.
<!-- hpca100-archive-20260929:end -->

## 3. 복원 방법

NAS의 tar에서 특정 실험을 서버나 로컬로 되살릴 때:

```bash
A=/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/hpca100_archive_20260909

# 무엇이 들어 있는지만 확인
tar tvf $A/<run이름>.tar | head

# 특정 체크포인트 하나만 꺼내기 (전체를 풀 필요 없다)
tar xf $A/<run이름>.tar -C <목적지> '<run이름>/*/last_checkpoint.pth'

# 서버로 바로 되돌리기 (11G 기준 약 11분, 17MiB/s)
cat $A/<run이름>.tar | ssh hpca100 "tar xf - -C ~/SSDb/jemo_maeng/src/drone-MemorySAM/outputs/ReliaDINO"
```

보류 상태인 **P52 시드1·시드2를 재개**하려면 위 방식으로 `last_checkpoint.pth`를 서버에 되돌린 뒤 `AUTO_RESUME`으로 이어가면 된다.

---

## 4. 기존 아카이브 색인 (같은 `ckpts/` 루트)

| 묶음 | 내용 |
|---|---|
| `hpca100_archive_20260902/` | hpca100 1차 이관 250G — P38-m2f·P39-DPC·P40-RCA·P46-CTR·P51-CMLC·P41~P44·P47 계열 20건(tar) |
| `hpca100_archive_20260909/` | **이번 2차 이관**(이 문서 §2) |
| `daily_cards_20260908/` | 일일 카드 E2·E7·E7c의 대표 웨이트 + `/tmp` 휘발분(E1M ep25) 회수분 |
| `B200_backup_20260715/`, `B200_final_sweep_20260716/` | B200 서버 반납 전 회수분 |
| `P34_final_20260713/`, `P36_physaug_20260715/`, `P37b_DELIVER_20260719/`, `P38_DELIVER_20260720/` | DELIVER 계열 주요 완주 웨이트 |
| `P38_MUSES_20260720/`, `P39_MUSES_3modal_20260720/`, `MUSES_P34_*`, `MUSES_P37a_3modal_20260719/` | MUSES 계열 정본 웨이트 |
| `MUSES_P39_4modal_brokenradar_20260721/` · `MUSES_P39_4modal_radarfix_20260722/` | radar 디코딩 버그(ISSUE-025) 전/후 4모달 웨이트 |
| `det_D1_*`, `det_P37_*`, `P29*`, `P30*` | detection 계열 웨이트 |
| `analysis_logs/` (형제 폴더) | 평가·분석 산출물 40여 묶음 |

---

## 5. 다른 세션이 지켜야 할 것

1. 서버에서 체크포인트가 안 보이면 **먼저 이 문서와 `ckpts/` 색인을 확인**하라. 재학습하지 마라.
2. 서버 산출물을 지우기 전에는 **NAS 사본과 md5가 같은지 확인**한다. 크기만 같은 것으로는 부족하다.
3. 새로 이관하면 이 문서 §2와 §4에 행을 추가한다. 아카이브 폴더에는 `README.md`를 같이 둔다.
4. `/tmp`에 저장하는 학습은 휘발성이다. 완주 즉시 NAS로 회수한다.
