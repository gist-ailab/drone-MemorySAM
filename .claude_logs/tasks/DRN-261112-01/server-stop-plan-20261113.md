# 서버 정지(금 11-13 17:00 ~ 토 11-14) 대비 배치 계획 — manager 2026-11-12 (범위 확정판 v2)

요청: specialist-paper inbox 20261112-162333-request-1ebb59f1 (user 2026-11-12 지시). **계획만이다. 정지·kill 은 user 동의 후.**

정지 범위(user 2026-11-12 확정, paper 세션 전달):
- 정지: 210 대역 = yeon·levine·bengio·lecun·gyuri(210.125.85.207 포트 구분) + /ailab_mat2 NAS.
- 유지: 172 대역 = jarvis·hinton·drone-demo·amphibious-demo·drone_nas·nas_jm.

## 1. 지금 상태 (실시간 조회 11-12 16:25)
| 서버 | 대역 | 우리 MMSAM 작업 | 빈 GPU |
|---|---|---|---|
| amphibious-demo (5090×1) | 172 유지 | L6 s903 학습(DRN-261111-02, 증류 제거 셋째 시드, DELIVER 40ep), 11-12 16:03 기동, 약 1 epoch/시간 → 완주 예상 토 11-14 08~10시 | 없음 |
| jarvis (4090×8) | 172 유지 | GPU4~7: analysis 의 증류 5축 측정(kd_axes_20261111, mm_eval_v2) | GPU0·GPU2 |
| drone-demo (5090×1) | 172 유지 | 없음 | GPU0 |
| hinton | 172 유지 | 미조회 | 미조회 |
| bengio·yeon·lecun·levine·gyuri | 210 정지 | **없음**(타 사용자 + touchNplug 시뮬만) | 규칙상 빈 GPU 없음 |

결론: **정지로 끊기는 우리 MMSAM 작업은 없다.** 정지·kill·사전 회수 대상이 없으므로 user 동의가 필요한 정지 조치도 없다.

## 2. 서버×시간대 표
| 서버 | 지금 ~ 금 17:00 | 정지 중(금 17:00 ~ 토 복구) | 토 복구 직후 | 예상 종료 |
|---|---|---|---|---|
| amphibious-demo | L6 s903 계속 | L6 s903 계속(영향 없음) | 완주 → legal v2 평가 → drone_nas 보존 | 토 오전 학습 완주 |
| jarvis GPU4~7 | analysis 증류 5축 측정 | 측정이 끝나면 DRN-261112-02 2센서 학생 2런(2GPU DDP×2) | 계속 | 런당 약 30시간 → 일~월 |
| jarvis GPU0·2 | 정지와 무관하게 지금 배치 가능: DRN-261112-01 MUSES 4모달 A2 s4(2GPU DDP, 768) 또는 밀린 평가 | 계속 | 계속 | 약 1.5~2일 → 토~일 |
| drone-demo | DRN-261111-01 MUSES 결측 평가(약 3~4 GPU시간, 도구 준비되면) | 그 뒤 DRN-261112-02 학생 1런 | 계속 | — |
| hinton | 상태 조회 후 빈 GPU 가 있으면 DRN-261112-02 나머지 런 | 계속 | 계속 | — |
| 210 서버 5대 | 우리 작업 없음 | 꺼짐 | 복구 감지 후 빈 GPU 가 생기면 DRN-261112-02 남은 런·밀린 평가 | — |

## 3. 정지 직전(금 16:00~16:45)
- 210 서버에는 우리 실행 작업이 없어 회수할 진행 중 학습이 없다.
- 210 서버에 남은 우리 산출물 중 아직 drone_nas 에 없는 것만 확인한다(11-09~10 정리에서 1118 GB 이관 완료). 행선지는 /ailab_mat2 가 아니라 drone_nas.
- /ailab_mat2 가 꺼지므로, 정지 구간에 돌릴 작업이 /ailab_mat2 경로(데이터셋·교사 가중치)를 읽지 않는지 금 오전에 확인한다. jarvis 는 /ailab_mat2 접근 금지(메모리 jarvis-ailab-mat2-hang)라 이미 로컬 경로를 쓴다.

## 4. 재부팅 직후
- 감지 대상: 210 서버 5대(ssh 접속 + `nvidia-smi` 정상)와 /ailab_mat2 재마운트(허브 `ls` 응답). Monitor 폴링 5분 간격.
- 점검: GPU 인식(NVML 오류가 나도 드라이버 재설치 금지, 관리자 리부트 요청), tmux `jemo` 재생성, 빈 GPU 규칙(≤2000MiB·≤10%·compute-apps 0).
- 기동은 tmux 안에서 한다(nohup 금지).
- 재개 가능 여부: train_reliadino 는 매 epoch last_checkpoint 를 원자적으로 저장하고(train_reliadino.py:1614) AUTO_RESUME 이 있다(:1755-1762). config 에 AUTO_RESUME true 면 중단된 학습은 진행 중이던 epoch 1개 이하만 잃고 재개된다.

## 5. 정지 전에 준비할 것
- DRN-261112-01(MUSES 4모달 A2 s4) config: DRN-261019-02 의 yeon 768 config 에서 시드·SAVE_DIR·서버 경로만 바꾼 jarvis 파생. 아직 없음 → 생성·develop 병합 필요.
- DRN-261112-02(2센서 학생 4런) config 4개: 교사 DRN-261109-01/-02 재사용. 아직 없음 → 생성·병합 필요. 교사 가중치와 MUSES 데이터가 jarvis·drone-demo 로컬에 있는지 확인 필요.

## 6. 막힘
- DRN-261111-01 MUSES 결측 평가: 저장소에 MUSES 용 EMM/RMM/NM 도구가 없다(tools/mm_eval_v2.py·missing_modality_eval.py 는 DELIVER 전용, CAFuser·DGFusion 은 detectron2 계열 별도 env). analysis 요청(20261111-022530)은 아직 미착수다. 정지와 무관하게 172 서버에서 돌 수 있으므로 마감은 정지 시각이 아니라 원고 일정이다.
