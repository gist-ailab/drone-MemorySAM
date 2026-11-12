# 서버 정지(금 11-13 17:00 ~ 토 11-14) 대비 배치 계획 — manager 초안 2026-11-12 16:40

요청: specialist-paper inbox 20261112-162333-request-1ebb59f1 (user 2026-11-12 지시). **계획만이다. 정지·kill 은 user 동의 후.**
가정: 연구실 서버 전부가 금 17:00 에 꺼지고 토요일 중 복구. drone-demo·amphibious-demo 포함 여부는 미확인(user 확인 필요).

## 1. 지금 상태 (실시간 조회 11-12 16:25)
| 서버 | 우리 작업 | 빈 GPU |
|---|---|---|
| amphibious-demo (5090×1) | L6 s903 학습(DRN-261111-02, 증류 제거 셋째 시드, DELIVER 40ep) 11-12 16:03 기동, 1.10 it/s ≈ 1 epoch/시간 → 금 17:00 에 약 ep24, 완주 예상 토 08~10시 | 없음 |
| jarvis (4090×8) | GPU4~7: analysis 의 증류 5축 측정(kd_axes_20261111, mm_eval_v2) 8프로세스 | GPU0·GPU2 |
| drone-demo (5090×1) | 없음 | GPU0 |
| bengio·yeon·lecun·levine·gyuri | 우리 MMSAM 작업 없음(타 사용자 + touchNplug 시뮬) | 규칙상 빈 GPU 없음 |

## 2. 서버×시간대 표
| 서버 | 지금 ~ 금 17:00 | 정지 중 | 토 복구 직후 | 예상 종료 |
|---|---|---|---|---|
| amphibious-demo | L6 s903 계속(약 ep24 까지) | (꺼지면) 중단 | 같은 config 재기동 → AUTO_RESUME 로 last_checkpoint 에서 재개(꺼지지 않으면 그대로 계속) | 복구 후 약 16~17시간 |
| jarvis GPU4~7 | analysis 증류 5축 측정 완주(평가) | — | DRN-261112-02 학생 2런(2GPU DDP×2) | 런당 약 30시간 → 월 11-16 |
| jarvis GPU0·2 | 정지 전 끝나는 평가만: ① DRN-261111-01 MUSES 결측 평가(도구 준비되면) ② 밀린 평가(B1 단순 융합 legal v2 재채점, A2-B EMM15, L5 s821/s902, L2/L4 s902 재채점) | — | DRN-261112-01 MUSES 4모달 A2 s4(2GPU DDP, 768) | 약 1.5~2일 → 월 11-16 |
| drone-demo | DRN-261111-01 MUSES 결측 평가 1순위 자리(약 3~4 GPU시간) | (꺼지면) — | DRN-261112-02 나머지 학생 1런, 또는 L6 재개 대체지 | — |
| 나머지 연구실 서버 | 빈 GPU 없음 → 배치 없음 | — | 복구 직후 빈 GPU 가 생기면 DRN-261112-02 나머지 런 우선 | — |

## 3. 정지 직전(금 16:00~16:45) 회수
- amphibious L6 s903: `outputs/ReliaDINO/amphibious_deliver_..._A2L6nokd/` 의 last_checkpoint.pth·best·train.log 와 `l6s903_20261112/logs/` → drone_nas `ckpts/L6_s903_20261112/`, `train_logs/`. md5 대조.
- 회수 시점은 epoch 저장 직후(로그의 epoch 경계 확인)로 맞춘다.
- 저장은 `_atomic_save`(임시 파일 → rename)라 쓰는 도중 전원이 나가도 직전 last_checkpoint 는 남는다(train_reliadino.py:1614, :167 주석). 그래서 학습을 미리 kill 할 필요는 없다고 판단한다. user 가 원하면 16:45 에 정상 종료한다.
- jarvis 평가 산출물(kd_axes_20261111, GPU0·2 평가 결과) → drone_nas `analysis_logs/` 같은 이름, md5.

## 4. 재부팅 직후 절차(스크립트로 준비)
- 복구 감지: Monitor 폴링(ssh 접속 + `nvidia-smi` 정상 + sshfs NAS 마운트 응답), 5분 간격.
- 점검: GPU 인식(NVML 오류가 나도 드라이버 재설치 금지, 관리자 리부트 요청), tmux `jemo` 재생성, 빈 GPU 규칙(≤2000MiB·≤10%·compute-apps 0).
- 기동(모두 tmux 안, nohup 금지. amphibious 는 tmux 가 없어 setsid 를 쓰는 기존 예외):
  1. amphibious L6 s903 재개: 같은 config 재실행(AUTO_RESUME true, configs/amphibious-..._A2L6nokd.yaml:19).
  2. DRN-261112-01 MUSES 4모달 A2 s4: DRN-261019-02 의 yeon 768 config 에서 시드·SAVE_DIR·서버 경로만 바꾼 파생(아직 없음 → 정지 전에 만들어 develop 병합).
  3. DRN-261112-02 2센서 학생 4런: 교사 DRN-261109-01/-02 재사용, config 4개 아직 없음 → 정지 전에 생성·병합.
- 재개 가능 여부: train_reliadino 는 매 epoch last_checkpoint 를 저장하고 AUTO_RESUME 이 있으므로(train_reliadino.py:1755-1762), config 에 AUTO_RESUME true 면 세 항목 모두 재개된다. 손실은 진행 중이던 epoch 1개 이하다.

## 5. 막힘
- DRN-261111-01 MUSES 결측 평가: 저장소에 MUSES 용 EMM/RMM/NM 도구가 없다(tools/mm_eval_v2.py·missing_modality_eval.py 는 DELIVER 전용, CAFuser·DGFusion 은 detectron2 계열 별도 env).
- 평가 코드를 먼저 만들어야 하며, analysis 요청(20261111-022530)은 아직 미착수(claimed_by 없음)다. 금 16:00 전 완주하려면 코드가 금 오전까지 준비돼야 한다.
