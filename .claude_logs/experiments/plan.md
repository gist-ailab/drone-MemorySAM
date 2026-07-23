---
created: 2026-07-16
updated: 2026-07-21 (ISSUE-025 MUSES radar 디코딩 버그 수정 반영 → 대기열 #3 "P39-4모달 radar-fix 재실험" 신설 + 사고 기록 1줄, 이하 대기열 번호 +1) ; ISSUE-026 ColorAugSSD RGB 붕괴 버그 반영 → hpca100 P39-DPC resume 오염 표기 + 사고 기록 1줄 + 대기열 #1 클린런 표기 ; 2026-07-23 대기열 #1 "P39.1 Rank 수리" MUSES-jarvis 분기 착수 → 실행중 표에 행 추가(jarvis 2,3,4,5, 기동검증 통과)
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

## 🖥 GPU 예약 현황 (2026-07-18 12:40)

| 서버 | GPU | 상태 | ETA |
|---|---|---|---|
| **hpca100** | 0-3 (A100×4) | 🔵 seg-P38(MaskQueryLite) 본학습 중(07-18 launch) | **07-19 완주 예상** |
| **jarvis** | 2,3,4,5,6 (4090×5) | 🟢 P37a-CEFR(ckpt=false, eff20) 학습 중 best val 62.56@ep24 → 완주 시 P37b 자동 | P37a **07-18 02:06** → P37b 이어서 |
| **yeon** | 3,5,6,7 (3090×4) + GPU0(seg-P38 스모크) | 🔴 det_P37 재붕괴(ep17 절벽), best ep11 0.8367. 완주/중단 user 판단 대기 / GPU0=seg-P38 실데이터 2ep 스모크 진행 중 | ~07-18 06:08 (det_P37) / ~07-18 14:00 (스모크) |
| **bengio** | — | 🔴 **노드 CUDA 전체 장애(GPU5 HW 고장) → 재부팅 후 SSH 미복귀. 물리 개입 필요** | 불명 |
| **lecun** | — | 🔴 타인(openvla) 점유 | — |
| ~~B200~~ | — | 🔴 상실(07-15 마감) | — |

## 🔬 실행 중

| 실험 | 서버/GPU | EPOCHS | ETA | 우선순위 | 목적 |
|---|---|---|---|---|---|
| **diag_C (dup_lidar)** | bengio 4-7 | ep6까지만 | ~1h | **P0** | radar *콘텐츠* vs *4모달 구조* 판정. ep2=49.08 ≈ ArmB 48.37 → 구조 의심 |
| **det_P37 (yeon)** | yeon 3,6,7 | 50 | 07-17 19:00 | **P1** | 붕괴 처방 검증(eff 18+LR 2e-4 유지). ep5까지 warmup 완주, AP50 0.827~0.846 안정 |
| **seg-P38 (MaskQueryLite) 본학습** | hpca100 0-3 (A100×4) | **200** | **07-19 (~24-26h)** | **P0** | 본학습 launch(07-18 서버시간 03:39:31 ≈ KST 12:39). config `configs/hpca100-deliver_rgbdel_P38_m2f.yaml`, develop @c3d1184, launch 스크립트 `launch_p38_m2f.sh`, log `logs/hpca100-deliver_rgbdel_P38_m2f/run_20260718_033931.log`. ~0.77s/it·497it/ep. 기동 검증 통과(iter 342→420/497 전진, 4GPU 25GB/83-100%, 에러 0, M2F ENABLE 확인, params 355.4M/trainable 52.3M). 판정 게이트 = P36 fair(val 67.74/test 55.62) 대비 + thin-class(Wall/Water/RailTrack) IoU |
| **seg-P38 스모크** (실데이터 2ep) | yeon GPU0 | **2** | ~07-18 14:00 | P1 | 본학습 선행조건이던 실데이터 미검증 스모크 진행 중(합성 스모크만 PASS였음). log `/SSDb/jemo_maeng/src/p37_test/logs/p38smoke_20260718_121834.log`, 10.2GB@bs1GPU. 완주 시 GPU0 반납, 수치는 참고용(본학습은 이미 hpca100에서 병행 launch됨) |
| **hpca100 P39-DPC resume** (DELIVER 4모달) | hpca100 GPU 2,3 | 200 | 07-22 09:00 | — | resume 후 val 44ep·test 64ep 무갱신 정체(val 66.14@ep96/test 55.50@ep76, P38 대비 +0.95/+0.45 계보최고 유지). 🔴 **ISSUE-026 오염 상태로 학습 중 — 지속/중단 user 판단 필요**(07-16 이후 DGFUSION_AUG:true 런이라 ColorAugSSD RGB-dropout 오염 해당, 상세 `issues/issues-and-fixes.md` ISSUE-026) |
| **P39.1-rank MUSES** (jarvis, 대기열 #1 착수) | jarvis 2,3,4,5 (4090×4) | 300 (ep30 조기게이트) | ep30 게이트 ~07-23 18:35 · 완주(300ep) ~07-24 09:30 (추정, eval 오버헤드 별도) | **P0** | R-1(gated_mlp trunk, γ=0.1 init) + R-2(VICReg var+cov, lidar×1.0/기타×0.25) + M-2(gate/calib/veto off) — P39-MUSES 표준분석이 지목한 lidar effective-rank 붕괴(4.7)·fog_night 붕괴(62.68) 수리. config `configs/jarvis-muses_rgbel_P39_1_rank.yaml`, develop @a06b666(≥ac5c7fe). 07-23 16:5x 기동, tmux `jemo:p39_1_rank`, log `logs/jarvis_muses_rgbel_P39_1_rank_*.log`. **기동검증 통과**: iter 0→160/375(epoch1) 전진 확인·GPU 2,3,4,5 전부 99-100%util·~19GB/24GB(활성화 수준)·에러 0·wandb만 미설정(비치명, no-API-key). **판정 게이트(사전등록, ep30)** = lidar effective-rank ≥15 & fog_night drop-lidar ≥4.0 (미달 시 R-3: r8→16+rsLoRA 재기동) |

## 📋 대기열 (우선순위 순)

| # | 실험 | 필요 자원 | 언제 | 근거 |
|---|---|---|---|---|
| **1** | **P39.1 Rank 수리 본학습** (DELIVER + MUSES) | hpca100 4×A100(DELIVER) / jarvis 4090×N(MUSES) | 선행조건 분석 **완료**: ① MUSES fog val per-scene 감사 완료 → 파국장면 가설 기각·GO 판정([analysis/2026-07-21-p39-fog-scene-audit.md](analysis/2026-07-21-p39-fog-scene-audit.md)) ② P39 ckpt trunk_exp-off 재측정은 **무효 판정으로 취소**(ep30 조기판정 rank 게이트가 이를 대체) + yeon 실데이터 스모크. hpca100/jarvis 첫 빈 슬롯 (**MUSES/jarvis 분기 07-23 16:5x 착수 — 위 "실행 중" 표 참조, DELIVER/hpca100 분기는 미착수**) | **구현 완료(develop ac5c7fe)** — P39-MUSES 표준분석이 지목한 lidar rank 붕괴(4.7) + fog_night 붕괴(62.68)를 즉시 수리. R-1: V1 트렁크 결합을 `fused += tanh(γ)·MLP_m(f_m)`(LN→1×1→GELU→1×1, γ init 0.1 — 0이면 gradient 완전 차단이라 절충, 스모크 실증)로 교체. R-2: VICReg var+cov 정규화(per-modal 토큰, lidar ×1.0/기타 ×0.25, λ_var 0.1/λ_cov 0.01, 2048 서브샘플, fp32). M-2: gate/calib/veto config off(fog_night 유해 실증 반영). eval마다 per-modal effective-rank 로그(`p391/rank_*`) 추가. **판정 게이트(사전 등록, ep30)** = lidar effective-rank ≥15 & fog_night drop-lidar ≥4.0(미달 시 R-3: r8→16 + rsLoRA로 재기동). **EPOCHS 200**(ep30 조기판정 규칙). **ISSUE-026(ColorAugSSD RGB 붕괴) 픽스 적용 후 첫 클린 DELIVER 런** — P36/P38/P39-DPC와 달리 오염 없는 상태로 진행됨. configs `jarvis-muses_rgbel_P39_1_rank.yaml`/`hpca100-deliver_rgbdel_P39_1_rank.yaml`. 합성 스모크 PASS(γ/MLP grad 흐름, eval 결정론, linear 모드 하위호환). 상세 [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md) / [models/arch-evolution.md](../models/arch-evolution.md) P39.1 |
| **2** | **P40 RCA-Fusion 본학습** (DELIVER + MUSES) | P39.1과 동일 자원, 완주 후 이어서 | **P39.1 rank 게이트(lidar effective-rank ≥15) 통과 확인 후** 투입 — rank가 죽은 채면 C-3 lidar readout이 헛돎 | **구현 완료(develop ac5c7fe)** — P39.1 위에 Reliability-Conditioned Attenuation 추가. C-1: lidar 리턴 유효성(입력 유도 내부 신호) → 가드/분석. C-2: 자기추정 rel(img) 배치 하위 분위(30%) 샘플의 img feature soft 감쇠(α 0.1~0.5, hard-zero 금지, p_max 0.5, warmup 20ep, 학습 전용). C-3: 감쇠 샘플 한정 lidar readout 보조 CE(w 0.5, gradient 출구). **판정 게이트(사전 등록)** = MUSES test ≥79.025 & fog_night ≥74(P38 복원 우선) · DELIVER = P36 fair + thin-class 유지. configs `jarvis-muses_rgbel_P40_rca.yaml`/`hpca100-deliver_rgbdel_P40_rca.yaml`/`yeon-deliver_rgbdel_P40_rca_smoke.yaml`(스모크). 합성 스모크 PASS(RCA pick 발생, C-1 가드 동작, 손실 유한, grad 흐름). 상세 [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md) / [models/arch-evolution.md](../models/arch-evolution.md) P40 |
| **3** | **P39-4모달 radar-fix 재실험** | hpca100/jarvis 4모달 슬롯 | P39.1/P40 완료 후 | ISSUE-025(MUSES radar 디코딩 버그) 픽스 후 radar 기여 재측정 — P34 4모달 test −0.72 판정이 broken-radar 상태 기반이라 보류 중 |
| **4** | **4모달 구조 버그 수정 + 재도전** | 4~8 GPU | diag_C 판정 직후 | diag_C가 "구조" 판정 시. **DGFusion CLRE 센서 패리티 = 공정 비교 필수 조건.** 수정되면 Arm A의 검증된 투영을 얹어 공짜 이득 |
| **5** | **동일 박스 대조군** (3모달+기존투영, bengio) | 4 GPU | GPU 여유 시 | 현 대조군은 **B200 수치**인데 Arm A는 bengio → **cross-box 교란**. 같은 박스 대조군이 있어야 "DGF 투영=중립" 판정이 단단해짐 |
| **6** | **시드 복제 (2~3 seed)** | 4 GPU × N | GPU 여유 시 | 세션 내내 "+0.13/+0.10은 노이즈"라 말했으나 **분산 데이터 없음**. ablation 표에 ± 를 달 수 있음 |
| **7** | **P36_physaug ep64 이어달리기** | 4~8 GPU | yeon 완주 후(user 지시) | test가 `ep56 55.60` **상승 중 B200 마감으로 잘림**(P34는 ep140까지 상승). **DELIVER test −0.09를 메울 정당한 경로 후보.** `last_checkpoint`(ep64) NAS 보유 |
| **8** | **TTA-on 실측** (참고용) | 1 GPU × ~7h(4090) | 여유 시 | **헤드라인 사용 불가 확정**(경쟁자 미사용) → ablation 행 전용. 준비물 배치 완료(hinton/jarvis). TTA-off는 G0a가 이미 확보(val 68.20/test 56.64) |
| **9** | **class-transfer 공략** | 미정 | 설계 후 | 분석이 지목한 **지배 원인, 복구 상한 +7.9pt**. 0.09짜리가 아니라 판을 바꾸는 크기 |

## ✅ 완료·판정 (재실행 금지)

| 실험 | 결론 |
|---|---|
| **A/B 격리 (Arm A/B)** | **radar(또는 4모달 구조)가 범인. lidar 재투영·event dilation·eff batch 전부 무죄.** Arm A ep24 best 73.85@ep18 — 대조군(ep10 74.24)에 앞서지 않음 → **DGF 투영 = 중립** |
| **TTA 판정** | **경쟁자 3종 전부 미사용** → 헤드라인 사용 불가. CMNeXt 논문 명시(*"single-scale test strategy"*). 우리 MSF는 **dead config**라 과거 수치 무오염 |
| **투영 정합** | DGFusion 파라미터 재현 완료(공개 PIXEL_MEAN 오라클로 −0.1% 적중). **실제 차이는 lidar뿐**(radar·event 30ms는 이미 동일). **성능 이득 0, 공정성만 확보** |
| **module ablation** | **제안 모듈 전부 ≈0**(ATTN_BIAS=RBMA 간판 포함). gate+calib만 test +0.26. **성능 출처 = DINOv3 백본 + per-modal LoRA** |
| **det 붕괴 진단** | 원인 = **BS1의 gradient 노이즈**(n_pos 1~3), LR 아님. 처방 = 배치↑ + **LR 유지**. warmup 5ep 완주로 검증 |
| **seg-P37a/b (bengio분)** | **사망 확정** — bengio 노드 CUDA 전체 장애(GPU5 HW 고장, 재부팅 후 SSH 미복귀)로 ep1~2에서 종료. jarvis 재기동분(P37a→P37b 체인)이 계보 승계 — 남 세션 소관이라 수치 갱신하지 않음 |

## ⚠️ 사고 기록 (반복 금지)

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
