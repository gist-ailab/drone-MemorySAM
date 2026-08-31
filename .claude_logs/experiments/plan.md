---
created: 2026-07-16
updated: 2026-07-21 (ISSUE-025 MUSES radar 디코딩 버그 수정 반영 → 대기열 #3 "P39-4모달 radar-fix 재실험" 신설 + 사고 기록 1줄, 이하 대기열 번호 +1) ; ISSUE-026 ColorAugSSD RGB 붕괴 버그 반영 → hpca100 P39-DPC resume 오염 표기 + 사고 기록 1줄 + 대기열 #1 클린런 표기 ; 2026-07-23 대기열 #1 "P39.1 Rank 수리" MUSES-jarvis 분기 착수 → 실행중 표에 행 추가(jarvis 2,3,4,5, 기동검증 통과) ; 2026-07-26 P43-MUSES 완주(val 82.51@ep156, seed2 미돌파) → 대기열 #11 P44-BMR을 hpca100 GPU2,3에 착수(develop 678c493, 기동검증 통과); 2026-07-27 seed4 완주(81.92)→해방 GPU에 첫 4-modal(P39.1+radar) 착수(yeon 0,1,5, 305b030); seed2 분석 완료(trunk+2~7·VICReg lidar rank 78~100 검증); 2026-07-27 jarvis 리부트(드라이버 595.84 복구)→DELIVER 2실험 착수(P39.1-rank GPU0-3 / P44-BMR GPU4-7, BS1, develop be2603c) — DELIVER 첫 캠페인 실험; 2026-07-27 4-modal ep2 eval OOM→EVAL BS1+expandable_segments 수정 재기동(9f199be), ep4 eval 통과 확인; 2026-07-28 P44-MUSES 완주(80.71)→해방 A100에 2번째 4-modal(P44-BMR+radar) 착수(hpca100 0-3, 1cf1e66, BS1 OOM수정); 2026-07-28 hpca100 4모달 HF 백본 이중고장(offline=RANDOM INIT/online=hang) 확진 → RELIADINO_LOCAL_BACKBONE env fix(encoder.py 697a10a) → P39.1+radar seed2 클린 기동(ep2 47.61); 2026-07-28 seed3 완주(81.89@204, 5-seed variance 완결) → P44-DELIVER seed2 yeon6,7 수동기동; P44-MUSES(80.71) test staging+분석 lecun; 2026-07-28 P46-CTR 제안 등재(DELIVER SOTA class-transfer, 내부신호 RCS+MIC+prototype) ; 2026-08-03 P46 C3-only λ0.2 DELIVER 완주(200/200, test-best 57.05@ep108) = **DELIVER test SOTA 돌파 확정**(DGFusion 56.71 대비 +0.34, @768 동일 프로토콜) → λ 스윕 상단탐색 λ0.3을 jarvis GPU4-7(회수됨)에 착수, 기동검증 PASS ; 2026-08-03 λ0.2 SOTA 재현성 검증을 위해 seed2를 jarvis GPU1-3(4090×3, GPU0=user 예약)에 착수, config `jarvis-deliver_rgbdel_P46_ctr_c3only_lam02_seed2.yaml`(develop b925c90), 기동검증 PASS ; 2026-08-04 **정정**: 57.05는 test-best 체크포인트 값으로 규약상 무효 확인됨 — legal 재계산(val-best/final-iter) 결과 최고 test 55.62~55.69, DGFusion 56.71 대비 −1.0로 **SOTA 미달**(base 대비 실제 이득은 test +1.35~1.74/val +0.97로 견고, λ 최적 0.05~0.2 평탄). 상세 [experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md](analysis/2026-08-03-p46-c3only-lambda-sweep.md) ; 2026-08-06 MUSES val PQ 첫 측정(P47-MUB D-1 ep172, native, tools/eval_pq.py b6d3da0) → things PQ 22.87 ≤ 30 = P48(쿼리 경로 인스턴스 감독) 사전등록 게이트 미달 → **설계 폐기** (analysis/2026-08-06-pq-first-measurement-p48-gate.md) ; 2026-08-08 대기열·예약표 청소(완주 4건 제거, bengio 잔재 제거, CEA 프로브 등재) ; 2026-08-18 "실행 중" 표 청소(3~4주치 완주 런 박제 제거, 실상태 2런만 유지) + 대기열 #13 spatial-modality oracle 등재
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

## 🔬 실행 중 (2026-08-31 실상태 동기화)

> ⚠️ **이 표는 "지금 도는 것"만 담는다.** 완주·종결된 런은 registry/analysis로 즉시 이동.

| 실험 | 서버/GPU | EPOCHS | ETA | 목적 |
|---|---|---|---|---|
| **P47-2 UniBal** (MUSES 4모달) | hpca100 GPU2,3 (A100×2) | 300 | ~2.9일 (08-31 기동, 로그디렉토리 버그 픽스 후 재기동판) | modality-laziness 처방(모달별 aux CE, 학습전용·추론불변) — **4모달 역전 유일 레버**. 게이트: 4모달 val ≥82.62(3모달 역전) / test ≥79.788 / drop-radar ≥+0.5 / ep30 kill −1.0 |

**직전 완결(08-31)**: P50 파인튠 게이트 통과(+0.74, H22✓) · N2 믹서 판정(mean 55.45, H21✗) · N6 재선택 5/5(54.39±0.76) · MCubeS 3-seed(58.07±0.49) · MUSES 시드 3점(spread 0.66) · P51 페어2 완주(각주) — 전부 analysis/registry 반영됨.

## 📋 대기열 (우선순위 순) — 2026-08-24 전면 재설계 (논문-가치 필터)

> **재설계 기준**: ①논문(accept) 기여 — A(P51 확장)·B(진단-프레임워크) 어느 분기에서도 쓰이는가 ②24GB(yeon 3090/jarvis 4090)에서 도는가 ③원장 반증 경로가 아닌가. 옛 대기열 대부분은 계보 사망·중복으로 종결 처리(하단 🗑).

### 🔵 진행 중 (3트랙)
| 실험 | 자원 | 다음 이벤트 |
|---|---|---|
| ~~P51-CMLC 페어1~~ (#16) | — | 🔴 **판정 종결(2026-08-26): H19 반증** — legal Δ(on−off) **−0.82**(off 55.40 > on 54.58), per-condition 악조건일수록 유해(fog −2.03·night −0.58, 유일우세 sun +0.26) → **F 기각 + B 재프레이밍 확정**. jarvis 페어2(5,6)는 완주 시 재현성 각주만. 판정 [analysis/2026-08-26-p51-cmlc-verdict-h19.md](analysis/2026-08-26-p51-cmlc-verdict-h19.md) |
| P50-MAP finetune (seed821 매칭) (#11) | yeon 1,2 | 완주 → Δ vs 53.57, 게이트 ≥+0.5 |
| 시드 n=5 (#3) | jarvis (822 마무리) | 완주 → n=5 mean±std 확정 → jarvis GPU 해방 |

### 🎯 신규 대기열 (논문-가치 순, 여유 GPU 투입 대상)
| # | 실험 | 자원 적합 | 논문 가치 (A/B 분기별) | 상태 |
|---|---|---|---|---|
| ~~N1~~ | ~~MUSES 시드 분산 ×2~~ | — | ✅ **완결(2026-08-27)** | 공식 val 3점 {82.13, 81.79(s824), 81.47(s825)} spread **0.66** = MUSES val 시드 안정 확정 → test 단일제출 방어 근거 | registry 참조 |
| ~~N2~~ | ~~MLE-SAM 평균융합 baseline~~ | — | ✅ **완료·판정(2026-08-31, H21)** | legal test **55.45** — gated-MLP(54.2~55.4)와 동급 이상 = **우리 트렁크 우위 주장 철회**, 믹서 3점 완성(mean≈gated-MLP>xattn). 소거 논지 완결 재료 | 판정 [analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md](analysis/2026-08-31-p50-gate-pass-n2-mixer-verdict.md) |
| **N3** | **C3 진단-구동 검출기 (분석, 학습 0)** — 기존 ckpt들의 val confusion에서 class-transfer 붕괴 지표(비대각 집중도) 정량화 → C3 on/off 효과와 상관 검증 (DELIVER 붕괴有/MUSES 無) | **GPU ~0**(캐시 confusion 재집계, 필요시 1 GPU eval) | **B 헤드라인 기둥·A여도 통일 서사 필수** — "벤치별 C3 on/off"를 원칙적 자동설정으로 전환하는 근거. 검출기가 두 벤치의 경험적 C3 효과와 일치하면 통일 아키텍처 주장 성립 | 🟡 분석 설계 = discussion 세션 직접 — **즉시 가능** |
| ~~N4~~ | ~~MCubeS 이식 파일럿~~ | — | ✅ **완결 + 3-seed(2026-08-31)** | {57.93, 57.67, 58.62} = **mean 58.07±0.49, published 최고(54.65) +3.42 / min +3.02 = MCubeS 1등 통계 확보**. N4b(C3-on)는 dose-response 예측 2/2 적중(H20✓) | [analysis/2026-08-25-n4-mcubes-first-entry-verdict.md] · [analysis/2026-08-27-n4b-dose-response-confirmed.md] |
| ~~N4b~~ | ~~MCubeS C3-on 페어~~ | — | ✅ **완료·판정(2026-08-27): 사전등록 예측 2/2 적중** | rubber **+9.76**(18.80→28.56, published 대역 복귀) + overall Δ−0.10(범위 내) → **dose-response 3점 성립(H20 ✓)** — 진단-구동 프레임워크 예측력 실증, 논문 헤드라인 기둥. 판정 [analysis/2026-08-27-n4b-dose-response-confirmed.md](analysis/2026-08-27-n4b-dose-response-confirmed.md) |
| ~~N6~~ | ~~DELIVER legal-val ckpt 재선택~~ | — | ✅ **완결(2026-08-31)** | 5/5: mean **53.82→54.39±0.76** (816 +2.21·821 +0.63, 선택 아티팩트 2/5런 실재). **최고 단일런 = seed816 55.29**(base 아님 — outlier 서사 완전 해소). 이후 모든 legal 수치는 하네스 가드 `--check` 필수 | registry·current.md 반영 |
| **N7** | **VICReg-off 격리 토글 1런** (seed821 매칭, VICReg만 off) — 컴포넌트 기여표 마지막 미지 행 | yeon 2장 ×~15h | 🔵 **착수(2026-08-31, user 승인)** — config `yeon-...seed20260821_vicregoff.yaml`(MODEL.P39.VICREG.ENABLE만 off, diff검증), yeon GPU0,1 200ep, 부팅검증 PASS(SEED 일치·카운트 일치·RANDOM INIT 0). ETA ~15h — N2(H21)로 트렁크-타입 기여가 죽어서, 우리 설계 기여 중 VICReg(rank 복원)의 단독 격리가 필요해짐. MUSES +0.76이 트렁크+VICReg 묶음이었음 | 게이트: Δ(VICReg-on − off) — 격리 수치가 곧 논문 ablation 행 |
| **N8** | **P50 런 legal-val 재선택** (eval-only, N6 프로토콜 대칭 적용) | 1 GPU ×~2h | 🔵 **진행 중(2026-08-31)** — n8_batch_eval.sh(N6 패턴, **하네스가드 --check 첫 스텝 내장**), P50 런 ckpt 6개 legal VAL 스윕, yeon GPU2 — P50 최종치 확정(현 54.95는 trainer-top1 기준, 재선택 시 상승 가능성만 있음) | 완료 시 H22 수치 최종화 |
| **N9** | **정성 증명 패키지** — ①VICReg 전후 유효랭크 스펙트럼 ②C3 전후 confusion(RailTrack·rubber) ③per-class 예측맵 패널 | 1 GPU 간헐 (기존 seg-analysis 도구) | 🟡 **스코프 설계 중** — 대상 ckpt가 3서버 분산(jarvis/yeon/hpca100)이라 실행 위치 설계 후 착수(모니터링 세션) — user 요구 "정량+정성 개선 증명"의 정성 절반. 논문 그림 직결 | 산출 = analysis_logs/ + 논문 figure 후보 |
| **P52** | **RxDINO 개정(2026-08-31 user 승인): 단일-config 자기-적응 처방** — C3-adaptive(per-class λ=f(온라인 붕괴지표)) + UniBal-adaptive(per-modal λ_u=f(laziness지표)) + P50 init. 벤치별 config 차이 0, 차이는 λ 궤적으로 창발 | 3벤치 × 단일 config × 시드2+ | 🔵 **컨트롤러 구현 위임 중(GLM)** → 검수·스모크 → UniBal 고정런 완주(~2일, G2 기준 확정) 후 본런 | **게이트 사전등록**: G1/G2/G3 = 각 벤치 최선 고정팔 −0.3 이내(단일 config) + **G4 창발**(λ 궤적이 RailTrack↑/MUSES≈0/rubber↑를 스스로 재현). 실패 시 1회 수정 재시도 후 철회·정직 보고. 개정문서 [decisions/2026-08-31-p52-rxdino-adaptive-amendment.md](../decisions/2026-08-31-p52-rxdino-adaptive-amendment.md) |
| **P50-EXT** | P50 사전학습 확장(코퍼스/에폭 스케일업) — 제안서 "통과 시 확장" 조항 | 대규모(수일) | ⏸ 등록 + 슬롯 후보 = hpca100 0,1(현재 타유저 점유, 해방 감시) — 🟢 **user 승인(2026-08-31, 디스커션 B)** — **Phase1 설계 확정(2026-08-31)**: 편광 = 새 모델 0 — **기존 depth gradient→normal→Fresnel(n=1.5) 유도**(비용 2배→1배), NIR = luminance+excess-green 결정론. 출력 = MCubeS 로더 원자규격(aolp_sin/cos·dolp npy + nir png, 로더 실측 스펙 기반). **코퍼스 = 500k, 사전학습 = 등-스텝(500k×12ep ≈ 200k×30ep = 6M샘플)로 다양성 효과 격리**, 통과 시 에폭 확장. **생성기 병합 완료(2026-08-31, 검수 PASS)** → **(a) 200k 증분 완주·검증(2026-08-31)** — 4모달×200k 각 전량, 기존 파일 무변경(mtime), meta 병합 정상, uint8 PNG 포맷. **✅ 500k 코퍼스 완성(2026-09-01)** — (b) 300k×8모달 전량(실패 0, DA 백엔드 4워커 일관, 254GB/SSDc), 200k@SSDe + 300k@SSDc(Phase2 multi-root). SHA256SUMS 기준선 생성 중. **잔여 = Phase2 사전학습**(hpca100 해방 대기, 등-스텝 500k×12ep, DELIVER-모달 팔 + MCubeS-모달 팔). Phase2 사전학습(등-스텝 500k×12ep)은 hpca100 해방 대기 → Phase2=hpca100 해방 시 본 사전학습. 게이트 사전등록 예정(프로브 +0.74 대비 스케일 이득). 확장 시 **편광/NIR proxy 추가 생성**으로 3벤치 공통 init화가 설계 포인트 | H22 이득(+0.74)이 스케일에 비례하는지 |
| **N5** | TTA-on 실측 (구 #4, 참고용 ablation 행) | yeon/jarvis 1장 ×7h | 낮음 — 헤드라인 불가 확정, ablation 완결성용 | ⏸ 위 소진 후 필러 |

### 🅰️ A100 대기열 (hpca100 P51 완주 후)
| 순위 | 실험 | 근거 |
|---|---|---|
| ① | **P47-2 UniBal** (MUSES 4모달 역전 유일 레버, 구현·스모크 완료) | A100 필요(보조 head 메모리). P51 판정 후 슬롯 |
| ② | P51 후속 (F 추가 재프로브 or MUSES 비회귀) — P51 Δ 판정에 따라 | 게이트 분기 결과 대기 |

### 🗑 종결 처리 (2026-08-24 재설계에서 제거 — 재등재 금지 사유 명시)
- ~~#1 P40 RCA-Fusion~~: 계보 사망(P39.1 게이트 자체가 P46/P51로 승계) + C-2 감쇠는 적응계열(H1~H4 폐쇄) 인접 — **원장 저촉**.
- ~~#2 P39 radar-fix 재실험~~: 이미 충족 — fixed-decoder drop-radar ablation(2026-07-30)이 radar 무익(+0.13)을 재확정. 별도 학습 불요.
- ~~#5 P47-MUB D-1~~: **이미 실행·폐기**(P47-D1 공식 test 78.790, val 과적합, 08-17). D-2는 위 A100 ①로 승계.
- ~~#10 P49-AIR~~: 계열 종결(08-16, 양 벤치 패배·H14). plan 갱신 누락분 정리.
- ~~A100 ③ P49 @1024 대조~~: P49 계열 종결로 무의미.
- ~~ProbeA2-7B 추가 측정~~: H12 폐쇄(7B +0.18)로 종결.

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
