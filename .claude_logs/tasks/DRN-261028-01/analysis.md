판정은 이 파일이 아니라 `lab-plan verdict DRN-261028-01 --analysis .claude_logs/tasks/DRN-261028-01/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-10-29. 요청 inbox `20261028-162457-request-00635845`(method). 학습 없음.

## 방법
- 도구 `tools/probe_hires_degrade_signal.py`(develop caa8408, codex 작성·이 세션 검수, dry-run·4장 스모크 통과).
- 모델: A2(Q2 레시피 + 고해상 경로) 시드821 `epoch40_68.93_top1`(md5 f05b54d4…), config `configs/eval/a2s821_eval1024_yeon.yaml`, legal 1024 입력, lecun GPU2, DELIVER test 1897장 전수(가중치 missing 0·unexpected 0).
- 신호원 2개(센서별):
  - **고해상 s4**: `model.hires.spm.stem` 출력 c1(stride 4, 64채널). 공유 스템이 센서 순서로 호출된다.
  - **블록 6**: 센서별 `encoder` 호출 직후 `last_taps[0]`(stride 16 토큰).
- 토큰 통계 3종: 채널 L2 노름(norm), 고주파 에너지(F − 3×3 평균)²의 채널 평균(hf), 채널 분산(cvar).
- 이미지 점수 = 통계의 토큰 평균·표준편차(6개). 토큰 단위 = 이미지당 무작위 256 토큰.
- 파티션(`common.parse_condition_case`): 열화 5종 = 센서 고장 케이스 motionblur 300 · overexposure 100 · underexposure 100 · lidarjitter 99 · eventlowres 100장 vs clean(case none) 1198장. 참고 2종 = night 240 · fog 239 vs sun 240장(모두 case none).
- AUROC 는 동점 평균 순위 Mann-Whitney 이다. 통계의 방향을 모르므로 양방향 값 max(a, 1−a)를 쓴다. 표의 이미지 단위 값은 **6개 점수 중 최대값**이라 선택 편향만큼 낙관적이다.
- 원본: NAS `analysis_logs/hires_probe_a2s821_20261029/`(auroc.csv·image_scores.csv·token_samples.npz·run.json, md5 일치), 사본 `.claude_logs/experiments/analysis/DRN-261028-01/`.

## 사전 기준 대비 사실
- 통과 조건: 고해상이 실제 열화 5파티션 중 ≥3 에서 AUROC ≥ 0.8 **그리고** 블록 6 보다 ≥ 0.15 높음.
- 반증 조건: 고해상 통계도 ≤ 0.76 이면 토큰 적응 LoRA 축 종결.

### 해당 센서 기준 이미지 단위 AUROC(양방향 최대, 괄호 = 점수)
| 파티션(해당 센서) | 고해상 s4 | 블록 6 | 차(고해상 − 블록 6) | Q1d 기준선(학습형 품질 헤드) |
| --- | --- | --- | ---: | ---: |
| motionblur (RGB) | 0.655 (hf_mean) | 0.617 (cvar_std) | +0.04 | 0.46 |
| overexposure (RGB) | 0.788 (cvar_mean) | 0.756 (norm_mean) | +0.03 | 0.76 |
| underexposure (RGB) | **0.942** (cvar_std) | **0.963** (cvar_std) | −0.02 | 0.76 |
| lidarjitter (LiDAR) | 0.624 (cvar_mean) | 0.586 (hf_mean) | +0.04 | — |
| eventlowres (Event) | 0.612 (norm_mean) | 0.749 (norm_std) | −0.14 | — |
| 참고 night (RGB) | 0.962 (cvar_mean) | 0.996 (norm_std) | −0.03 | 0.28 |
| 참고 fog (RGB) | 0.845 (norm_std) | 0.992 (cvar_std) | −0.15 | 0.44 |

- 고해상 ≥ 0.8 인 실제 열화 파티션: **1/5**(underexposure). 기준 ≥3 **미달**.
- 고해상이 블록 6 보다 ≥ 0.15 높은 파티션: **0/5**(최대 +0.04). 기준 **미달**.
- 반증 조건(고해상 ≤ 0.76 전부): 이미지 단위로는 해당 안 됨(overexposure 0.788, underexposure 0.942). **토큰 단위로는 해당됨**: 고해상의 해당 센서 토큰 AUROC 는 motionblur 0.519 · overexposure 0.623 · underexposure 0.698 · lidarjitter 0.512 · eventlowres 0.569 로 모두 ≤ 0.70 이다(블록 6 토큰 단위: 0.533 · 0.622 · 0.779 · 0.506 · 0.572).
- 다른 센서 열의 값(예: RGB 열화에서 depth·event·lidar)은 대부분 0.50~0.64 다.

## 원인 해석 (검증 여부 표시)
- [사실] 학습 없는 토큰 통계만으로도 RGB 의 밝기 계열 열화(저노출·야간·안개)는 이미지 단위로 잘 갈린다(0.85~1.0). 이는 고해상 s4 와 블록 6 에서 비슷하거나 블록 6 이 더 높다. Q1d 의 학습형 헤드(0.28~0.76)보다 높은 것은 신호원의 차이가 아니라 측정 방식의 차이다(Q1d 는 합성 열화로 학습한 헤드를 실제 열화에 옮긴 것).
- [사실] 모션블러·LiDAR 지터·Event 저해상은 두 신호원 모두 0.59~0.75 에 머문다. 고해상 경로가 이 세 열화를 특별히 더 잘 보지는 않는다.
- [사실] 토큰 단위 AUROC 는 모두 0.78 이하다. "토큰마다 열화를 알아보는 신호" 는 두 신호원 어느 쪽에도 약하다. 이미지 단위 분리는 토큰 분포의 평균·산포 차이에서 온다.
- [미검증] 따라서 P56-C(상태 조건부 LoRA 라우터)의 붕괴를 "블록 6 에 신호가 없어서" 로 설명하기는 어렵다. 이미지 단위 신호는 블록 6 에도 있었다. 토큰 단위로 라우팅하는 설계라면 두 신호원 모두 입력으로 약하다.
- 주의: 이미지 단위 값은 6개 점수 중 최대값이라 낙관적이다. 단일 시드(821)·단일 ckpt 다.

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- 없음(측정 완료). 필요하면 A2 시드902·903 으로 같은 프로브를 반복해 시드 의존을 확인할 수 있다(GPU 약 45분/시드, amphibious-demo 에서는 약 20분 예상).
