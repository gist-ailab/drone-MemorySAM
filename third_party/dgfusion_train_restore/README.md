# DGFusion / CAFuser DELIVER 학습 재현 킷 (2026-09-08)

DGFusion 공개 저장소(timbroed/DGFusion, RA-L 2026)는 **학습 코드를 의도적으로 뺀 채** 공개되어 있다
("we do not include the training code in this project"). 이 폴더는 CAFuser(timbroed/CAFuser, 학습 코드
완전 공개)를 대조해서 빠진 조각을 복원한 산출물이다. 2026-09-08에 jarvis(DGFusion)·lecun(CAFuser)에서
실제 학습 기동까지 검증됐다 (registry.md의 `dgfusion_swin_tiny_bs8_200k_deliver_clde` /
`cafuser_swin_tiny_bs6_267k_deliver_clde_lecun` 행 참조).

## 공개판에서 빠져 있던 것 (전부 이 킷이 복원)

| 결손 | 복원물 |
|---|---|
| `train_net.py` 자체가 없음 (`test_net.py`만 제공) | `train_net.py` — CAFuser train_net.py에 test_net.py의 dgfusion import·DepthEvaluator 배선을 병합 |
| 모델 forward의 학습 분기가 예외로 차단 (`dgfusion.py`: "Training code ... not provided") | `dgfusion_training_restore.patch` — CAFuser cafuser.py의 학습 블록 + depth GT 배선(gt_depth/image/sem_seg를 모델 패딩 크기로 맞춰 targets에 주입; 패딩 값은 각 손실의 무시값 0/−1/255) |
| criterion.py의 학습 전용 유틸 4종 미정의 (`calculate_uncertainty`, `dist_collect`, `sigmoid_ce_loss_jit`, `dice_loss_jit`) | 같은 patch — CAFuser와 동일하게 `oneformer.modeling.criterion`에서 import + jit 스크립트 |

손실 가중치·SOLVER 설정(bs8, LR 1e-4, 200k iter, AdamW, poly)·depth 손실 자체(SetCriterion.loss_depth)는
공개 config/criterion에 온전히 남아 있어서 복원 없이 그대로 쓴다. **DELIVER config에서는 DGFusion과
CAFuser의 LR이 1e-4로 동일**하다 (과거 기록의 "1.8×" 교란은 MUSES config 이야기).

## 파일

- `setup_dgfusion_train_env.sh` — 환경 구축 통합본. 공식 INSTALL.md + 실전에서 잡은 빌드 픽스 4종
  (빌드 격리 해제, gcc-11 강제, setuptools 59.5.0 선치, natten 인덱스 SSL 만료 우회) + 복원물 적용까지.
- `train_net.py` — DGFusion용 학습 스크립트 복원본 (저장소 루트에 복사해 사용).
- `dgfusion_training_restore.patch` — `dgfusion/dgfusion.py`·`dgfusion/modeling/criterion.py` 패치 (`git apply`).
- `setup_cafuser_lecun.sh` — CAFuser 학습 세팅 (lecun, 기존 conda env `dgfusion` 재사용).
- `cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml` — ⚠️ 3 GPU 제약 파생 config
  (bs6·LR 0.75e-4·266,667 iter = 총 샘플 수 동일). 공식 bs8 4 GPU와의 편차를 결과 보고에 반드시 명시.
- `detectron2_nonfinite_skip.patch` — detectron2 `engine/train_loop.py` 패치(아래 "NaN 대책" 절). detectron2
  클론 루트에서 `patch -p1 < detectron2_nonfinite_skip.patch`.
- `dgfusion_wait_and_resume.sh` — 빈 GPU 4장이 20초 간격 두 번 연속 확인되면 `--resume`으로 재개하는 대기 스크립트
  (다른 세션이 GPU를 곧바로 채우는 서버에서 30분 간격 점검이 매번 빈 틈을 놓쳐서 만든 것).
- `dgfusion_test_sweep.sh` — 저장된 체크포인트들을 공식 README의 test 평가 명령으로 차례로 평가
  (`bash dgfusion_test_sweep.sh <gpu> 0009999 0019999 ...`). val→test 전이 곡선용.

## 실행 (jarvis 검증 커맨드)

```bash
cd /SSDb/jemo_maeng/dgfusion_train
conda activate dgfusion
export PYTHONPATH=$PWD/OneFormer:$PWD
export WANDB_MODE=offline DETECTRON2_DATASETS=$PWD/datasets
CUDA_VISIBLE_DEVICES=1,2,4,5 python train_net.py --dist-url tcp://127.0.0.1:50366 --num-gpus 4 \
    --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
    OUTPUT_DIR output/dgfusion_swin_tiny_bs8_200k_deliver_clde
```

## NaN 대책 (2026-09-13)

**증상**: 공식 설정(fp16 AMP) 그대로 학습하면 80k 이후 NaN으로 학습이 종료된다. jarvis에서 3회 재현됐다 —
처음부터 학습 시 iteration 87,370, 80k 재개 후 87,842, 80k 재개 후 86,572 (jarvis의 모든 로그를 합친 서로 다른
크래시는 이 3건뿐이다. 같은 재개 기록이 `log.txt`와 `dgfusion_resume_80k.log`에 중복 기록돼 있어 4건으로 세기 쉽다).
`SEED: 0`이라 두 번의 80k 재개는 같은 데이터 순서로 시작했는데도 서로 다른 지점에서 죽었으므로, 특정 배치 하나가
원인이라기보다 **80k 이후 모델 상태에서 fp16 순전파가 가끔(약 7천 iteration에 한 번) 넘치는 현상**으로 본다
(GPU 연산의 비결정성 때문에 같은 데이터 순서라도 수치가 조금씩 달라진다). NaN 시점에 loss_ce·loss_contrastive가 NaN이고 매처가
"Matrix contains all NaN values!"를 출력한다(= 순전파 출력 자체가 NaN). loss_depth·loss_condition은 정상값.

**원인 구조**: detectron2 `AMPTrainer.run_step` 순서는 역전파 → `_write_metrics` → `grad_scaler.step`이다.
GradScaler는 기울기가 NaN/inf면 그 스텝의 가중치 갱신을 원래 건너뛰는데, 그 직전의 `write_metrics`가
손실이 유한하지 않다고 `FloatingPointError`로 학습 전체를 종료시킨다.

**대책**: `detectron2_nonfinite_skip.patch` — `write_metrics`가 NaN 손실을 만나면 경고(`[nonfinite-skip]`,
연속 횟수·누적 횟수 포함)를 남기고 기록만 건너뛴다. 가중치 갱신은 GradScaler가 건너뛴다. 진짜 발산이면 NaN이
연속으로 나오므로 **20회 연속이면 종전대로 종료**한다. 모델의 정규화 층은 LayerNorm(Swin)·GroupNorm(픽셀
디코더·depth head)뿐이라(설정의 SyncBN은 쓰이지 않는 ResNet 항목) NaN 순전파가 running 통계를 오염시킬 경로가 없다.
CPU 스모크 테스트 통과(단발 NaN 건너뜀·정상값에서 연속 카운터 초기화·20회 연속에서 종료).

**보고 의무**: 이 패치는 fp16 레시피를 바꾸지 않고 드문 NaN 배치만 건너뛰지만, 공식 코드와의 차이이므로
결과 보고 시 **건너뛴 스텝 수**(로그의 `total_skipped`)를 함께 적는다.

## 프로토콜 주의

- 원 논문 계열은 BestCheckpointer 없이 **final-iter 체크포인트가 정식**이다. 재현 비교 시 final-iter와
  val-best를 병기하되 유리한 쪽만 고르지 말 것 (memory `seg-report-sota-gap` 규약).
- 데이터 심링크는 반드시 로컬 디스크 사본으로 (sshfs `/ailab_mat2` 금지 — I/O hang 이력).
- 복원 학습 블록은 CAFuser와 구조 동일 + depth 배선만 추가이므로, 공개 가중치와의 정합성 검증은
  학습 완료 후 공개 수치(val 66.51/test 56.71) 재현 여부로 판단한다.
