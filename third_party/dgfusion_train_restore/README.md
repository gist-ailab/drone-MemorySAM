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
- `train_net.py` — DGFusion용 학습 스크립트 복원본 (저장소 루트에 복사해 사용). `DGFUSION_AMP_BF16=1`이면 bf16 autocast(아래 "NaN 대책").
- `dgfusion_training_restore.patch` — `dgfusion/dgfusion.py`·`dgfusion/modeling/criterion.py` 패치 (`git apply`).
- `setup_cafuser_lecun.sh` — CAFuser 학습 세팅 (lecun, 기존 conda env `dgfusion` 재사용).
- `cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml` — ⚠️ 3 GPU 제약 파생 config
  (bs6·LR 0.75e-4·266,667 iter = 총 샘플 수 동일). 공식 bs8 4 GPU와의 편차를 결과 보고에 반드시 명시.
- `detectron2_nonfinite_skip.patch` — detectron2 `engine/train_loop.py` 안전장치 패치(아래 "NaN 대책" 절의 1차 대책). detectron2
  클론 루트에서 `patch -p1 < detectron2_nonfinite_skip.patch`.
- `dgfusion_wait_and_resume.sh` — 빈 GPU 4장이 20초 간격 두 번 연속 확인되면 `--resume`으로 재개하는 대기 스크립트
  (다른 세션이 GPU를 곧바로 채우는 서버에서 30분 간격 점검이 매번 빈 틈을 놓쳐서 만든 것).
- `act_probe.py` — 체크포인트별 모듈 최대 |활성값| 측정(fp16 넘침 진단용, `python act_probe.py <ckpt> ...`).
- `bf16_smoke.py` — 80k 체크포인트로 bf16·fp16 autocast 순전파가 정상인지 확인.
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

## NaN 대책 (2026-09-13, 원인 확정·정정판)

**증상**: 공식 설정(fp16 AMP) 그대로 학습하면 86.5k~89.9k 구간에서 NaN으로 학습이 종료된다. 4회 재현됐다 —
jarvis에서 처음부터 학습 시 87,370, 80k 재개 후 87,842, 80k 재개 후 86,572, yeon에서 80k 재개 후 89,855
(jarvis의 서로 다른 크래시는 3건이다. 같은 재개 기록이 `log.txt`와 `dgfusion_resume_80k.log`에 중복 기록돼 있어
더 많이 세기 쉽다). NaN 시점에 `loss_ce`·`loss_contrastive`가 NaN, 매처가 "Matrix contains all NaN values!"를
출력한다(= 순전파 출력 자체가 NaN). `loss_depth`·`loss_condition`은 정상값.

**원인 (측정으로 확정)**: OneFormer의 **`task_mlp` 출력이 학습 내내 한 방향으로 커져 fp16 최대값 65,504를 넘는다.**
`act_probe.py`로 val 8장 fp32 순전파의 모듈별 최대 |활성값|을 체크포인트마다 잰 결과:

| ckpt | 50k | 60k | 70k | 80k | 90k (bf16 학습) |
|---|---|---|---|---|---|
| `task_mlp` 출력 최대 | 52,605 | 57,533 | 60,564 | **62,918** | **66,372** |
| fp16 최대 대비 | 80.3% | 87.8% | 92.5% | **96.1%** | **101.3%** |

90k 체크포인트는 이미 fp16 한계를 넘었다 — fp16 그대로였다면 90k 이후는 학습 불가였다는 직접 증거다.

10k당 약 2,400~3,000씩 늘어 88k~89k에서 한계를 넘는다(학습 이미지는 증강 때문에 조금 일찍 닿는다). 백본 등 다른
모듈은 최대 1.5k 수준이고, 가장 큰 파라미터도 |57|로 가중치 자체는 멀쩡하다. `task_mlp`의 입력은 과제 설명
문장("the task is semantic")인데 DELIVER는 semantic 과제만 쓰므로 **모든 배치에서 입력이 같다** → 한 번 넘기
시작하면 이후 모든 배치가 NaN이다. yeon 재개에서 89,855부터 **20회 연속** NaN이 난 것이 이것을 보여 준다.

**1차 대책(불충분, 기록용)**: `detectron2_nonfinite_skip.patch` — detectron2 `write_metrics`가 NaN을 만나면 학습을
끝내지 않고 기록만 건너뛰게 한 것(가중치 갱신은 GradScaler가 건너뜀, 20회 연속이면 종료). "드문 배치 하나가
넘친다"는 1차 진단에 맞춘 대책이었으나, 원인이 입력과 무관한 상시 넘침이라 이것만으로는 이어갈 수 없었다
(위 20회 연속 사례). 안전장치로는 여전히 유효해 적용해 둔다. 또한 1차 진단의 "BatchNorm이 없다"는 판단도
틀렸다 — `dgfusion/modeling/depth_feature_fusion/concat.py`에 `nn.BatchNorm2d`가 있다(학습 모드에서는 배치
통계를 쓰므로 이번 NaN의 원인은 아니다).

**최종 대책**: **bf16 혼합정밀**. `train_net.py`에서 환경변수 `DGFUSION_AMP_BF16=1`이면 detectron2 `AMPTrainer`의
autocast 정밀도를 `torch.bfloat16`으로 바꾼다(기본값은 공식 fp16 그대로). bf16은 fp32와 같은 지수 범위라
넘침이 사라지고 속도·메모리는 fp16과 같다(RTX 3090에서 약 0.97 s/iter, 18.5 GiB). `bf16_smoke.py`로 80k
체크포인트의 bf16 순전파(MSDeformAttn 커스텀 커널 포함)가 정상이고 출력이 유한함을 확인했다.

```bash
DGFUSION_AMP_BF16=1 bash dgfusion_wait_and_resume.sh   # 또는 train_net.py 직접 호출 앞에 같은 환경변수
```

**검증 (2026-09-13 13:17)**: bf16 재개가 yeon fp16 사망 지점 89,855와 90k를 NaN 0건으로 통과, 90k 체크포인트 저장·val 65.50.

**추론 주의**: `task_mlp`가 fp16 한계를 넘었으므로 **90k 이후 체크포인트는 fp16(AMP) 추론 시 NaN**이다. 공식 test
명령(`test_net.py --eval-only`)은 fp32로 돌아 영향이 없지만, 가속을 위해 fp16 추론을 켜면 깨진다. bf16 또는 fp32로만 추론할 것.

**보고 의무**: 80k 이후 구간은 **fp16이 아니라 bf16으로 학습**했다는 점(공식 레시피와의 차이)을 결과와 함께 적는다.
bf16은 fp16보다 가수 비트가 적어(8 vs 10) 활성값 정밀도가 낮다. 공개 가중치(저자 학습)가 같은 문제를 겪었는지는
알 수 없다(학습 코드 미공개). CAFuser도 같은 `task_mlp`를 쓰지만 **fp16으로 191k(lecun, batch 6 보정판)까지 NaN 없이 진행**됐다 — 같은 모듈이 DGFusion에서만 빠르게 커진다는 뜻이며, depth 보조 과제 등 DGFusion 고유 구조와의 관계는 CAFuser 완주 후 `act_probe.py`로 비교할 한계분석 항목이다.

## 프로토콜 주의

- 원 논문 계열은 BestCheckpointer 없이 **final-iter 체크포인트가 정식**이다. 재현 비교 시 final-iter와
  val-best를 병기하되 유리한 쪽만 고르지 말 것 (memory `seg-report-sota-gap` 규약).
- 데이터 심링크는 반드시 로컬 디스크 사본으로 (sshfs `/ailab_mat2` 금지 — I/O hang 이력).
- 복원 학습 블록은 CAFuser와 구조 동일 + depth 배선만 추가이므로, 공개 가중치와의 정합성 검증은
  학습 완료 후 공개 수치(val 66.51/test 56.71) 재현 여부로 판단한다.
