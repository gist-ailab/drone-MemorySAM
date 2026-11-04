판정은 이 파일이 아니라 `lab-plan verdict DRN-261017-02 --analysis .claude_logs/tasks/DRN-261017-02/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-04. 요청 inbox `20261104-205740-request-1d46ee87`(method). 학습 없음.

## 측정 방법
- 대상 5 ckpt(hpca100, md5): C2D2 768 s827 `epoch122_59.8`(881342c6…) · C2D2 768 s828 `epoch92_59.76`(8bf199bc…) · w/o C2D2 1024 s827 `A2hires epoch180_60.27`(f3509574…) · E1Mc 1024 s827 `epoch114_58.94`(bc83178e…) · E1Mc 1024 s828 `epoch174_58.61`(4dbdbc94…).
- **하네스**: 요청은 val.py 였으나 val.py 에는 MCubeS 분기가 없다(`create_dataset` 이 MULTIAQUA·MUSES·DELIVER 만 받음). 그래서 `tools/qualbank_dump.py`(학습기 평가 경로 그대로: `get_val_augmentation(EVAL.IMAGE_SIZE=1024)` + 동결 로더 `semseg/datasets/mcubes.py` + `semseg.metrics.Metrics`, BS1)로 채점했다. 기동 전 `tools/eval_harness_guard.py --check` = OK(8파일 동결 manifest 일치).
- **split**: MCubeS 로더는 `split='val'` 이면 `list_folder/test.txt`(102장)를 읽는다(mcubes.py:142). 즉 학습기 'val' 수치 = 커뮤니티 test 102장 수치다(CMNeXt 관행, 10-30 user 결정 "MCubeS 는 test 로 ckpt 선택"과 같음). 평가 config 는 각 ckpt 의 학습 config 그대로(`configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed2026082{7,8}{,_A2hires,_A2c2d2_768}.yaml`), 평가 1024.
- 실행: hpca100 A100 GPU1~3, 코드 = develop HEAD archive(`~/SSDb/jemo_maeng/src/dms-analysis-20261104`), 스크립트 `hpca_mcubes_rescore.sh`. 원본: NAS `analysis_logs/mcubes_rescore_20261104/`(모델별 pred·gt PNG 102장, summary.json, class_table.md, 로그, ckpt_md5.txt).

## 사실 — 재채점 결과
| 모델 | 학습 크롭 | 학습기 top1 | 하네스 mIoU | Δ | 장수 |
| --- | --- | ---: | ---: | ---: | ---: |
| C2D2(두 패스 + E1Mc 교사 증류 + 고해상 경로) s827 | 768 | 59.80 | 59.80 | +0.00 | 102 |
| C2D2 s828 | 768 | 59.76 | 59.76 | +0.00 | 102 |
| w/o C2D2(고해상 경로만, 두 패스·증류 없음) s827 | 1024 | 60.27 | 60.27 | +0.00 | 102 |
| E1Mc(4탭 clean, 고해상 경로 없음 = 교사) s827 | 1024 | 58.94 | 58.94 | +0.00 | 102 |
| E1Mc s828 | 1024 | 58.61 | 58.61 | +0.00 | 102 |

- 5개 모두 키 missing 0·unexpected 0, 학습기 기록과 소수 둘째 자리까지 같다(Metrics 가 소수 둘째 자리 반올림).
- 2시드 평균: C2D2 **59.78**, E1Mc **58.78**(58.775). w/o C2D2 는 1시드(s827)뿐.
- C2D2 2시드 평균 − w/o C2D2 s827 = **−0.49**, 같은 시드 s827 끼리 −0.47. C2D2 2시드 평균 − E1Mc 2시드 평균 = **+1.00**.
- 🔴 교란: C2D2 는 768 크롭 학습(24GB 카드 대응, DRN-261027-03), w/o C2D2·E1Mc 는 1024 크롭 학습이다. C2D2 − w/o C2D2 차이에는 레시피 효과와 크롭 효과가 섞여 있다. 학습 길이도 ckpt 선택 epoch 가 다르다(122·92 vs 180).

## 클래스별 IoU (test 102)
| 클래스 | C2D2 s827 | C2D2 s828 | C2D2 평균 | w/o C2D2 s827 | E1Mc 평균 | C2D2평균−w/o | C2D2평균−E1Mc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asphalt | 89.36 | 89.87 | 89.62 | 89.39 | 88.81 | +0.23 | +0.81 |
| concrete | 55.48 | 56.12 | 55.80 | 55.29 | 55.99 | +0.51 | −0.19 |
| metal | 64.62 | 65.58 | 65.10 | 66.01 | 64.56 | −0.91 | +0.54 |
| road_marking | 76.78 | 77.48 | 77.13 | 78.62 | 74.08 | −1.49 | +3.05 |
| fabric | 40.21 | 43.82 | 42.01 | 50.74 | 45.16 | −8.73 | −3.14 |
| glass | 62.35 | 61.34 | 61.84 | 62.72 | 61.72 | −0.88 | +0.12 |
| plaster | 1.13 | 1.90 | 1.51 | 1.75 | 0.19 | −0.24 | +1.33 |
| plastic | 50.48 | 54.30 | 52.39 | 53.19 | 50.53 | −0.80 | +1.86 |
| rubber | 31.18 | 34.59 | 32.89 | 35.18 | 27.18 | −2.30 | +5.70 |
| sand | 69.14 | 60.69 | 64.91 | 60.31 | 61.97 | +4.60 | +2.95 |
| gravel | 77.57 | 66.98 | 72.28 | 64.53 | 67.91 | +7.75 | +4.36 |
| ceramic | 43.61 | 40.98 | 42.30 | 44.55 | 40.21 | −2.25 | +2.08 |
| cobblestone | 74.83 | 77.99 | 76.41 | 73.25 | 73.47 | +3.16 | +2.94 |
| brick | 48.45 | 54.08 | 51.27 | 50.92 | 52.65 | +0.35 | −1.38 |
| grass | 60.93 | 60.86 | 60.90 | 62.69 | 62.72 | −1.79 | −1.83 |
| wood | 50.68 | 50.37 | 50.52 | 54.56 | 53.03 | −4.04 | −2.50 |
| leaf | 78.34 | 76.17 | 77.25 | 78.11 | 77.78 | −0.86 | −0.53 |
| water | 86.59 | 86.03 | 86.31 | 86.82 | 85.30 | −0.51 | +1.01 |
| human | 37.60 | 39.21 | 38.40 | 40.09 | 35.98 | −1.69 | +2.42 |
| sky | 96.59 | 96.75 | 96.67 | 96.75 | 96.21 | −0.08 | +0.46 |

- [사실] C2D2 의 w/o C2D2 대비 이득은 sand(+4.60)·gravel(+7.75)·cobblestone(+3.16) 에 몰리고, fabric(−8.73)·wood(−4.04)·rubber(−2.30)·ceramic(−2.25) 에서 잃는다. sand·gravel 은 C2D2 두 시드 사이에서도 8~11점 차이(69.14/60.69, 77.57/66.98)라 102장에서 시드 분산이 큰 클래스다.
- [미검증] fabric·wood 손실이 768 크롭 때문인지 두 패스 학습 때문인지는 이 표로 가를 수 없다(1024 크롭 C2D2 ckpt 없음).

## 정성 은행 덤프
- C2D2 s827·s828, w/o C2D2 s827, E1Mc s827·s828 의 test 102장 전체 예측(학습 ID PNG, 라벨 해상도)과 GT 가 NAS `analysis_logs/mcubes_rescore_20261104/<모델>/{pred,gt}/` 에 있다(pred 510장 = 5×102 확인). image_id = list-file 이름.

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- C2D2 vs w/o C2D2 를 크롭 교란 없이 보려면 768 크롭 w/o C2D2(A2hires_768, lecun/yeon config 존재) 또는 1024 크롭 C2D2 가 필요하다.
