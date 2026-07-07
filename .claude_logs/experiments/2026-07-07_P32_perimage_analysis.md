# P32 (CoRB) — Per-Image Test-Set Analysis @ best weight ep108 (2026-07-07)

> **Best weight**: `test_epoch108_54.79_top1` (Test mIoU 54.79 / Day-Val best ep98 64.12). P32 학습은 계속 진행 중(ep108 시점 스냅샷). ep108은 **P31(54.75)을 처음으로 추월**, P28(55.27)에 −0.48까지 근접.
> **범위**: DELIVER test **전체 1897장 per-image** 시각화 + 수치. 모달 순서 `[img, depth, event, lidar]`.
> **도구**: `tools/viz_features_full.py`(신규) — 이미지별 6행 패널 + per-image CSV, corroboration ON/OFF 효과 측정.
> **산출물**: `/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/` — `panels/`(1897장) · `per_image_ALL.csv`(1897행) · `stats.json` · `per_class_matrix.md`.

---

## 0. 헤드라인 (3줄)
1. **신호는 맞고 라우팅은 실패** — corroboration 신호가 event/LiDAR 신뢰도를 잘 매기지만(무학습 반전), 이것이 **출력 결정을 거의 못 바꾼다**: 1897장 중 **1700장(89.6%) mIoU 무변화**, 픽셀 flip **0.046%**, net ΔmIoU **−0.013**.
2. **융합이 competence를 무시** — depth 단독 competence 43.7로 압도적(img 26.2, event 16.9, lidar 15.3)인데 **UAMM 가중치는 균일**(0.27/0.27/0.23/0.23) → **misallocation 51.6%**(최적 모달=depth인데 융합 top-weight=img).
3. **test 죽은 클래스는 용량한계가 아니라 도메인 전이 붕괴** — Wall/Bridge/Water/TrafficLight는 **val에선 살아있으나**(train log: Wall 56·Bridge 46·Water 33·TrafficLight 79) **test에선 붕괴**(0.9/0.0/0.0/15.4). frozen-backbone ceiling이 아니라 day→night/test class-transfer 문제.

---

## 1. per-image / per-condition 성능
- **per-image 평균 mIoU = 51.73** (dataset-level aggregate 54.79와 차이 = per-image 평균은 소수 클래스만 있는 프레임에서 페널티. 둘 다 유효, 비교는 조건 간 상대값으로).
- **per-condition mIoU**: rain 53.65 · fog 53.58 · cloud 52.15 · sun 50.31 · **night 48.98(최약)**. spread = **4.67**.
  - 갭이 작다(4.67) = 성능 저하의 주범은 **도메인시프트가 아니라 per-class transfer**(P28/P31 Mode B와 동일). night는 thin/geometry 클래스에서 추가 붕괴.

## 2. per-class × per-condition IoU (worst→best 발췌; 전체는 `per_class_matrix.md`)

| class | all | cloud | fog | night | rain | sun | 판정 |
|---|---|---|---|---|---|---|---|
| Bridge | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | **완전사망** |
| Water | 0.02 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | **완전사망** |
| Wall | 0.88 | 0.95 | 0.37 | 1.54 | 0.68 | 0.74 | **완전사망**(val 56) |
| RailTrack | 3.5 | 4.05 | 1.76 | 3.4 | 5.25 | 2.53 | 거의사망 |
| Dynamic | 3.84 | 4.2 | 2.87 | 4.94 | 4.6 | 2.61 | 거의사망 |
| Ground | 6.35 | 6.84 | 3.39 | 8.0 | 6.47 | 7.03 | 거의사망 |
| Other | 7.2 | 4.36 | 8.66 | 7.17 | 6.65 | 9.15 | 거의사망 |
| TrafficLight | 15.4 | 16.2 | 16.0 | 16.0 | 16.6 | 12.2 | 약함(val 79) |
| Static | 17.1 | 14.1 | 18.6 | 15.4 | 18.5 | 19.3 | 약함 |
| TwoWheeler | 18.2 | 14.9 | 24.6 | 15.0 | 21.7 | 15.3 | 약함·night민감 |
| Fence | 21.2 | 20.5 | 20.8 | 16.2 | 23.9 | 24.9 | 약함·night민감 |
| Bus | 24.7 | 19.3 | 18.1 | 20.4 | 34.0 | 31.8 | 조건민감 |
| Truck | 46.7 | 51.4 | 52.2 | 36.4 | 49.8 | 44.1 | night −16 |
| Pedestrian | 53.0 | 57.5 | 55.5 | 49.4 | 57.8 | 44.7 | sun/night 저하 |
| … Road 97.2 / Sky 95.5 / Vegetation 75.5 / RoadLine 75.2 / Building 74.7 (강건) | | | | | | | |

- **완전사망(전 조건 ≈0)**: Bridge, Water, Wall — 단, **val에선 살아있음** → 학습 자체는 됨, **test 도메인에서만 붕괴**.
- **거의사망**: RailTrack, Dynamic, Ground, Other.
- **thin/rare 약함 + night 민감**: TrafficLight, Static, TwoWheeler, Fence(night 16.2), Truck(night 36.4).

## 3. 모달리티 분석 — "유리해야 하는데 안 쓰인" 정량 (핵심)
| modal | 단독 competence(smiou) | 평균 reliability | 평균 UAMM 가중치 |
|---|---|---|---|
| img   | 26.15 | 0.65 | **0.27** |
| depth | **43.73** | **0.94** | **0.27** |
| event | 16.94 | 0.29 | 0.23 |
| lidar | 15.25 | 0.36 | 0.23 |

- **depth가 압도적으로 유리**(43.7, img의 1.7배·event/lidar의 2.6배). reliability도 depth를 최고(0.94)로 정확히 매김.
- **그러나 융합 가중치는 균일**(config `AMF_MODE: uniform`) → depth와 img에 **동률 0.27**. 즉 **competence·reliability가 출력 융합에 전혀 반영되지 않음.**
- **misallocation = 51.6%** (993/1897 → ep108 재확인): `best_modal`(단독 최적)은 **depth가 1892/1897**인데, `top_uamm`(융합 최대가중 모달)은 **img 982 vs depth 915** → **절반 이상의 이미지에서 가장 유리한 depth 대신 img가 top-weight**.
  - 직접 증거: TrafficLight처럼 depth competence가 높은데도 융합이 못 살리는 케이스(패널 `pred:depth` vs 최종 `Pred` 대조에서 확인).
- **event/LiDAR는 competence ≈15-17로 사실상 죽음** — pred:event/lidar 패널은 노이즈, featPCA:lidar는 구조 없이 뭉개짐. 어떤 soft 라우팅도 정보가 없는 모달을 못 살린다.

## 4. 제안 모듈(corroboration) 효과 — 출력 공간 측정
> 반환 융합피쳐 `m_feat`는 corroboration ON/OFF에 **완전 동일**(max|Δ|=0). 제안 모듈은 memory-attn bias를 통해 **디코더 출력 logit**에만 작용(ablate: `core.corroboration_bias=False`, lambda_bias=0.52 활성).

| 지표 | 값 | 해석 |
|---|---|---|
| Δlogit 평균 | **0.25** | 신호는 확실히 주입됨 |
| argmax flip 비율 | **0.046%** | 결정경계를 거의 안 넘음 |
| ΔmIoU (ON−OFF) 평균 | **−0.013** (median −0.007) | net 이득 없음(미세 손해) |
| helped(>+0.1) / hurt(<−0.1) / neutral | 61 / 136 / **1700** | 89.6% 무변화, 손해가 이득의 2배 |

- **결론**: corroboration은 "누가 신뢰되나"를 무학습으로 잘 맞히지만(AUROC 반전, Phase 0/doc 24), **그 신호가 결정에 도달하지 못한다**. soft attention-bias는 (a) 균일 UAMM 출력융합에 눌리고, (b) event/LiDAR feature가 비어 살릴 정보가 없다. → **신호 품질 ≠ 라우팅 이득** (ep40 진단을 best weight·전체 test에서 정량 재확인).

## 5. 근본 원인 종합
1. **비적응 융합(uniform UAMM)** → competence(depth 43.7)·reliability(depth 0.94)를 무시, misalloc 51.6%. **가장 큰 즉효 레버.**
2. **event/LiDAR competence≈15-17** → soft bias로 부활 불가. feature/decoder에 정보가 없음(drop-Δ≈0).
3. **corroboration이 결정에 도달 못 함** → bias 권위 부족 + 비보정 디코더(event/lidar self-entropy 0.29/0.36 anti-calibrated) 위에서 작동.
4. **도메인 전이 붕괴** → Wall/Bridge/Water/TrafficLight는 val 생존/test 사망. 용량이 아니라 day→night/test 일반화 문제(thin·geometry 클래스 집중).

## 6. 산출물 & 재현
- 데이터: `/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/{panels/,per_image_ALL.csv,stats.json,per_class_matrix.md}`
- 도구: `tools/viz_features_full.py` (`--split test --num -1` 전체, corroboration ON/OFF diff, per-image CSV). ep108 ckpt는 학습 로테이션 삭제 방지 위해 `/home/gm_huis/p32_best_ep108.pth`로 복사 후 사용.
- per-image CSV 컬럼: `miou, miou_off, dmiou, iou_<25cls>, smiou_<modal>, rel_<modal>, uamm_<modal>, best_modal, top_uamm_modal, misalloc, corrb_dlogit_mean/rel, corrb_frac_flipped`.

→ 개선 설계는 [`2026-07-07_P33_design.md`](2026-07-07_P33_design.md) 참조. (목차: [`../00_EXPERIMENT_LEDGER.md`](../00_EXPERIMENT_LEDGER.md))
