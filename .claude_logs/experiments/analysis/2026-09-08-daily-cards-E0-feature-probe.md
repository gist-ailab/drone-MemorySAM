---
created: 2026-09-08
type: 일일 카드 E0 판정 — 특징 정보 프로브(원 백본 특징 vs 어댑터 후 vs 융합 후)
run: bengio GPU4, `tools/probe_feature_info.py`, ckpt = DELIVER P46 C3-only seed821 val-best(ep90), fit=train 400장·클래스당 ≤4000토큰, eval=val/test 각 300장, 탭 블록 6/12/18/24, 선형 프로브
raw: bengio `/SSDe/jemo_maeng/src/drone-MemorySAM-daily/outputs/probe_E0/{probe_table.md,probe_results.json}` → NAS analysis_logs로 회수
---

# E0 판정 — "어댑터가 백본 정보를 버린다"는 전역 가설은 기각, 단 depth 중간층에 fused가 잃는 붕괴 클래스 근거가 남아 있다

## 1. 결과 (선형 프로브 mIoU, %)

| 특징 세트 | val mIoU | test mIoU | test 붕괴클래스 recall 평균 |
|---|---|---|---|
| raw_taps/all (LoRA off, 4탭 concat, 4센서) | 36.86 | 27.81 | 29.6 |
| adapted_taps/all (LoRA on, 4탭) | 39.47 | 32.24 | 32.5 |
| adapted_last/all (LoRA on, 마지막 블록) | 44.14 | 35.59 | 32.4 |
| **fused_prehead** (융합 트렁크 출력) | **50.23** | **45.14** | 37.8 |

순서가 raw < adapted < fused로 단조다. 어댑터(LoRA)와 융합 트렁크는 선형으로 읽을 수 있는 클래스 정보를 **늘린다**. 사전 등록 게이트(i) "원 특징의 붕괴 클래스 recall − fused recall ≥ +10%p"는 **−8.2%p로 미달**.

## 2. 붕괴 클래스별 (test recall, %)

| 특징 세트 | RailTrack | Wall | Water | TrafficLight |
|---|---|---|---|---|
| fused_prehead | 0.0 | 33.3 | 16.6 | 88.9 |
| adapted_taps/depth | **23.9** | **39.6** | **47.6** | 43.7 |
| raw_taps/depth | 6.2 | **40.0** | 31.0 | 44.7 |
| adapted_taps/all | 0.0 | 17.4 | 47.6 | 47.3 |
| raw_taps/lidar | 0.0 | 27.7 | 1.4 | 0.0 |
| adapted_last/img | 0.0 | 15.4 | 9.0 | 87.3 |

Bridge 열은 테스트 서브셋 토큰이 극소(50.0 반복 = 1~2토큰)라 제외. 표본 수는 JSON에 저장되지 않음(도구 개선점).

## 3. 판정

1. **전역 가설 기각**: 원 frozen DINOv3 특징(4탭)이 어댑터·융합 후 특징보다 선형 판독에서 크게 뒤진다(test −17.3). "어댑터가 얕아 백본이 가진 정보를 못 받는다"는 형태로는 성립하지 않는다. 구조 사다리 S3(블록 교환)·S5(별도 인코더)의 전제 "정보가 백본에 있다"는 약해졌다 → **E5·E6 우선순위 하향**.
2. **국소 근거는 실재**: depth의 **중간층 탭**(adapted_taps/depth)에 fused가 테스트에서 잃는 붕괴 클래스 근거가 남아 있다 — Water 47.6 vs 16.6, RailTrack 23.9 vs 0.0, Wall 39.6 vs 33.3. RGB(img)·lidar·event에는 없다. 즉 잃어버리는 것은 "백본 정보 일반"이 아니라 **depth 중간층의 기하 단서**이고, 그것을 잃는 지점은 어댑터가 아니라 **마지막 층만 읽는 FPN 입력 + 융합**이다.
3. 이 결과는 **E1(4탭 읽기)의 예측**("depth·lidar 의존 클래스에서 이득 집중")과 **E3(센서별 prototype: depth가 클래스 정체성을 독립 지지)**의 근거를 유지·강화한다. E1은 bengio GPU 6,7에서 학습 중(2026-09-08 저녁 완주).
4. 오라클 센서 선택 프로브(H26, 학습형 라우팅 재개 여부)는 `--oracle-dir` 미지정으로 **미실행** — H16 산출물 위치 확인 후 별도 실행 필요. 단 1번 결과로 "원 특징에 라우팅 단서가 풍부하다"는 기대는 낮아졌다.

## 4. 제한

선형 프로브(비선형 판독 가능성 미측정) · eval 300장 서브셋 · 클래스당 4000토큰 상한 · 토큰 = 패치 다수결 라벨 · 단일 시드·단일 ckpt.

## 5. 후속

- E1 완주 시 이 표와 대조(4탭이 Water·RailTrack을 실제로 회복하는지).
- 도구 개선: per-class 토큰 수를 JSON에 기록(다음 실행부터).
- 카드 문서 §1 E0 행 판정: **게이트(i) 미달 → 구조 카드 E5·E6 하향, E1·E3 유지**.
