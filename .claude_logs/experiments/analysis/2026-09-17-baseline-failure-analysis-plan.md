---
created: 2026-09-17
author: fable (background 세션 "MMSAM | 생각정리") — 설계·판정 담당
status: 🟡 설계 완료 — 집행 위임(감시 세션: 덤프·집계 = sonnet, 분석 코드 = labcode), 판정은 이 세션
depends: registry `dgfusion_swin_tiny_bs8_200k_deliver_clde` · `cafuser_swin_tiny_bs6_267k_deliver_clde_lecun` · decisions/2026-09-17-strategy-exploration-moe-lora-fusion-training.md
---

# DGFusion·CAFuser 재학습본 vs 우리 모델 — 평가셋 전수 실패 사례·특징 추출 분석 설계 (2026-09-17)

> **user 지시(2026-09-17)**: "depth 보조 감독을 사용하더라도 다른 노벨티 라인이 있어야 한다. CAFuser·DGFusion 재학습을 맡은 쪽에 평가 데이터를 전부 뽑아서 특징이 어떻게 추출되는지, 어떤 상황에서 failure case가 나오는지 분석을 진행하라."
>
> **목적**: depth 유도 융합(DGFusion)·조건 유도 융합(CAFuser)·동결 VFM + 센서별 LoRA(우리) 세 모델을 **같은 평가 이미지 위에서 이미지별·클래스별·조건별로 대조**해, (1) 기준선이 이기는 상황과 우리가 이기는 상황을 분리하고, (2) 기준선의 내부 기제(depth 토큰·조건 토큰·cross-attention 가중)가 어느 조건에서 실제로 작동하는지 측정하며, (3) 그 대조에서 **depth 보조 감독과 독립인 노벨티 라인**의 후보와 반증 가능한 예측을 뽑는다.

## 0. 대상 모델·체크포인트·평가셋

| 모델 | 체크포인트 | 위치 | 비고 |
|---|---|---|---|
| DGFusion Swin-T (재학습) | `model_0079999.pth`(val-best 80k, test 55.68) **및** `model_final.pth`(200k, test 55.56) | yeon `/SSDb/jemo_maeng/dgfusion_train/output/dgfusion_swin_tiny_bs8_200k_deliver_clde/` | 80k 이후 bf16(보고 의무). 둘 다 덤프해 체크포인트 선택 효과를 분리 |
| CAFuser Swin-T (재학습, DGFusion 대조군) | `model_final.pth`(val 66.04) + val-best(있으면) | lecun `/SSDb/jemo_maeng/cafuser_train/output/cafuser_swin_tiny_bs6_267k_deliver_clde/` | bs6·3 GPU 파생 설정(보고 의무) |
| 우리 ReliaDINO | E1 확정 시드1 val-best(24클래스 55.26 기준) **및** 우리 최고 단일 런 P46 C3-only ep70(test 56.99) | NAS `ckpts/` (registry 경로) | legal 프로토콜: val.py 1024·BS1·native GT |

평가셋 = DELIVER **val 2005장 + test 1897장 전부**(native 1042×1042 GT). 조건(cloud/fog/night/rain/sun)과 케이스(motionblur/overexposure/underexposure/lidarjitter/eventlowres/none)는 파일 경로에서 파싱한다(DELIVER 표준 구조). 산출물 루트 = NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/baseline_failure_20260917/{model}/{split}/`.

## 1. 산출물 D1~D6

### D1. 이미지별 예측 덤프 (sonnet 집행, labcode가 덤프 코드)
- 세 모델 × 두 스플릿 전체에 대해 **trainID PNG(0~24, ignore 255) 예측 마스크**를 원본 해상도로 저장. 파일명은 원본 이미지 id와 1:1.
- DGFusion·CAFuser: detectron2 `--eval-only` 경로에 argmax 저장 훅을 붙인다(OneFormer `SemSegEvaluator` 입력 시점, 원본 크기로 복원된 `sem_seg` 사용). 공식 test 명령과 **동일 전처리**여야 한다(등록된 test 수치 55.68/55.56과 mIoU가 재현되는지 먼저 확인 — 재현되지 않으면 덤프 무효).
- 우리 모델: `val.py`에 예측 저장 옵션이 있으면 그것을, 없으면 `_unpad_resize_to_orig` 직후 저장. legal 수치(56.99 / 시드1 값)가 재현되어야 한다.
- 추가 덤프(용량 허용 시): 각 모델의 **클래스 확률 최댓값(confidence) 맵**을 uint8로. 실패 사례에서 "확신하며 틀림"과 "모호" 구분용.

### D2. 이미지별 지표 표 (labcode 코드, sonnet 실행)
- 각 모델·스플릿에 대해 CSV 한 장: `image_id, condition, case, [IoU_c for 25 classes], mIoU_img, pixel_acc, [present_c 픽셀수 25]`. 이미지에 없는 클래스는 NaN(평균에서 제외).
- 세 모델을 image_id로 join한 표: `ΔmIoU(ours−DGF)`, `ΔmIoU(ours−CAF)`, 얇은 객체 4클래스(Pole·TrafficLight·Pedestrian·Static)와 큰 영역 3클래스(RailTrack·Wall·Water) IoU 차.
- 전체 mIoU를 이 표에서 재계산해 등록 수치와 일치하는지 검산(불일치 시 D1로 돌아감).

### D3. 실패 사례 채굴 (labcode 코드, 판정은 이 세션)
1. **조건×케이스 행렬**(5×6) — 모델별 mIoU, 얇은 객체 평균, 큰 영역 평균. 세 모델 차이 행렬.
2. **상위 사례 목록**: (a) 우리 ≪ DGFusion 상위 50장 (b) DGFusion ≪ 우리 상위 50장 (c) 세 모델 모두 < 40 (공통 실패) (d) CAFuser와 DGFusion이 갈리는 상위 50장(depth 유도의 순효과가 드러나는 이미지). 각 목록에 조건·케이스·클래스 분포 요약.
3. **혼동 행렬**: 모델별·조건별 25×25 픽셀 혼동(정규화). 얇은 객체가 무엇으로 흡수되는지(Pole→Building? TrafficLight→Vegetation?), RailTrack·Wall·Water가 기준선에서 무엇으로 새는지.
4. **객체 크기 구간별 재현율**: GT 연결 성분의 픽셀 면적을 5구간으로 나눠 모델별 recall. 우리 약점이 "작아서 못 보는 것"(격자 밖 정보 가설)인지 "보고도 다른 클래스로 부르는 것"(클래스 전이 가설)인지 분리한다 — **세부 가지(E17)와 Lovász(E18)의 우선순위를 정하는 근거**.
5. **센서 고장 케이스 민감도**: 같은 조건 안에서 case=none 대비 각 케이스의 ΔmIoU(모델별). lidarjitter·eventlowres에서 기준선(LiDAR·Event를 실제로 쓰는가)과 우리의 반응 차이.
6. **val→test 전이**: 클래스별 val IoU − test IoU를 모델별로. 우리 클래스 전이 붕괴(Wall·Water·Bridge)가 기준선에도 있는지.

### D4. 기제(특징 추출) 측정 (labcode 코드 — 모델 내부 훅)
- **DGFusion**: (i) depth 보조 헤드 예측 vs LiDAR/GT depth 오차(AbsRel·d1) 이미지별 → seg mIoU와 상관(조건별). (ii) 윈도별 **로컬 depth 토큰**과 전역 조건 토큰의 값 분포(조건별 평균·분산, PCA 2D). (iii) 각 레벨 cross-attention에서 RGB 쿼리가 보조 모달 K/V에 주는 **가중 합(모달별)** 을 조건별로 — CAFuser 논문의 "RGB 가중 68%(맑은 낮)→48%(안개 밤)" 재현 여부와 depth 토큰이 그 이동을 얼마나 만드는지(depth 토큰 0으로 치환한 추론과 대조).
- **CAFuser**: 조건 토큰의 조건 분류 정확도(텍스트 대조 감독의 실제 작동), CAA/CA² 모달 가중 조건별.
- **우리(E1)**: `tools/module_diagnostics.py`·`viz_features.py` 그대로 — 모달별 aux 디코더 IoU 조건별, P36 라우터 가중(클래스×모달) 조건별, 트렁크 γ_m, TAPS 탭별 기여(탭 끔 ablation), 모달별 effective rank. **기준선의 "모달 가중 이동"과 우리 라우터 가중 이동을 같은 축(조건)에서 나란히** 놓는다.
- 모달 제거 추론(zero-out) 4종을 세 모델에 공통 적용: 각 모달을 0으로 두고 test mIoU 변화 — 어느 모델이 어느 모달에 실제로 의존하는지.

### D5. 시각화 (labcode 코드)
- D3-2의 각 목록 상위 12장: 패널 `[RGB | Depth | Event | LiDAR | GT | 우리 | DGFusion | CAFuser | 오류맵 3장]`.
- 얇은 객체 실패 대표 6장: 우리 stride-4·stride-16 특징 PCA vs DGFusion 레벨별 특징 PCA — "격자 밖 세부가 없다"가 특징에서 보이는지.
- 조건별 모달 가중 이동 그래프(세 모델 나란히).

### D6. 종합 판정 (이 세션)
- 기준선이 이기는 (조건, 클래스, 크기) 셀과 우리가 이기는 셀의 지도.
- depth 유도가 실제로 기여하는 셀(DGFusion − CAFuser)과, 그 셀을 depth 감독 **없이** 메울 수 있는 기제 후보.
- 반증 가능한 예측을 붙인 노벨티 라인 후보 2~3개(예: "우리 실패의 X%가 면적 < N픽셀 객체에 집중 → 세부 경로가 답", "우리 실패가 혼동 행렬의 특정 쌍에 집중 → 클래스 표현 문제"). 결과는 `experiments/analysis/2026-09-XX-baseline-failure-analysis.md`로, 카드 문서 §5와 노션 실험노트에 반영.

## 2. 집행 분담·검증 기준

| 단계 | 담당 | 검증 |
|---|---|---|
| D1 덤프 코드(3종) + D2·D3·D4·D5 분석 코드 `tools/baseline_failure/` | labcode (`.claude_logs/meta/conventions.md` 검수 파이프라인, 새 코드는 develop 병합 후 실행) | 등록 수치 재현(55.68/55.56/66.04 val/56.99·시드1) ±0.05 이내 |
| D1·D2 실행(yeon: DGFusion, lecun: CAFuser, 빈 GPU 1장: 우리) | sonnet (빈 GPU만, 학습 방해 금지) | 장수 2005/1897 일치, PNG 값 범위 0~24/255 |
| NAS 이관·대장 기록 | sonnet | `du`+파일수 대조, `infra/artifact-locations.md` 갱신 |
| D3~D5 실행·표 생성 | sonnet | CSV 행수·NaN 비율 보고 |
| D6 판정·문서 | 이 세션 | — |

주의: lecun은 `/SSDb`가 100%라 산출물은 `/SSDc`에, yeon도 여유 확인 후. 덤프 PNG 3모델×2스플릿×3902장 ≈ 수 GB. 실행 중 학습이 있는 체크아웃은 pull 금지(코드는 파일 단위 전송 + md5).

## 3. 우선순위와 기대 일정

D1→D2→D3(1·2·4·6)이 핵심이며 2일 안에 나와야 E17/E18 우선순위 판정에 쓸 수 있다. D4는 기준선 내부 훅이라 시간이 더 걸리므로 병렬로 진행하되 D3 결과를 먼저 보고한다.

## 7. 보완(2026-09-18, DGFusion·CAFuser 재학습 담당 세션 제안 5건 + 추출값 1건 — 생각정리 세션 채택 판정)

| # | 제안 | 판정 | 반영 방식 |
|---|---|---|---|
| A1 | test 재추론 불필요(DGFusion 20개·CAFuser 27개 체크포인트의 test 예측 JSON 잔존, val은 마지막 것만) | ✅ 채택(이미 감시 세션이 JSON→PNG 변환으로 D1 완료, 재현 검산 통과 4건) | D1 문면을 "JSON 변환 우선, val만 재추론(DGFusion 80k·CAFuser 170k)"으로 읽는다 |
| A2 | 채점 방식이 다르다: 기준선은 `CMNEXT_EQUIVALENT_EVAL: true`(GT를 1024 최근접 축소), 우리는 native 1042 GT | ✅ 채택 — **이미지별 지표를 두 프로토콜로 모두 계산해 병기** | `per_image_metrics.py`에 `--gt_protocol {native,resized1024}` 추가(resized1024 = GT를 1024 최근접 축소, 예측은 1024 그대로). 🔴 헤드라인 56.99도 같은 이유로 프로토콜 확정 전(§5-34 참조) |
| A3 | 기준선 `dataset_dict` 모달 키는 `CAMERA/LIDAR/EVENT/DEPTH`(주 모달은 `image`에도 중복) | ✅ 채택 | `d2_zero_modality.patch`의 `_bf_keys`를 이 이름으로 수정 |
| A4 | DELIVER의 depth 감독 정답(`depth/`)은 입력 DEPTH(`hha/`)와 같은 원본 → **HHA 입력을 0으로 둔 채 depth 헤드 정확도가 유지되는지** 측정 | ✅ 채택 — D4에 항목 추가. 무너지면 depth 감독의 이득은 "새 정보"가 아니라 정규화 효과이며, 이것이 E23(DGFusion식 depth 감독 이식) +α 도출의 직접 근거 | D4-(iv): DGFusion 80k·final로 HHA=0 추론 시 depth AbsRel·d1과 seg mIoU 동시 기록 |
| A5 | 체크포인트 선택 잡음 분리: 후반 체크포인트 11개(100k~200k)의 test 예측으로 이미지별 IoU를 재서 "항상 실패 / 뒤집힘 / 항상 성공"으로 가르고 해석은 항상 실패에 한정(한 런 내 후반 test 표준편차 DGFusion 0.74·CAFuser 0.47) | ✅ 채택 — D3-7 신설 | 기준선은 JSON 변환 11개, 우리 모델은 확정 런 저장 체크포인트(top-k)로 같은 분류. 임계: 이미지 mIoU가 11개 중 ≥9개에서 데이터셋 중앙값 미만이면 "항상 실패" |
| A6 | depth GT 거리 구간별 IoU(DGFusion−CAFuser 차이가 먼 거리에 몰리는지) | ✅ 채택 — D3-8 신설 | 원본 depth를 5구간(로그 스케일)으로 나눠 픽셀 단위 IoU, 모델별 |

산출물 공유 위치는 user 결정대로 `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917/`(서버 간 연동)로 하고, NAS `analysis_logs/`에는 사본을 둔다. 분담: 감시 세션 = 이미 완료한 D1·D2·D3(test) 유지, 재학습 담당 세션(GLM 실행·자체 검수) = A2~A6 코드 보강과 실행, 생각정리 세션 = 코드 검수·D6 판정.
