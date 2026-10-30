---
id: 20261030-122055-analysis_request-ce3dc370
to: analysis
from: specialist-paper
type: analysis_request
ref: DRN-261027-01
status: open
created: '2026-10-30T12:20:55.483560'
claimed_by: null
claimed_at: null
done_at: null
note: ''
dedup: paper-qual-dumps-261030
---
## 요청: 논문 정성 그림용 장면 선별 + 전 모델 예측 덤프 (DRN-261027-01, user 지시 2026-10-30)

user 지시: "데이터셋별로 학습한 우리 모델과 비교군 추론을 이어 붙여 그림에 넣는다. **차이가 큰 영상들을 기준으로** 그림을 구성한다." 이 요청은 inbox 20261027-182054(정성 장면 선별)을 대체·보강한다. 판정 불필요, 측정·덤프만.

### 1. 장면 선별 규칙 (모든 데이터셋 공통)
- 이미지별 mIoU 를 모든 모델에 대해 같은 채점(DELIVER = legal v2 축, MUSES = 공식 채점 축)으로 계산한다.
- 차이 Δ = 우리 모델(A2) − max(비교군) 의 **이미지별 mIoU 차이**. Δ 내림차순으로 고른다.
- 조건(DELIVER: sun·cloud·rain·fog·night·low exposure / MUSES: clear·fog·rain·snow × day·night)마다 상위 3장, 같은 MAP·시퀀스 중복 금지.
- **역방향도 같이**: Δ 오름차순(우리가 지는 장면) 조건별 상위 1장 — 한계 그림용.
- 결과 = `qual_scenes_<데이터셋>.csv`: scene_id, condition, map/sequence, 모델별 이미지 mIoU, Δ, 순위, 선택 규칙 문자열(파일 머리 주석). paper 세션이 원고 `_paper/ieee-access-2026/figures/source_data/` 로 옮겨 쓴다.

### 2. 덤프 규약
- 위치: NAS `analysis_logs/paper_qual_<데이터셋>_<YYYYMMDD>/<모델>/<scene_id>.png` (경로 구분자는 `__` 로 평탄화, ISSUE-034 주의).
- 내용: **원시 예측 라벨 PNG**(색칠 전, 0~K−1, ignore=255), 해상도 = 평가 해상도(DELIVER native 1042, MUSES native 1080×1920). 같은 폴더에 `input/`(RGB 및 각 센서 시각화용 원본), `gt/`.
- `manifest.csv`(모델·ckpt 경로·ckpt md5·채점 축·덤프 명령) + 파일별 md5. 정본 보존 규약은 analysis-session-protocol §4.
- hpca100 임대가 10-30 종료 — 그 서버에만 있는 ckpt 는 먼저 옮기고 덤프.

### 3. 데이터셋·조건별 범위와 우선순위
**P1 DELIVER (지금 가능)**
- 모델: A2 3시드 중 대표 시드821(+ 같은 장면 시드902·903 은 선택), Q2(고해상 경로 없음, 시드821), DGFusion 공식 가중치(md5 0f40a641…), DGFusion 재학습 (b), CAFuser 재학습 (b). 기존 `native_axis_baseline_20261016` 의 dgfusion 예측이 (a)·공식 중 어느 판인지 manifest 에 명시.
- 조건 A — 센서 전부(Fig.4 DELIVER 행): 위 선별 규칙.
- 조건 B — 센서 고장(Fig.3): depth 통째 결측 / RGB 통째 결측 / LiDAR+event 결측 / S&P 0.2(채널 공통 위치, 원고 정의). 각 조건에서 **같은 선별 규칙**(그 조건의 이미지별 Δ)으로 상위 3장 + 역방향 1장, 그 장면에 대해 전 모델 덤프.
**P2 MUSES (4모달 A2 s2 완주 후, ETA 10-30 06:20 경과)**
- 모델: A2 4모달 s2, DGFusion 공식, CAFuser 공식(가중치 있으면). split = val 250장(GT 있음)으로 선별·덤프. test 는 GT 비공개라 선별 불가.
**P3 MCubeS (A2 학습 후)**
- 모델: A2, 비교군(StitchFusion 등 공개 가중치 확보 가능 여부부터 확인해 보고). test 102장, 체크포인트 규칙은 user 결정(2026-10-30) 대로 test 선택.

### 4. 보고
- 데이터셋별로 `qual_scenes_*.csv` 경로·md5, 덤프 폴더 경로, 선택된 장면 수, 각 장면 Δ 범위를 inbox 로 specialist-paper 에 보내 달라. paper 세션이 `figures/src/make_qual_figs.py`(이어 붙이기·오류 화소 표시·사이드카 JSON)로 Fig.3·4·7 을 만든다.
