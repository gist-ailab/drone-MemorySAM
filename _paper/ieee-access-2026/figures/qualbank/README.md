# qualbank — 정성 그림용 추론 은행

평가 split **전체**의 모델별 예측을 한 번 모아 두고(은행), 이미지별 IoU로 후보 장면을 미리 계산해 두는 시스템이다.
그림에 들어갈 장면을 바꾸려면 추론을 다시 돌리지 않고 `selection.yaml` 한 줄만 고친 뒤 `make.sh figs`를 실행한다.

## 구성
| 파일 | 역할 |
|---|---|
| `registry.yaml` | 데이터셋(DELIVER·MUSES·MCubeS)과 모델 목록: 예측 원본 경로, ckpt·md5, 기대 mIoU, 상태(ready/pending), 조건 그룹 규칙 |
| `qualbank.py` | `status` / `ingest` / `score` / `candidates` / `all` (아래) |
| `selection.yaml` | **사람이 고치는 파일**: 그림마다 어떤 장면을 어떤 열로 넣는지 |
| `make.sh` | `status`, `bank <데이터셋>`, `figs`(그림 + PDF 재빌드) |
| `check_against_analysis.py` | 이 시스템의 후보가 analysis 의 수동 선별(24장면)을 그대로 재현하는지 대조 |
| `../src/make_qual_figs.py` | selection → 그림 PDF·사이드카 JSON |
| `../source_data/qualbank/<ds>_per_image.csv` | 전체 이미지 × 모델의 이미지 mIoU·얇은 클래스 mIoU·Δ (git) |
| `../source_data/qualbank/<ds>_candidates.csv` | 조건별 후보: top 10 · reverse 5 · typical 3 · 얇은 클래스별 6 (git) |

은행 본체(예측 PNG·정답·이미지별 혼동행렬)는 NAS `analysis_logs/paper_qualbank/<ds>_<split>/`에 있다(git 아님).

## 측정 정의 (재구현 없음, `tools/baseline_failure/common.py`)
- 이미지 mIoU = 그 이미지 정답에 있는 클래스만 평균한 IoU. Δ = 우리 모델 − max(`compare` 모델들).
- 같은 시퀀스에서 두 장면을 뽑지 않는다. DELIVER는 날씨(케이스 없음)와 내장 센서 고장 케이스(과노출·저노출·모션블러·LiDAR jitter·event 저해상)를 별도 그룹으로 둔다.
- `score`는 모델별 전역 mIoU를 다시 계산해 `expected_miou`(판정값)와 대조한다. DELIVER: A2 s821 58.61 · Q2 57.73 · DGFusion 공식 56.71 · CAFuser (b) 55.77 모두 일치(2026-11-02).

## 자주 하는 일
```bash
cd _paper/ieee-access-2026
bash figures/qualbank/make.sh status            # 무엇이 은행에 있는지
# 장면 교체: selection.yaml 의 pick 을 바꾸거나 image: 로 직접 지정 → 
bash figures/qualbank/make.sh figs              # 그림 + PDF(build/root.pdf)
# 새 모델·데이터셋: registry.yaml 에 경로를 넣고 status: ready →
bash figures/qualbank/make.sh bank deliver      # 적재·채점·후보 재계산
```
후보를 고를 때는 `<ds>_candidates.csv`(조건별 순위)와 `<ds>_per_image.csv`(전체 이미지)를 본다.
