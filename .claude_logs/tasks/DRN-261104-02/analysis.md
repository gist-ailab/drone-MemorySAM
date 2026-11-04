판정은 이 파일이 아니라 `lab-plan verdict DRN-261104-02 --analysis .claude_logs/tasks/DRN-261104-02/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-04. 요청 inbox `20261104-165050-request-461f9124`(method). 학습 없음.

## 측정 방법
- 모델: A2 s821 `epoch40_68.93_top1`(md5 f05b54d42c7dff25841aa3656fcdda12, drone-demo 백업본), 평가 config `configs/eval/a2s821_eval1024_dronedemo.yaml`(= amphibious 1024 legal v2 config 에서 데이터 경로만 바꿈). DELIVER test 1,897장 전수.
- 도구: 새 `tools/viz_hires_pyramid.py`(develop, 작성 codex2·검수 이 세션). 한 번의 forward 에서 픽셀 헤드의 s4 입력을 공급원별로 잡는다: (b) SimpleFPN `fpn(fused)[0]`(forward 훅) · (c) 탭 `_p43_taps[0]` 을 s4 로 bilinear · (d) 고해상 경로 `_hires_feats[0]` · (e) 헤드 입력 s4(forward pre-hook). **분해 검증: (b)+(c)+(d) − (e) 최대 상대오차 0.0**(정확히 일치) — detail·boundary_refine 는 이 config 에서 꺼져 있다(도구가 확인 후 거부).
- H-V1 지표: 각 공급원의 채널 L2 노름 맵을 native GT 크기로 bilinear 확대 → 얇은 4클래스(Pole·TrafficLight·Pedestrian·Static) GT 화소 평균 노름 ÷ 나머지 유효 화소 평균 노름. 영상별 비율의 평균과 전체 화소 풀링 둘 다.
- H-V2: 같은 ckpt 로 `hires_infer_mode='off'` 를 대입해 평가한 legal v2 test 덤프(DRN-261020-01 에서 이미 측정: `--hires off`, on 재현 58.6139 = 등록 58.6138). 24클래스 = RailTrack 제외.
- 실행: drone-demo RTX 5090(venv `dms` = MMSS_SAM + timm 1.0.30), 약 30분. 원본: NAS `analysis_logs/hires_viz_a2s821_20261104/`(stats_summary.json, stats_per_image.csv, panels/, run.log, 실행 스크립트·장면 목록·H-V2 집계 스크립트).

## 사실 — H-V1 노름 집중도(test 1,893장, 얇은 클래스가 있는 영상)
| 공급원 | 풀링 비율 | 영상별 평균 | Pole | TrafficLight | Pedestrian | Static |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (b) SimpleFPN s4 | 1.001 | 1.000 | 1.00 | 1.00 | 1.00 | 1.01 |
| (c) 탭 s4 | 0.927 | 0.941 | 0.92 | 0.89 | 0.95 | 0.94 |
| **(d) 고해상 경로 s4** | **2.522** | **2.887** | 2.89 | 2.85 | 2.30 | 1.93 |
| (e) 헤드 입력 합 | 2.103 | 2.369 | 2.38 | 2.35 | 1.97 | 1.66 |
- 조건별(풀링): (d) cloud 2.58 · fog 2.46 · night 2.57 · rain 2.46 · sun 2.55, (c) 0.92~0.93, (b) 1.00 — 조건에 따라 거의 변하지 않는다.
- **사전 기준(d ≥ 1.5, (b)(c) < 1.2) 문자 그대로 충족.**
- 🔴 (b) 의 1.00 은 구조적이다: SimpleFPN 각 레벨이 `LayerNorm2d(out_dim)` 로 끝나(`semseg/models/reliadino/encoder.py:408-410`) 화소별 채널 노름이 거의 상수다. 따라서 (b) 비율은 "FPN 이 얇은 구조에 집중하지 않는다"의 증거가 아니라 이 지표로 볼 수 없는 값이다. 비교로서 의미 있는 것은 (c) 0.93 vs (d) 2.52 이다. 패널의 (b) 노름 맵도 단색이고, (b) 의 정보는 PCA 패널에서만 보인다.

## 사실 — H-V2 추론 시 고해상 경로 끄기(test, legal v2)
| 묶음 | on | off | Δ(on−off) | recombine(dense off·질의 on) | Δ(on−rec) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 얇은 4 평균 | 54.40 | 33.95 | **20.45** | 49.39 | 5.01 |
| 나머지 20 평균(RailTrack 제외) | 59.54 | 50.30 | **9.24** | 56.06 | 3.48 |
| 24클래스 | 58.68 | 47.57 | 11.11 | 54.94 | 3.74 |
- **얇은 4 Δ / 나머지 Δ = 2.21 → 사전 기준(2배 이상) 충족.** recombine 에서는 1.44 배.
- 클래스별 off 하락 상위: RoadLine −37.59 · Pole −35.37 · TrafficSign −25.59 · Pedestrian −25.58 · Fence −21.92 · GroundRail −18.10 · TwoWheeler −16.23 · TrafficLight −12.88. 얇은 4 중 Static 은 −7.99 로 작다. 얇은 4 밖의 RoadLine·TrafficSign·Fence·GroundRail 도 크게 떨어진다(가늘거나 작은 구조).
- 🔴 off 는 학습 때 본 적 없는 결선이다(mIoU 58.61 → 47.97, 전 클래스가 떨어짐). 따라서 Δ 에는 "고해상 경로가 공급하던 정보 손실"과 "분포 밖 입력에 대한 헤드 오작동"이 섞여 있다. recombine(덜 극단적)에서는 비율이 1.44 로 기준 미만이다.

## 패널(6장)
- 장면: 정성 은행 후보(`_paper/.../qualbank/deliver_candidates.csv`) top 의 cloud·fog·night 각 1·2위. 요청의 "Pole·TrafficLight·Pedestrian 포함" 조건으로 고르지 않았다(후보 상위를 그대로 사용).
- NAS `panels/img/<cond>/test/<seq>/<frame>_rgb_front/` 에 rgb·gt·{b,c,d,e}_{norm,pca}.png·grid.png(2×5).
- [사실, 1장 육안] night MAP_10_point5/001200: (d) 노름이 차선·가로등 기둥·자전거·창틀 윤곽에서 밝고 노면·벽 내부는 어둡다. (c) 는 16px 격자 흔적이 보이는 흐린 맵이다. (d) PCA 에 간판 글자까지 보인다.

## 사전 기준 대비 요약
| 가설 | 기준 | 결과 |
| --- | --- | --- |
| H-V1 | (d) ≥1.5, (b)(c) <1.2 | (d) 2.52 / (c) 0.93 / (b) 1.00 → 충족. 단 (b) 는 LayerNorm 으로 지표가 무의미 |
| H-V2 | 얇은 4 Δ ≥ 나머지 Δ × 2 | 20.45 vs 9.24 = 2.21배 → 충족(off). recombine 은 1.44배 |

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- 4번(extractor 샘플링 오프셋 패널)은 하지 않았다(선택 항목).
- 원고에 쓸 때 (b) 노름 맵은 LayerNorm 때문에 단색이므로 노름이 아니라 PCA 패널로 보이거나, 노름 비교는 (c)·(d) 만 쓰는 것을 권한다.
- 패널 장면을 "얇은 클래스 포함" 기준으로 다시 고르려면 GPU 없이 장면 목록만 바꿔 `--mode panels` 재실행(약 30분, 전수 패스 필요).
