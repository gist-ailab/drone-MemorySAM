---
created: 2026-08-06
---

# PQ 첫 측정 — P48 사전등록 게이트 판정 (2026-08-06)

> 판정 = coordinator(opus). 아래 판정 문구는 coordinator가 지정한 그대로 기록한다.

## 0. 측정 대상

- ckpt: `epoch172_82.58_top1_checkpoint.pth` (**P47-MUB D-1**, MUSES 4모달 val 82.58@ep172)
- cfg: `configs/hpca100-muses_rgbelr_P47_d1_dgfproj_4modal.yaml`
- 데이터: MUSES val 250장, `--geometry native` (un-letterbox 후 원본 해상도 채점)
- 코드: `tools/eval_pq.py` / `tools/pq_format.py` (커밋 `e0890d6` 도입, `b6d3da0`에서 `resolve_gt` 폴더 페어링 버그 수정)
- 산출물: lecun `/tmp/pq_d1_out3/report.json` (유일하게 report.json을 생성한 실행 — v1은 채점 전 크래시, v2는 lecun 재부팅으로 소실)

## 1. PQ 표 (report.json 원본값, ×100)

|        | PQ    | SQ    | RQ    | n  |
|--------|-------|-------|-------|----|
| All    | 35.55 | 81.67 | 41.15 | 19 |
| Things | **22.87** | 79.14 | 26.24 | 8  |
| Stuff  | 44.78 | 83.51 | 52.00 | 11 |

## 2. per-class (things 중심)

| class | PQ |
|---|---|
| person | 1.5 |
| bicycle | 1.4 |
| rider | 7.1 |
| motorcycle | 11.3 |
| car | 10.4 |
| truck | 41.8 |
| bus | 51.7 |
| train | 57.7 |

stuff 참고: road 81.3 · sky 64.6 · vegetation 59.0.

**패턴**: 큰/드문 things(train, bus, truck)는 PQ 40~58로 상당히 잡히지만, 작고 흔한 things(person, bicycle, rider, motorcycle, car)는 PQ 1~11로 사실상 0에 수렴.

## 3. 판정 (coordinator 판정 문구, 그대로)

**사전등록 자기반증 게이트(things PQ > 30) 미달 (22.87 ≤ 30) → D2 진단 반증 → P48 설계 폐기.**

근거: 큰 things만 잡고 작고 얇은 인스턴스는 사실상 0 → "쿼리에 잠재된 인스턴스 능력을 감독으로 깨운다"는 전제가 성립하지 않는다(깨울 잠재력 자체가 부재). 필요한 것은 감독 추가가 아니라 **인스턴스 능력의 신설**이며, 이는 P48 제안의 범위(기존 쿼리 경로에 감독만 추가)를 벗어난다.

## 4. 부수 확정

- **PQ 파이프라인 해금**: `tools/smoke_pq.py` 73개 체크 전부 PASS, 그 중 공식 AUPQ 스코어러와 수치 정확 일치(포화 신뢰도 조건에서 AUPQ=96.9, 우리 PQ=96.9) 확인 — 근사/부분구현이 아니라 검증된 실제 스코어러. 250/250 val 이미지 채점 완료.
- **경쟁군 대비**: All PQ 35.55 vs DGFusion PQ-val 58.88 / CAFuser 59.26 → **−23점대 격차, PQ 표는 SOTA 비교 불가 수준**. 논문 헤드라인은 mIoU 기반 유지, PQ는 한계(limitation)로 명시.

## 5. 오보 정정 (철회)

이 세션 중간에 **"things PQ 44.10 / All 52.05 → 게이트 통과"**로 보고된 바 있으나, 이 수치는 `/tmp/pq_d1_out*` 어느 산출물에도 존재하지 않는다:
- v1(`/tmp/pq_d1_out`): 채점 전 `FileNotFoundError`(gt 폴더 페어링 버그)로 크래시, report.json 미생성.
- v2(`/tmp/pq_d1_out2`): lecun 재부팅(`system boot 2026-08-06 15:31`)으로 프로세스·로그·출력 전부 소실, report.json 미생성.
- v3(`/tmp/pq_d1_out3`, 16:18 KST): **유일하게 report.json을 생성한 실행** — All 35.55/Things 22.87/Stuff 44.78.

**"44.10/52.05"는 철회한다.** 유효한 유일 수치는 본 문서 §1의 v3 report.json 값이다.

## 관련 문서

- 제안 원문: [../decisions/2026-08-05-p48-instance-supervision-proposal.md](../decisions/2026-08-05-p48-instance-supervision-proposal.md) (상단에 폐기 판정 표기)
- PQ 코드 도입: 커밋 `e0890d6` / 폴더 페어링 버그 수정: 커밋 `b6d3da0`
