---
created: 2026-08-06
---

# per-class PQ ↔ 인스턴스 밀도 상관 — D2 진단의 직접 확인 (2026-08-06)

> 판정 = 이 세션(opus). 두 독립 측정을 교차한 것이며 새 실험 없음.

## 0. 교차한 두 측정

| 출처 | 내용 |
|---|---|
| `2026-08-06-pq-first-measurement-p48-gate.md` | P47-MUB D-1 ep172 의 MUSES val per-class PQ (native 채점) |
| 2026-08-05 이 세션 실측 | MUSES `gt_panoptic/val.json` segments_info 250장 전수 파싱 → 클래스별 inst/img · singleton 비율 |

## 1. 결과

| class | singleton% | inst/img | PQ |
|---|---|---|---|
| bus | 100.0 | 1.00 | **51.7** |
| train | 100.0 | 1.00 | **57.7** |
| truck | 93.8 | 1.06 | **41.8** |
| rider | 68.9 | 1.47 | 7.1 |
| motorcycle | 66.7 | 1.78 | 11.3 |
| bicycle | 48.0 | 2.05 | 1.4 |
| person | 27.2 | 4.17 | 1.5 |
| car | 17.6 | 5.46 | 10.4 |

**Spearman(singleton%, PQ) = +0.786** · **Spearman(inst/img, PQ) = −0.762**

## 2. 판정 — D2 는 반증되지 않았다. 오히려 확인됐다

D2(클래스단위 타깃이 things PQ 상한을 만든다)의 예측은 *"이미지당 인스턴스가 1개인 클래스는 클래스단위 타깃이 곧 정답이므로 잘 나오고, 여러 개인 클래스는 병합돼 무너진다"* 이다.
**실측이 정확히 그 형태다** — singleton 100% 인 bus/train 은 PQ 51.7/57.7, inst/img 4~5인 person/car 는 1.5/10.4.

## 3. 🔴 게이트 적용 오류 정정

`2026-08-06-pq-first-measurement-p48-gate.md` 는 **인스턴스 감독을 적용하지 않은 현행 모델**(P47-MUB D-1)의 things PQ 22.87 에 대해
"사전등록 게이트(>30) 미달 → D2 반증 → P48 폐기" 로 판정했다. **게이트 적용 시점이 뒤바뀌었다.**

- 제안서의 게이트는 `완주 보조` 행 = **P48 학습(S3) 완주 모델**에 적용되는 사후 반증 게이트다.
- 현행 모델의 things PQ 가 낮은 것은 **반증이 아니라 제안의 전제**(개선 대상)다.
- D2 를 실제로 반증하려면 **인스턴스 감독 없이** things PQ > 33.6 이 나와야 한다(상한 미구속 입증).

⚠️ **오해를 부른 책임은 제안서 쪽에 있다** — falsifiable 문장에 시점("P48 학습 완주 후")이 빠져 있었다. 제안서를 수정했다(같은 날).

## 4. 부수 소득 — 22.87 은 18.33 보다 높다

기존 things PQ 18.33(2026-08-05, overlap_thresh 0.5) → 이번 22.87(native + `resolve_gt` 픽스 후).
**두 값 모두 D2 가 산출한 클래스단위 상한 33.6 아래**이며, 상한이 여전히 구속 중임을 뜻한다.

관련: `decisions/2026-08-05-p48-instance-supervision-proposal.md` §1 D2′ · `2026-08-06-pq-first-measurement-p48-gate.md`
