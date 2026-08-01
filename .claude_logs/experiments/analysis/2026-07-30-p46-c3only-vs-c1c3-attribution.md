# P46-CTR C3-only vs C1+C3 — 기여 귀속 판정 (ep40, test-SOTA 예비 도달)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 비교표 (ep40 중간 체크포인트, 전부 동일 프로토콜/env)

| 구성 | 해상도 | RailTrack test | Overall test mIoU |
|---|---|---|---|
| Base (P39.1) | @768 | 4.02 | 52.47 |
| C1+C3 (C1 RCS + C3 PROTO) | @768 | 59.10 | 54.92 |
| C1+C3 | @1024 | 60.14 | 56.12 |
| **C3-only (C3 PROTO 단독)** | @768 | 64.13 | 55.64 |
| **C3-only** | @1024 | **64.41** | **56.82** |
| DGFusion (SOTA) | (native-res) | 64.47 | 56.71 |

- ckpt: 두 실험 모두 ep40 중간 체크포인트(val 67.36@C1+C3, val 66.98@C3-only) — **완주(ep200) 전.**
- 게이트(사전등록: RailTrack test 4→≥40)는 **두 구성 모두, 양 해상도 모두 압도적으로 통과**.
- eval: `tools/eval_reliadino_ckpt.py`(+ `--drop-modality` 계열과 무관, 순수 체크포인트 fwd), lecun idle GPU, checkpoint load missing=0/unexpected=0 전부 확인.

## 판정 (코디네이터)

- **C-3(prototype consistency) 단독이 핵심·충분 기제다.** C3-only가 RailTrack·overall 양쪽 모두 모든 해상도에서 C1+C3보다 높다 — C-1(RCS)을 빼는 것이 오히려 결과를 개선시킨다는 뜻이므로, **C-1(RCS)은 이 조합에서 순유해로 판정한다**(앞선 jarvis GPU0-3 kill 판정과 정합).
- **Overall test-SOTA 예비 도달**: C3-only @1024 overall test **56.82 > DGFusion 56.71(+0.11)**. RailTrack도 64.41로 DGFusion(64.47)과 사실상 동률(−0.06).

## 🔴 주장 신중 — 검증 필요 3가지 (아직 확정 아님)

1. **완주 + val-best 재판정**: 이 수치는 **ep40 중간 체크포인트**다. 학습은 ep200까지 계속 진행 중이며, 완주 후 val-best ckpt로 다시 측정하기 전까지는 "SOTA 돌파"를 논문/보고에 확정 서술할 수 없다.
2. **프로토콜 정합성**: DGFusion은 **final-iter** 체크포인트(BestCheckpointer 없음, seg-report-sota-gap 컨벤션 참조)인 반면 우리는 **@768 학습 후 @1024로 eval**한 결과다 — 해상도 mismatch가 있다. 동일 조건(우리도 재현한 DGFusion 56.73 수치)과 비교하면 격차는 **+0.09**로 더 좁혀진다(56.82 vs 56.73) — 방향은 유지되나 마진이 매우 얇다.
3. **RailTrack val<test 역전 미해명**: C1+C3·C3-only 모두에서 RailTrack **val(18~20대)이 test(59~64)보다 낮은** 역전이 일관되게 재현되고 있다. 원인 미상 — 데이터 분포차 추정이나 확인 안 됨. 이 역전이 무엇을 의미하는지 이해하기 전까지 결과 해석에 유보가 필요하다.

## 후속 검증 실험 (착수)

- **[LAUNCH-C2C3]** hpca100 A100×2(GPU2,3): C1 off + C2 MCC on + C3 PROTO on(cross_view=true) — all-on의 C2 기여를 C1 없이 격리(40GB라 보조 branch 수용 가능, all-on이 4090서 OOM났던 문제 회피).
- **[LAUNCH-C3SEED2]** jarvis GPU1-3: C3-only의 재현성 검증(seed만 변경).

두 실험 결과가 나오면 C1/C2/C3 삼자 기여 분해가 완성되고, C3-only 재현성도 확인된다.
