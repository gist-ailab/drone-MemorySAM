# 세션 인수인계 — a830ad4d → 4e9bdc6f (2026-08-06)

> user 지시(2026-08-06): 동일 이름의 모니터링 세션이 2개 존재(4e9bdc6f=유지, a830ad4d=인계 후 정지).
> a830ad4d가 07-30~08-05에 수행한 작업은 대부분 develop에 이미 커밋되어 있다. 이 문서는 **인계 목록**이며,
> 정정된 수치는 코디네이터(user)가 이미 흡수 완료(메모리 정합 완료). repo 문서 기준으로 작성.

## §1 정정된 기준값 (현행)

- **DELIVER SOTA**: **val 69.60 / test 57.35**(MM SAM-adapter, RGB+Depth 2-modal) — 구 기준(68.79/56.71, CAFuser/DGFusion)은 **2위권**으로 강등.
- **P46 legal 최고**: test **55.62~55.69**(val-best/final-iter 기준, test-best 아님) — SOTA(57.35) 대비 **-1.7**, 구 DGFusion 기준(56.71) 대비 **-1.0**.
- **MUSES 4모달 공식 test**: **79.571**(3모달 79.788 대비 **-0.217**) — radar 무익 확정.
- **radar는 야간 특이적으로 유해**: night 4/4 조건 전부 악화(fog_night 최악 -5.37), 주간은 오히려 +0.19. 디코딩 버그 아님(검증됨) — RGB 신뢰도가 야간에 떨어질 때 모델이 최저-정보 모달(radar)로 라우팅하는 것으로 추정.
- **"모달 수↑ = 성능↓" 철회**: 교차-방법론 비교의 confound였음(방법론 차이가 섞임). 통제된 within-method ablation(CAFuser Table IX, DGFusion)은 모달 추가 시 단조 개선을 보임 — radar 유해는 **우리 모델 고유의 결함**이지 MUSES 일반 법칙이 아님.
- (참고, 흡수 완료) DGFusion 수치 3종(79.72/79.49/79.5)은 불일치가 아니라 val/test(리더보드)/paper Table II 반올림의 차이였음.

## §2 인계된 진행중 작업

| 작업 | 서버/GPU | 상태 |
|---|---|---|
| 재평가 11건(`/tmp/finaliter_batch.sh`) | jarvis (GPU1,2,4,5 그룹) | 살아있음(g5 완료, g1/g2/g4 마지막 항목 진행). ⚠️ §4 참조 — 신뢰 전 검증 필요 |
| lam005_2modal | yeon (`drone-MemorySAM-p38` 체크아웃) | 진행 중, ep148, best 66.46@66 |
| lam005_CLE | yeon (`drone-MemorySAM-p46-cle005` 체크아웃) | 진행 중, ep78, best 60.28@30 |
| lam015 | yeon | **현재 미실행으로 보임**(GPU가 lam005_2modal에 쓰이고 있음) |
| normall | jarvis | 진행 중 — modality 정규화 정합성 수정 검증(hpca100 상실로 jarvis 이관) |
| P47-2 img-only arm | jarvis | **ep82/300에서 정체, 원인 미상**(문서에 설명 없음, 살아있는지 미확인) |
| P47-2 all arm | jarvis | **완주**(ep300/300, Best Val 81.93@182) — base(82.35)·게이트(82.62) 모두 미달 |
| MUSES RGB-L 2-modal | jarvis | config만 커밋됨(`ce40624`), **기동 흔적 없음** |
| P47-MUB D-1(LiDAR projection density) | hpca100 | **hpca100 상실**(타테넌트 점유), 이 run의 운명 미확인 |
| P47-MUB D-2(UniBal) | jarvis(추정) | 구현·스모크 완료, 문서상 "실데이터 미기동"이나 위 jarvis 3모달 arm들이 실제 D-2 실행으로 보임 — 문서 간 정합 안 됨 |
| P48(instance supervision) | — | **제안 단계만**, 구현 없음, S1 미실행 |

## §3 미결

- **P48 S1→S2 게이트**: query-only 성능이 바닥이면 제안 보류 — S1(GT connected-component 통계 + query-only mIoU, `p39_dense_off` 토글로 측정 가능·이미 구현됨) 실행 여부 미확인.
- **P48 D2 자기반증 게이트**: things PQ가 30을 넘지 못하면 D2 진단 자체가 틀린 것 — 사전등록됐으나 아직 평가 안 됨.
- **PQ 측정 블로커**: 코드는 완료됐으나 데이터 미비 — MUSES `gt_panoptic`/`gt_uncertainty` 미다운로드. test 스플릿은 GT 비공개로 **구조적으로** 차단(도구가 `--split test` 자체를 거부).
- 🔴 **ISSUE-030 미수정**: `last_checkpoint.pth` 저장이 비원자적(tmp+rename 없음) — 저장 도중 사망(preempt/OOM/SIGKILL) 시 파손, AUTO_RESUME 파손 위험. 특히 preemptible pod(hpca100류)에서 실증된 리스크.
- **ISSUE-031**(완화됨): hpca100 P47-1이 3090/4090용 BATCH_SIZE:1을 A100에서 재프로파일 없이 사용 → GPU util 60%(24.6/40GB, 목표 85-90% 미달). 해당 run은 유지(재시작 리스크 우려)했으나, 향후 A100 신규 기동 전 `torch.cuda.max_memory_allocated()` 프로파일링이 의무화됨.

## §4 검증 필요 — 재평가 11건 수치 불일치

`/tmp/finaliter_batch.sh`(jarvis)가 반환하는 값이 다른 곳(§1 legal test 55.6~55.7 등)과 어긋남:
- **test 수치가 낮게 나옴**: 53.06~53.67 (vs legal 기준 55.62~55.69) — 격차 약 -2~-2.6pt.
- **val 모드는 전부 PARSE_FAIL** — 출력을 못 읽음.

**원인 규명 완료 → ISSUE-032로 등재.** `val.py`의 `evaluate()`(val 모드 함수)에 `@torch.no_grad()`가 누락돼 있었다 — ViT-L 전체 autograd 그래프가 유지된 채 iteration 1에서 100% OOM(ckpt 종류 무관, 통제실험도 동일 재현). `run_test_inference()`(test 모드)는 정상 데코레이션돼 있어 test만 성공했다. 커밋 `c0e413c`로 1줄 수정(develop push 완료). test 수치가 legal 기준(55.6~55.7)보다 낮게 나온 건(53.06~53.67) — final-iter vs val-best 체크포인트 차이 가설이 남아 있음, 재기동 후 같은 런의 두 체크포인트를 나란히 비교해 확인 예정.

## §5 2026-08-06 처리분 (인수 세션이 오늘 집행)

- **P47-MUB D-1**: §2의 "hpca100 상실, 운명 미확인" 기록을 **정정** — 실제로는 **완주 확인됨**(Best Val mIoU 82.58@ep172, Total Training Time 22:15:29, 체크포인트 hpca100 디스크에 온전). val-best ckpt를 jarvis로 rsync(체크포인트+DGF 밀도화 투영 데이터 3.8GB) 후 MUSES test 750장 예측 완료, **제출 zip 스테이징 완료**(`/ailab_mat2/.../submission/muses/muses_P47MUB_D1_4modal_ep172_submission.zip`, 12.49MB, 750/750 PNG·19클래스 전부 예측됨·degenerate 0). 제출 자체는 user 몫.
- **P47-2 img-only arm**: §2의 "ep82 정체, 원인 미상"을 조사 — 프로세스 사망 확인(ps 미존재), 로그가 Epoch[83/300] iter130/750에서 **트레이스백 없이 그대로 멈춤**(mtime 08-04 18:37, ~2일 정지). 에러/OOM 로그 없음 — 외부 SIGKILL 또는 커널 OOM-killer로 추정. **재기동하지 않음**(user 지시 없었음, 판단 대기).
- **ISSUE-030 수정 완료**: `train_reliadino.py`에 `_atomic_save()` 헬퍼(tmp+os.replace) 추가, `last_checkpoint.pth`+topK best 저장 양쪽 적용. 스모크 3건(fresh save/overwrite/mid-write 사망 시뮬레이션) 전부 통과. 커밋 `0bc65f5`.
- **ISSUE-032 발견+수정**: `val.py`의 `evaluate()`에 `@torch.no_grad()` 누락 — val 모드 100% OOM의 단일 원인(§4 참조). 커밋 `c0e413c`.
- **RGB-L 2모달 기동**: `configs/jarvis-muses_rgbl_P39_1_rank_2modal.yaml`(commit `ce40624`), jarvis GPU1-3, 기동검증 4항 PASS, 진행 중(ep4 기준 val 65.91).
- **P48 S1 게이트 실행 완료**: `p39_dense_off`/`p39_query_off` 토글로 DELIVER val 5-weather-condition 측정(C3-only 본 val-best ckpt). **쿼리-온리(dense off) 성능은 base 대비 -1.06~-2.77pt로 바닥이 아님**(catastrophic 아님), **query_off(쿼리 제거, dense 유지)는 거의 0(-0.45~+0.43)** — 기존 "쿼리 경로=dense의 무의미한 복제" 진단과 일치. **PQ 게이트(things PQ>30)는 아직 미해금**(gt_panoptic 데이터 미비, §3 참조) → **S1 결과만으로는 P48 착수를 확정 지지하지 않음, PQ 게이트 대기 상태로 보류**.
- **P46-C3only fair-eval 최종 판정**(양쪽 런 완주): legal 최고 = 본(original) val-best@ep70 @1024 평가 → **val 69.44/test 56.99**. 현행 SOTA(MM SAM-adapter 69.60/57.35) 대비 val -0.16/test -0.36 **미돌파**, 구 DGFusion 기준(66.51/56.71)은 val·test 동시 상회 **no-tradeoff 우위 유지**. seed2 재현성 확인(test -0.59 단일런 편차). 상세 `analysis/2026-08-06-p46-c3only-fair-eval-final.md`.

---
*작성: 2026-08-06, 세션 4e9bdc6f. 근거 = develop 커밋(`8bf37ed`/`fc392d8`/`622aca2`/`610bff9`/`943f57d`/`ade4e54`/`c99b9cb`/`04bc430`/`255a1f0`/`52a3c48`) + `.claude_logs/issues/issues-and-fixes.md`(ISSUE-030/031) + `.claude_logs/status/current.md`/`plan.md` + 서버 실측(jarvis/yeon/hpca100).*
