# 27 — P32 검증 + P33-v2 개정 (2026-07-08, 멀티에이전트) — 포인터 문서

> **상세 리포트는 옵시디언 볼트에 있다** (NAS canonical, repo 사본 동기화됨):
> - P32 검증: `research_vault/P32_CoRB/P32_정량검증_실패분석_20260708.md`
> - P33-v2 설계: `research_vault/P33_CGMoD/P33_v2_설계개정_20260708.md`
> - 백본 브레인스토밍(Cowork, 검증됨): `research_vault/material/brainstorm_next_arch_20260708.md`
> 동기화/보고서 규약: `research_vault/README.md` §🔄 + `scripts/sync_vault.sh`.

## 30초 요약

- **P32 최종**: val **64.12**@ep98(계보 최고) / test **55.00**(P28 55.27에 −0.27 미달). 목표 갭 val −2.39/test −1.71.
- **4축 독립 재계산 검증**(1897장 CSV): doc 25 수치 재현. 단 해석 2건 뒤집힘 — ① **misalloc 51.6% = 비용이 아니라 증상**(misalloc=1 이미지가 +6.5pt 높음, p=2e-34; 레버는 argmax 교정이 아니라 가중치 진폭 적응화) ② **event/lidar = 정보부재가 아니라 depth와의 잉여**(어댑터는 작동; 현 competence는 anymodal SOTA 천장 부근).
- **CoRB attn-bias = 유의한 순손해** (ΔmIoU −0.013, Wilcoxon p=4.5e-22, flip 0.046%) — P32 이득을 모듈에 귀속 불가(추론시 꺼도 동일). "신호는 유효(corr_veto AUROC), pre-softmax 주입은 무효."
- **지배 원인 = per-class 전이 붕괴** (Wall/Bridge/Water/TL val생존/test사망, 복구 상한 +7.9pt). **NEW: sun 하드씬**(worst-50 최다 28%, Pedestrian −35pt).
- **P33-v2** (원안 CG-MoD 개정): M0 진단 3종(무학습, SOTA per-class test 삼각측량 포함) → M1 class-transfer 복구(RCS+CLIP-text anchor+MIC식 consistency; night+sun) → M2 dropout+**distillation 필수** → M3 soft gate(top-k 삭제, hinge-entropy, 입력 corr_veto+veto floor) → M4 CoRB attn-bias 제거. 기대 test 56.5~58 / val 65~66. **Global escape: val<65.5 → 카드 A(DINOv3-RBMA) 전환.**

## 기록 정정 (다른 세션 주의)

- doc 25의 "ep108이 P31 첫 추월"은 stale 기준(P31 54.75) — P31 최종 best = **54.85**@ep182. misalloc "993/1897" → **979**/1897.
- "test 55.00@ep154"의 epoch 라벨 미확인(수치 자체는 유효). P31/P32 계보 단일 출처 = 재구조화 브랜치(9501129) `experiments/monitor-log.md`.
- moddiag_P32.json은 ep40·n=100 기준 — M2 게이트 수치는 ep108 재측정 후 확정.
- MULTIAQUA 최고 M-score = **82.10**(재제출), CLAUDE.md의 81.98은 stale.

## 산출물 원시 경로

`/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/` (CSV 1897행+패널) · `P32_eval_20260706/` (moddiag/relauroc/커브) · `P32_phase0_20260705/` (corr_veto AUROC) · `P32_reliability_figs_20260706/` (fig0~5, 볼트 assets에 사본).
