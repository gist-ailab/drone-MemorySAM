---
created: 2026-08-16
type: C2-MCC A/B 최종 판정 (캠페인 종결)
---

# C2-MCC A/B 판정 — 유해 확정, DELIVER SOTA 경로 종결 (2026-08-16)

> 판정 = discussion 세션(fable), 사전 등록 게이트 적용. 실측 = val.py(c2c3 ep62 val-best ckpt).

## 수치 (동일 측정기 A/B, λ0.1 통제)

| 구성 | test@1024 | test@768 |
|---|---|---|
| C3-only (본run ep70) | **56.99** | 55.91 |
| C2+C3 (c2c3 ep62) | 55.32 | 54.32 |
| **Δ (C2 순기여)** | **−1.67** | −1.59 |

**게이트 적용**: Δ ≤ −0.3 → **유해 확정** (양 해상도 일관).

## 판정

1. **C2-MCC(MIC-계열 masked consistency) 유해** — 원장 **H15 신설·반증**: "UDA에서 실증된 masked consistency가 지도 멀티모달 융합에서도 전이 이득을 준다" → ✗ (지도 세팅·강한 백본에서 역효과 −1.6대). 그 자체로 논문 분석 재료(UDA 기법의 지도 전이 한계).
2. **DELIVER SOTA의 마지막 산술 경로 소멸** — C2+C3 합산 시나리오 폐기. **DELIVER 최종 = P46 C3-only(λ0.1) 56.99** (SOTA −0.36 · 구SOTA no-tradeoff +0.28).
3. **학습전용 손실 축 = C3 단독 종결** — "레시피 통일(C2+C3 양벤치)" 옵션 소멸. 통일 스토리는 확정된 그대로: **단일 추론 아키텍처 + 축진단 학습손실(C3는 클래스축 벤치에만)**.
4. **캠페인 종결 선언**: 성능 축 전 경로 판정 완료 — 적응(H1~H4)·해상도(역발산)·스케일(H12)·구조 전환(H14)·학습손실 확장(H15) 전부 닫힘. **현 자산(DELIVER 56.99 / MUSES 79.788)이 이 스택 세대의 상한**. 잔여 = λ0.1 시드 2런(통계, ~1.5일)뿐. **임계 경로 = RA-L 리라이트 단독.**

원시: yeon drone-MemorySAM-p49 `logs/eval_c2c3_ep62/test{1024,768}.log`

관련: [2026-08-16-p49-1-muses-official-verdict.md](2026-08-16-p49-1-muses-official-verdict.md) · registry `hpca100_deliver_rgbdel_P46_ctr_c2c3` · 원장 H15
