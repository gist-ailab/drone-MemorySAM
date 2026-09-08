---
title: "볼트 체인지로그 2026-07-10"
tags: [changelog]
created: 2026-07-10
---

# 2026-07-10

- **추가**: `architecture/P33_v2_설계확장_구현상세_20260710.md` — P33-v2 개정안의 미구현 모듈을 구현 수준으로 확장 (세션 "MMSAM | 설계확장", de0b4608).
  - M0 진단 3종 실행 절차(M0-a SOTA per-class test 삼각측량 판정표, M0-c 보정모델 corr_veto AUROC 재측정; M0-b는 종결 처리)
  - 설계-M1 class-transfer 복구 상세: RCS(신규 RCSSampler+통계 캐시), CLIP-text anchor(`mask_tokens.weight[1:26]` 25×256 앵커, 오프라인 임베딩 — 런타임 CLIP 의존 0), masked night/sun consistency(EMA teacher, MULTIAQUA-가드 완화+신규 RandomSunShadow)
  - 설계-M2 잔여 full-modal KD(EMA teacher 재사용), 설계-M3 완전형 학습 게이트(조건부, corr_veto blend 공식 명문화)
  - **명칭 충돌 정리**: 코드 커밋 c441f1d의 "M1/M2/M3"은 설계-v2 번호와 불일치 → canonical = 설계-v2 번호 + 코드 약칭(CF-lite/MD/CAL/CoRB-off)
  - ablation 사다리 재정렬: P33.1(학습중)=CAL+CF-lite → P33.2=+MD → **P33.3(재정의)=+RCS+text-anchor** → P33.4=+consistency+KD → P33.5 조건부
- **갱신**: `00_P33_CGMoD_index.md`(파일 목록·상태·다음 게이트), `00_MOC_26_MultimodalSeg.md`(architecture 섹션 1행)
