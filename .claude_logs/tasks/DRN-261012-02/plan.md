착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# A2 · 고해상 경로(SPM conv 스템 + deformable extractor, injector 없음) — 1024 유지, Q2 레시피 40ep 스크린 2시드

- id: DRN-261012-02
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: 16px 토큰 해상도 한계를 입력 해상도를 올리지 않고 고해상 경로로 메울 수 있는가(DINOv3 공식 분할 결선)
- 가설: 우리 1/4 특징은 1/16 토큰의 보간뿐이다(decisions/2026-10-11-dinov3-official-seg-vs-ours.md). conv 스템이 만든 1/4·1/8 특징을 쿼리로 삼아 ViT 탭 토큰을 deformable attention 으로 샘플링하면(주 경로, 게이트 잔차 아님) 패치 이하 두께 성분의 '찾았지만 못 그림'이 줄어든다. E17(작은 conv 가지를 tanh 게이트로 합산)과 다른 점 = 의미를 끌어오는 extractor·주 경로·폭
- 바꾼 변수(change): SimpleFPN+탭 합산 → SPM(센서별 경량 conv 스템, 합산) + extractor(conv 쿼리 1/8·1/16·1/32 ← 4탭 deformable attn) + c1=up(c2)+conv1/4. 백본·LoRA·헤드·레시피 불변, 입력 1024
- 기준선: Q2 3시드 legal v2 1024(57.73/55.98/56.07) 및 A1 결과(상한 대조)
- 성공 기준: 2시드 평균 1024 legal v2: 24클래스 Δ vs Q2 ≥ +0.5 그리고 얇은4 평균 IoU Δ ≥ (A1 이 같은 해상도 대비 얻은 얇은4 이득의 절반) — A1 미완 시 +1.5. 두께 0.25~0.5 패치 '찾았지만 못 그림' ≤ .35(Q2 .445)
- 반증 조건: 얇은4 평균 Δ < +0.5 또는 두께 0.25~0.5 패치 '찾았지만 못 그림' 감소 < .05 면 기각(고해상 경로는 토큰 해상도를 대신하지 못함)
- configs: configs/lecun-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hires.yaml, configs/lecun-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2hires.yaml

## 정정
