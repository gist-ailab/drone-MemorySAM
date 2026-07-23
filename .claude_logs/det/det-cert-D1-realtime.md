# Det 공인인증 — D1 실시간(≥5fps) 백본 스윕 (2026-07-23)

목적: 국가 R&D 공인인증 제출용 det 모델 확정. 제약 = **배치 GPU RTX 5080, ≥5fps 실시간**.
"D1" = P37b-ClassToken seg backbone(ReliaDINO/DINOv3) + RF-DETR NMS-free det head.
백본 크기(S/S+/B/L)를 스윕해 5fps를 넘기는 것 중 mAP50 최고를 고른다.

## 1. RTX 5080 fleet 확인 — 없음, 추정치 사용

`ssh <host> nvidia-smi -L` 확인(2026-07-23): hpca100=A100-SXM4-40GB×4, jarvis=RTX4090×8,
yeon/lecun/bengio=RTX3090×7-8. **5080 없음.** 3090 실측(BS1 768², warmup15+측정150,
forward+postproc만, `analysis_logs/det_fps_3090_20260723.json`)에서 스케일 추정.

**스케일 근거** (5080 vs 3090, 공개 스펙):
- FP32: 56.28 vs 35.58 TFLOPS → **1.582×**
- Boost clock: 2617 vs 1695 MHz → 1.544×
- 메모리 대역폭: 960 vs 936 GB/s → **1.026× (거의 평평)** — GDDR7 고속 핀레이트가 256bit
  (3090은 384bit) 좁은 버스로 상쇄됨
- 아키텍처 2세대 도약(Ampere→Ada→Blackwell)

연산-바운드(큰 matmul, ViT-B/L 순전파)는 FP32/tensor 비율(1.58×) 근처로 스케일할 것,
BS1 소형 백본(ViT-S/S+)은 커널런치/메모리레이턴시 비중이 커서 거의-평평한 대역폭 비율
쪽으로 당겨짐 → **1.3×~1.6× 범위**를 채택(최초 가늠 1.8×~2.2×보다 낮음 — 그 범위는
현재 파이프라인이 쓰지 않는 FP8/FP4 텐서코어 활용이나 비현실적으로 대역폭에 안 걸리는
워크로드를 전제해야 나옴).

| 백본 | 3090 FPS(실측) | 추정 5080 FPS(1.3×~1.6×) | 5fps 통과 |
|---|---|---|---|
| ViT-S (`vit_small_patch16_dinov3`) | 7.950 | 10.3~12.7 | **예**, 여유 큼 |
| ViT-S+ (`vit_small_plus_patch16_dinov3`) | 7.379 | 9.6~11.8 | **예**, 여유 큼 |
| ViT-B (`vit_base_patch16_dinov3`) | 4.057 | 5.3~6.5 | **예이지만 여유 얇음** — 실측 시 재확인 필요 |
| ViT-L (`vit_large_patch16_dinov3`, D1 base/D1-recovered) | 1.600 / 1.658 | 2.1~2.6 | **아니오** — 낙관적으로도 미달 |

재현: `python tools/det_fps_bench.py --cfg <cfg> --ckpt <ckpt> --out <out> --gpu 0`
(5080 접근 가능해지면 그대로 재실행 — GPU-무관, 코드 변경 불필요).

## 2. 모델 확정 — D1 ViT-S+ (현재 1위)

| 변종 | 상태 | best AP50 | 비고 |
|---|---|---|---|
| D1 ViT-S | 🟢 완료 + cert eval 완료 | **0.9190@ep10** | cert breakdown: all 0.9164 / night 0.8926 / normal 0.9057 |
| **D1 ViT-S+** | 🟢 **완료 — CERTIFIED** | **0.9205@ep11** (S 대비 +0.0015) | cert breakdown(night/normal/per-class) **대기 — yeon/jarvis 빈 GPU 없음**(2026-07-23 ~15:00 KST 확인, 양쪽 전 GPU 타 학습 점유) |
| D1 ViT-B | 🟡 학습 중(ep16~17/20) | 0.9090@ep6(best-so-far) | S/S+ 대비 열세, ep7~16 0.88~0.91대 진동 — 역전 가능성 낮음. **완주 시 재비교 필요** |
| D1 base(ViT-L) / D1-recovered | 🔴 5fps 제외 | 0.8460 / 0.9321@ep6 | 정확도 계보 최고지만 fps 미달로 인증 대상 아님 |

**선택**: ViT-S+가 ≥5fps 변종 중 최고 AP50. ViT-B가 완주 후 역전하면 교체.

## 3. 재현 패키지

`/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/code/det_cert/`
— `run_cert_eval.sh`(단일 진입점: mAP breakdown + FPS) + `tools/`(det_eval_breakdown.py,
det_fps_bench.py, det_viz_samples.py, _det_common.py) + `configs/`(D1 vits/vitsp/vitb) +
README.md(env·절차·기대수치·eval-scope 정의). ckpt는
`submission/ckpts/det_D1_vits_20260723/`, `submission/ckpts/det_D1_vitsp_20260723/`.

코드 develop 병합: commit `410d803`(D1 configs 4종 + det_fps_bench.py + det_viz_samples.py
yeon 로컬 → 단일출처화) + merge `846dbda`.

## 4. 남은 일

1. yeon/jarvis 빈 GPU 확보 시 `tools/det_eval_breakdown.py --eval-scope predicted --workers 0`으로
   D1 ViT-S+ night/normal/per-class breakdown 실행 (cfg=`det_D1_vitsp_jarvis.yaml`,
   ckpt=`submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth`).
2. D1 ViT-B 완주(ep20) 후 best AP50 재확인 — S+(0.9205) 상회 시 인증 모델 교체.
3. RTX 5080 실기가 fleet에 들어오면 `det_fps_bench.py`로 실측, 추정치 대체.
