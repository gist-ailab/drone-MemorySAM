판정은 이 파일이 아니라 `lab-plan verdict DRN-261027-02 --analysis .claude_logs/tasks/DRN-261027-02/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-04. 요청 inbox `20261027-170333-request-184c7258`(method). 학습 없음.

## 측정 조건
- GPU: jarvis RTX 4090 1장, 측정 전후 같은 GPU 의 다른 compute 프로세스 0개 확인(우리 모델 = 도구 내부 기록 `latency_valid: true`, 기준선 = `nvidia-smi --query-compute-apps` 전후 스냅샷, `cost_table_20261104c/gpu_procs.log`).
- 입력: DELIVER test 첫 장, 4모달(RGB·Depth·Event·LiDAR) 1024², batch 1. bf16 autocast, no_grad. 지연 = 웜업 20회 후 100회 CUDA Event 평균. 메모리 = `torch.cuda.max_memory_allocated`(웜업 후).
- FLOPs: 모든 모델 같은 카운터 `torch.utils.flop_counter.FlopCounterMode`, 곱-더하기 1회 = 1 flop 로 환산(total/2). 카운트 전 파라미터 requires_grad 끔(A2 의 AssertionError 원인). fvcore 는 우리 모델에서 jit 추적이 실패해 쓰지 않는다.
- 학습 파라미터: 우리 모델 = 빌더가 남긴 requires_grad(백본 동결, LoRA·융합·헤드), 기준선 = 학습 config 기준 requires_grad(전체 학습).
- 도구: `tools/cost_table_ours.py`, `tools/baseline_failure/cost_table_d2.py`(develop a4f404a). 기준선은 jarvis `/SSDb/jemo_maeng/dgfusion_train` 저장소(env dgfusion)에서 실행.
- 가중치: A2 s821(md5 f05b54d4…), E1 s1(c648f925…, Q2 와 같은 구조라 Q2 값 겸용), DGFusion (b) `output/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade/model_final.pth`, CAFuser (b) NAS `ckpts/cafuser_b_resume_20261003/model_final.pth`(fbae170d…). 비용은 구조로만 정해지므로 기준선 공식 가중치와 같은 값이다.

## 사실 — 비용표
| 모델 | 총 params | 학습 params | GFLOPs(1 MAC=1) | 지연 ms/img(평균 ± std) | 최대 메모리 GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2(E1 구조 + 고해상 경로) | 377.05M | 73.98M | 10,135 | 254.5 ± 0.8 | 2.47 |
| Q2 = E1 구조 | 361.87M | 58.79M | 9,922 | 224.8 ± 1.3 | 2.18 |
| DGFusion (Swin-T) | 78.88M | 78.88M | 695.6 | 88.9 ± 2.0 | 1.83 |
| CAFuser (Swin-T) | 75.01M | 75.01M | 670.8 | 83.1 ± 2.1 | 1.57 |
- 원본: NAS `analysis_logs/cost_table_20261104/`(A2·E1 지연 유효본, 01:41~42) · `cost_table_20261104b/`(4모델 같은 FLOPs 카운터) · `cost_table_20261104c/`(기준선 지연 유효본, 03:41~42).
- A2 − E1 = 15.18M = 고해상 경로 파라미터(학습 로그 `HIRES params=15,184,512` 와 일치), FLOPs +213 G(+2.2%), 지연 +29.7 ms(+13%).

## 원인 해석 (검증 여부 표시)
- [사실] 우리 모델의 비용은 동결 ViT-L 백본을 센서 4개에 각각 1024²(토큰 4,096개)로 통과시키는 데서 대부분 나온다. FLOPs 는 기준선의 약 14.6배, 지연은 약 2.9배다.
- [사실] 학습 파라미터는 우리(73.98M)가 기준선(75~79M)과 같은 수준이다. 총 파라미터의 약 80% 가 동결 백본이다.
- [미검증] 지연 비가 FLOPs 비보다 훨씬 작은 것(2.9배 vs 14.6배)은 ViT 의 큰 행렬곱이 GPU 를 더 효율적으로 쓰기 때문으로 보인다.
- 측정 이력: 11-03 1차(GPU2)와 11-04 03:06 재실행(GPU6)은 같은 GPU 에 다른 사용자 프로세스가 있어 지연을 무효 처리했다(03:06 은 이 세션이 빈 GPU 확인 결과를 보지 않고 띄운 실수). 위 표의 지연은 모두 독점 측정분이다.

## 판정 제안
판정 없음(비용 기록 항목).

## 다음 결정
- 논문 표에 넣을 때 FLOPs 규약(1 MAC = 1 flop, FlopCounterMode)과 GPU·bf16 조건을 캡션에 적는다.
