append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-10-29 | L1(공유 LoRA r16) s821 강건성 측정 기동(MMSAM \| analysis, 요청 inbox 20261028-162006). ckpt yeon `…A2L1shared/…/epoch35_69.5_top1_checkpoint.pth` md5 ff7015ea… → NAS `ckpts/a2L1shared_s821_20261021/`·drone-demo 백업·amphibious-demo(4곳 md5 일치). A2 s903 과 같은 프로토콜(mm_eval_v2 batch 8·seed 0·전수) 8종 + clean legal 덤프(기대 58.84) | amphibious-demo `~/workspace/jemo_maeng/robust_l1_s821_20261029/`(tmux l1_robust_a·l1_robust_b), config `configs/eval/a2L1shared_s821_eval1024_amphibious.yaml`(97c6207) | 비교 = A2 s821 jarvis robust_a2_20261019/s821(batch 4) |
| 2026-11-04 | method 결정(강건성 추가 측정 반영) | analysis.md '추가 측정' 절(NAS analysis_logs/robust_l1_s821_20261029/) | 같은 시드 821 에서 L1(공유 LoRA r16) − A2 = test EMM15 −0.80 · val EMM15 −1.72(단일 센서만 남은 4조합에 손실 집중, val 평균 −5.64), NM 노이즈 +1.9~+2.2, 부분 결측 저하는 L1 이 0.25~0.89 덜 떨어짐. 사전 조건 'test EMM15 Δ ≥ 0' 미충족 → 논문 모델을 공유 LoRA 판으로 바꾸지 않는다(user 2026-11-02 결정과 일치: Access = A2 센서별 LoRA 유지, 공유 LoRA 는 Access 에 기재하지 않고 RA-L 별개 논문 재료). 센서별 LoRA 의 기여 = 단일 센서 폴백. 공유 LoRA 시드 902·903 학습은 하지 않는다. |
