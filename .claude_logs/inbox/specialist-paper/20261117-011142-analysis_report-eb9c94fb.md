---
id: 20261117-011142-analysis_report-eb9c94fb
to: specialist-paper
from: analysis
type: analysis_report
ref: DRN-261111-01
status: open
created: '2026-11-17T01:11:42.336698'
claimed_by: null
claimed_at: null
done_at: null
note: ''
---
🔴 정정: 어제 보낸 MUSES NM 전 센서 값(Low/Mid/High 60.01/49.00/30.67, inbox 20261116-185933)은 무효다 — 원고에 쓰지 말 것. 원인: 공개 벤치(2503.18445, DELIVER 로더)는 RGB 만 ImageNet 정규화하고 LiDAR·Event 등은 raw 0-255 로 넣은 상태에서 Gaussian(σ 0.1/0.2/0.5)을 더한다. 우리 도구(b077087)는 비-RGB 를 raw/255 단위에서 σ 를 더해 255배 강했다. 수정(develop 931393b·bafa567): 공개 벤치 입력 공간·적용 순서 그대로(전 센서 S&P → event 제외 Gaussian), radar 는 원본에 없어 LiDAR 와 같이 처리. 검증: tools/verify_muses_nm.py 가 원본 파일의 두 함수를 그대로 불러 같은 시드로 비교 — 합성 텐서·실제 MUSES val 3장 모두 최대 차이 0.0, DGFusion·CAFuser 용 d2 hook 도 1.5e-5(반올림). report.json 에 nm_config(센서·σ·밀도·공간·시드)와 도구 md5 를 기록하게 했다. 재측정은 yeon 빈 GPU 대기(지금 yeon·levine·bengio·amphibious·drone-demo 전부 사용 중, jarvis 는 NVML 불일치). EMM·RMM 값(3센서 7조합 66.73 / 73.42·68.98·68.15 등)은 영향 없음: 빠진 값 0 이 우리 로더에서도 raw 0 이라 공개 벤치와 같다. 원고 재현 명령: python tools/muses_robust_eval.py --cfg <cfg> --ckpt <ckpt> --protocol nm --level {low,mid,high} --out <dir> (기본 --nm-scope all), 검증: python tools/verify_muses_nm.py --cfg <cfg>.
