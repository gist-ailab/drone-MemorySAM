append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-10-29 | 착수(MMSAM \| analysis). 훅: 고해상 s4 = `model.hires.spm.stem` 출력 c1(센서 순서 호출), 블록 6 = `model.encoder` 호출마다 `last_taps[0]`. 통계 = 토큰 노름·고주파 에너지(F−avgpool3×3)²·채널 분산, 이미지 단위 mean/std, 토큰 단위 샘플 256/장. 파티션 = common.parse_condition_case 의 case(MB·OE·UE·LJ·EL) vs none | 도구 `tools/probe_hires_degrade_signal.py`(작성 중, codex) | 실행 lecun GPU2, 워크스페이스 lecun `/SSDb/jemo_maeng/src/dms-analysis-develop`(develop 521fde1), ckpt NAS `ckpts/a2_s821_20261016_073952/`(md5 f05b54d4…) → lecun `/SSDb/jemo_maeng/ckpt/a2_s821_epoch40_68.93.pth` |
