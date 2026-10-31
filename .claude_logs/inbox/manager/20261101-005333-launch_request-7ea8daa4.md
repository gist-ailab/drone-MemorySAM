---
id: 20261101-005333-launch_request-7ea8daa4
to: manager
from: method
type: launch_request
ref: DRN-261017-02
status: open
created: '2026-11-01T00:53:33.979349'
claimed_by: null
claimed_at: null
done_at: null
note: ''
dedup: method-mcubes-c2d2-hpca100-261101
---
user 직접 결정 2026-11-01(method 세션에서): ① hpca100 GPU0·1 을 MCubeS C2D2 1024 2시드(DRN-261017-02, configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed2026082{7,8}_A2c2d2.yaml)에 쓴다 — 동료 양보 규칙의 재가동 지시로 간주. ② /home/jovyan/SSDb/jemo_maeng/dset/places365_pseudo_200k·places365_pseudo_300k(약 300G 추정)는 삭제 허용. 순서: (1) 디스크: places365 두 폴더는 NAS 에 사본이 있는지 확인해 기록(없어도 user 허용으로 삭제 가능, 삭제 전 du 로 크기 기록) → 삭제; robust_0923(13G) 은 NAS analysis_logs/ 로 회수(md5·파일 수 대조) 후 삭제; src/outputs 비-best ckpt 는 NAS nonbest_weights_archive 로 검증 이동. 목표 여유 ≥ 100G. (2) 교사 ckpt 배치: NAS ckpts/hpca100_mcubes_…_E1Mc_seed20260827_20260920/epoch114_58.94_top1_checkpoint.pth → /home/jovyan/SSDb/jemo_maeng/teacher/mcubes_E1Mc_s827_epoch114_58.94_top1_checkpoint.pth, seed20260828 epoch174_58.61_top1 → …/mcubes_E1Mc_s828_epoch174_58.61_top1_checkpoint.pth (md5 대조; base64 경유 규칙). hpca100 체크아웃을 develop 최신(07d1266 이상)으로(실행 중 재개 런이 있는 체크아웃은 pull 금지 → 다른 체크아웃 또는 파일 단위). (3) 기동: s827 을 GPU1 에 즉시, s828 은 재개 런(DRN-261031-01, 194/200) 완주 후 GPU0 에. hpca100 표준 env 4종(PYTHONPATH sam2·HF_HOME·HF_HUB_OFFLINE·cuDNN), 작업 디렉터리 ~/SSDb/jemo_maeng 계층만(/tmp 금지), 웨이트는 drone-demo 백업 규칙. 검증 로그: '[QAF-T] teacher cfg 바탕 = TEACHER_CFG …' + 'missing=0 unexpected=0', trainable 73,964,904, iteration 전진·GPU util, P46_MEM_LOG 첫 스텝. (4) 재개 런 완주 시 ckpt NAS 복사 + 파이프라인 평가 → DRN-261031-01.
