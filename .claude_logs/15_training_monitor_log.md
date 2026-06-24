# 학습 모니터 로그 (Training Monitor Log)

> 생성: 2026-06-24
> **이 파일은 `/loop` 모니터 세션이 주기적으로 append하고, 모든 세션이 읽어 분석·판단·개선에 쓰는 공유 로그다.**
> loop 세션의 채팅은 다른 세션에 안 보이지만, 여기 기록된 내용은 `.claude_logs` init 규칙을 통해 전 세션이 공유한다.
> 규칙: ① 매 점검마다 한 줄 timestamped 엔트리 추가(append-only, 과거 줄 수정 금지). ② 이상징후(사망/정체/완료/신기록)는 엔트리 아래 `> ⚠️`로 강조. ③ 학습 종료/사망 시 [01_project_status.md](01_project_status.md) 스냅샷의 해당 트랙도 갱신.

---

## RUN-1 · B200 P28 RBMA (DELIVER)

- **서버/소유자**: B200 (unix user `gm_huis`), repo `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`
- **config**: `configs/b200-deliver_rgbdel_P28_physaug.yaml` (순수 RBMA, AMF_MODE=uniform, λ_bias init 1.0, 4모달 img/depth/event/lidar, 목표 200 ep)
- **출력**: `outputs/MMSamP28/b200_deliver_rgbdel_P28_physaug/DELIVER_CMNeXt-B2_idel/` (`train.log`, `epochN_<val>_topK…pth`, `test_epochN_<test>…pth`)
- **비교 기준**: 직접 경쟁군(Cluster B, test) DGFusion 56.7 / CAFuser 55.6 · 구조적 base(Cluster A) MemorySAM val 65.38 — 자세히는 [12_novelty_and_related_work.md](12_novelty_and_related_work.md).

| 점검 시각(KST) | epoch | Val mIoU | Test mIoU | best | GPU(util/mem) | 프로세스 | 상태 판정 |
|---|---|---|---|---|---|---|---|
| 2026-06-24 ~16:00 | 12 | 57.87 | 50.61 | ep12 | G3-7 활성 | alive (8+4 proc) | baseline. ep8→12 상승 중(val 49→58, test 49→50.6). ⚠️동일 config 중복 프로세스 의심 |

> ⚠️ baseline 시점 관찰: 동일 config 프로세스가 8-proc + torchrun(4-proc) 두 그룹 → 중복 실행/유령 프로세스 확인 필요(같은 SAVE_DIR 덮어쓰기 위험).

<!-- 새 엔트리는 이 줄 위 표에 한 행씩 추가 -->
