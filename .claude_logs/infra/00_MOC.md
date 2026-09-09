# 🗺 infra/ — MOC (Map of Content)

> 폴더 역할: **서버·환경·인프라** — 원격 실행 매뉴얼과 실행 환경/경로/포맷 기록.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [servers-and-launch.md](servers-and-launch.md) | 원격 서버 레지스트리 & 실험 자동 실행 매뉴얼 — "X를 <서버>에서 돌려줘" 지시 시 **먼저 읽기**. 단일출처=`scripts/servers.conf`, 실행=`scripts/remote_exp.sh` | 13 |
| [environment.md](environment.md) | 실행 환경/명령, 데이터·가중치 경로, 체크포인트 포맷, DDP, B200 튜닝, VRAM 프로브 | 14 |
| [artifact-locations.md](artifact-locations.md) | **산출물 위치 지도** — 어느 실험의 체크포인트가 지금 어디 있는지(서버 / NAS 아카이브). 서버에서 ckpt가 안 보이면 **재학습 전에 먼저 여기를 확인**. 이관·삭제 절차와 복원 명령 포함 | — |
| [watchdog.md](watchdog.md) | 학습 워치독(`scripts/watchdog.sh` cron scan·상태머신·알림) + `servers.conf` launch 정책(`ban:`/`off`) + run manifest — **launch 이후 감시는 세션이 아니라 이것** | — |
