# third_party — 격리된 외부(vendored) 코드 — ⛔ 수정 금지(frozen)

이 폴더의 코드는 **외부에서 가져온(vendored) 데이터셋 devkit**이다.
우리 프로젝트의 활성 학습/평가 경로는 이 코드를 import 하지 않는다(2026-06 확인).
**업데이트/리팩토링 대상이 아니다.** 원본 업스트림을 참조용으로 동결해 둔 것이며, 필요 시
원 저장소에서 다시 받아 통째로 교체한다.

| 디렉터리 | 출처 | 용도 |
|----------|------|------|
| `MUSES/`   | MUSES 멀티센서 devkit (License.pdf 포함, upstream `setup.py`) | LiDAR/Radar/Event 처리·시각화·AUPQ 메트릭. 비챌린지 데이터셋. |
| `MCubeS/`  | MCubeS dataset_visualization | MCubeS 멀티모달 시각화. 비챌린지 데이터셋. |

> 챌린지 데이터셋(MULTIAQUA) 전처리/시각화는 **우리 코드**이며 `MISC/MULTIAQUA_utils/`에 있다(여기 아님).

## SAM2 / SAM3 업스트림은 어디에?
SAM2/SAM3 백본은 우리 모델 코드(`sam_lora_*.py`, `sam3_lora_rbma.py`)와 디렉터리가
얽혀 있어 이 폴더로 물리 이동하지 못한다. 대신 in-place로 동결 경계를 표시했다:
- `semseg/models/sam2/VENDORED.md`
- `semseg/models/sam3/VENDORED.md`
